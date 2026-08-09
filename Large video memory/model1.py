# -*- coding: utf-8 -*-
"""模型加载 / LoRA 封装 / 统一输入构建（对应 docs/requirements.md 第六章）。

职责边界：
* 只负责模型侧：基座加载、LoRA 应用与存取、prompt 模板（单一来源）、
  训练/推理共用的图文输入构建；
* 不涉及数据扫描/校验（data.py），不在本文件内做数据层面的补丁；
* 所有超参数一律来自 config1.py（函数签名中的 config 参数即调用方传入的
  config 模块本身）。

关于 peft：本文件顶部采用"尝试导入、失败置 None"的懒加载写法，
保证未安装 peft 时（如 processor 冒烟测试阶段）import model 不失败；
调用依赖 peft 的函数时会给出明确的安装提示。
"""

import os

import torch
import torch.nn as nn
from transformers import AutoModelForVision2Seq, AutoProcessor

import config1
from utils1 import ensure_dir, get_logger

try:
    import peft
except ImportError:  # peft 尚未安装（测试阶段才装），相关函数被调用时再报错
    peft = None

logger = get_logger("model")

# ===== 训练/推理共用的指令模板（唯一来源，禁止在 train.py / test.py / data.py 重复定义） =====
# 注意：规格文档 6.5 示例中的 "n结论：xxx型拉米尔n分析：..." 是 markdown 转义错误，
# 实际应为换行符 "\n结论：xxx型拉米尔\n分析：..."。
INSTRUCTION_PROMPT = (
    "请判断图片中是否存在粉色头发的动漫角色。"
    "如果不存在，只输出：未识别到粉毛角色。"
    "如果存在一个或多个粉毛角色，请针对每一个粉毛角色，"
    "结合其服饰、配饰、姿态与整体氛围判断其类型，"
    "严格按以下格式输出：\n结论：xxx型拉米尔\n分析：（简要说明判断依据）"
)

# config1.LORA_TARGET_MODULES 为空时的兜底扫描关键字（规格 6.2）
DEFAULT_LORA_KEYWORDS = ["q_proj", "k_proj", "v_proj", "o_proj", "qkv", "proj"]

_IMAGE_PAD_TOKEN = "<|image_pad|>"


def _resolution_kwargs():
    """config1.IMAGE_SIZE 生效逻辑：None -> 用 processor 默认分辨率；
    设置整数 N -> 固定为 N×N 像素（Qwen2.5-VL processor 用 min/max_pixels 控制，
    训练与推理共用同一套分辨率）。"""
    if config1.IMAGE_SIZE is None:
        return {}
    px = int(config1.IMAGE_SIZE) ** 2
    return {"min_pixels": px, "max_pixels": px}


# ===== 基座模型加载 =====

def load_base_model_and_processor(config):
    """加载本地基座 VLM 模型与 processor（Qwen2.5-VL-3B-Instruct）。

    环境事实（transformers 4.51.3）：
    * 加载类用 AutoModelForVision2Seq，无需 trust_remote_code；
    * 基座权重为 bf16，torch_dtype 固定 bf16（8.6GB 显存下不用 fp32）；
    * attn_implementation 用 "sdpa"（torch 2.2.1 内置），不用 flash_attention_2。
    """
    logger.info("Loading base model and processor from %s ...", config1.BASE_MODEL_NAME_OR_PATH)
    processor = AutoProcessor.from_pretrained(config1.BASE_MODEL_NAME_OR_PATH)
    model = AutoModelForVision2Seq.from_pretrained(
        config1.BASE_MODEL_NAME_OR_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.to(config1.DEVICE)
    logger.info("Base model loaded, device=%s", config1.DEVICE)
    return model, processor


# ===== LoRA =====

def _scan_linear_modules_by_keywords(model, keywords):
    """按"后缀关键字"扫描模型中的 nn.Linear 模块。

    匹配规则：isinstance(m, nn.Linear) 且 (name == kw 或 name.endswith("." + kw))。
    Qwen2.5-VL 视觉 patch_embed 里有个 Conv3d 也叫 proj，因非 nn.Linear 被自动过滤。
    """
    matched = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        for kw in keywords:
            if name == kw or name.endswith("." + kw):
                matched.append(name)
                break
    return sorted(matched)


def apply_lora(model, config):
    """按 config 的 LoRA 超参包装模型（规格 6.2），返回 PeftModel。

    若 config1.LORA_TARGET_MODULES 为空，自动扫描 nn.Linear 并按常见注意力投影
    关键字兜底，最终实际使用的 target_modules 会打印出来供核对。
    """
    if peft is None:
        raise ImportError("peft 未安装，请先执行: pip install peft（训练/推理测试阶段才需要）")

    keywords = config1.LORA_TARGET_MODULES
    if not keywords:
        print("[apply_lora] config1.LORA_TARGET_MODULES 为空，使用兜底关键字: "
              + str(DEFAULT_LORA_KEYWORDS))
        keywords = DEFAULT_LORA_KEYWORDS
    target_modules = _scan_linear_modules_by_keywords(model, keywords)
    print(f"[apply_lora] 实际 LoRA target_modules（{len(target_modules)} 个）:")
    print(target_modules)
    if not target_modules:
        raise RuntimeError(
            "未扫描到任何可 LoRA 的 nn.Linear 模块，请检查 config1.LORA_TARGET_MODULES 关键字: "
            + str(keywords)
        )

    lora_config = peft.LoraConfig(
        r=config1.LORA_R,
        lora_alpha=config1.LORA_ALPHA,
        lora_dropout=config1.LORA_DROPOUT,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    model = peft.get_peft_model(model, lora_config)

    # 训练侧设置：关 KV cache 省显存；按 config 开梯度检查点。
    model.config1.use_cache = False
    if config1.GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
        # LoRA 只训练 adapter 参数，梯度检查点下 embedding 无梯度路径，
        # 必须强制输入 embedding requires_grad=True 才能回传（LoRA+checkpointing 标准做法）。
        model.enable_input_require_grads()
        # 视觉编码器输入（pixel_values）不走 embedding 层，enable_input_require_grads
        # 对视觉侧无效；而 Qwen2.5-VL 视觉 blocks 同样处于梯度检查点内，输入无
        # requires_grad 时整条视觉路径梯度为 None，视觉 LoRA 会完全得不到训练。
        # 标准做法：在视觉 patch_embed 输出处强制 requires_grad（多模态 LoRA 补丁）。
        # 注：此处遍历的是 PeftModel，模块名带 "base_model.model." 前缀，故用 endswith。
        for name, module in model.named_modules():
            if name.endswith("visual.patch_embed"):
                def _require_grad_hook(_module, _inp, out):
                    out.requires_grad_(True)
                module.register_forward_hook(_require_grad_hook)
                print(f"[apply_lora] 已为 {name} 注册 requires_grad hook（梯度检查点兼容）")
                break
        else:
            print("[apply_lora 警告] 未找到 visual.patch_embed，视觉侧 LoRA 在梯度检查点下可能无梯度")

    model.print_trainable_parameters()
    return model


def save_lora_checkpoint(model, save_dir):
    """保存 LoRA adapter 权重到 save_dir（只存 adapter，不存基座）。"""
    if peft is None:
        raise ImportError("peft 未安装，请先执行: pip install peft")
    ensure_dir(save_dir)
    model.save_pretrained(save_dir)
    print(f"[save_lora_checkpoint] LoRA adapter 已保存到: {save_dir}")


def load_lora_checkpoint(base_model, adapter_dir):
    """把 LoRA adapter 权重挂到基座模型上，返回 PeftModel（规格 6.3）。"""
    if peft is None:
        raise ImportError("peft 未安装，请先执行: pip install peft")
    if not os.path.isfile(os.path.join(adapter_dir, "adapter_config1.json")):
        raise FileNotFoundError(
            f"LoRA adapter 目录不存在或缺少 adapter_config1.json: {adapter_dir}"
        )
    model = peft.PeftModel.from_pretrained(base_model, adapter_dir)
    print(f"[load_lora_checkpoint] 已加载 LoRA adapter: {adapter_dir}")
    return model


def merge_and_save(model, save_dir):
    """预留工具：把 LoRA 权重合并进基座后导出完整模型（默认训练流程不调用）。"""
    if peft is None:
        raise ImportError("peft 未安装，请先执行: pip install peft")
    ensure_dir(save_dir)
    merged = model.merge_and_unload()
    merged.save_pretrained(save_dir)
    print(f"[merge_and_save] 合并后的完整模型已导出到: {save_dir}")
    return merged


# ===== 统一输入构建 =====

def _truncate_text_to_max_len(tokenizer, text, max_len):
    """在字符串层面按 token 数截断纯文本（截到 max_len 个 token 内）。

    不能用 processor 的 truncation 参数：图像占位符在 tokenize 之前会被展开为
    N 个 <|image_pad|>（N 由分辨率决定），processor 级截断会从序列末尾切断
    prompt 文本（实测：max_length=128 时 76-token 的 prompt 只剩约 45 token），
    因此必须先截纯文本、再套模板、最后交给 processor 时不传 truncation。
    """
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_len:
        return text
    return tokenizer.decode(ids[:max_len], skip_special_tokens=True)


def build_inputs(processor, images, texts):
    """统一的图文输入构建（规格 6.4，test.py 使用）。

    参数：
        processor: VLM processor
        images:    PIL 图片列表
        texts:     原始指令文本列表（未套模板，如 [INSTRUCTION_PROMPT]）

    流程：texts 先按 config1.MAX_TEXT_LENGTH 做字符串级截断 -> apply_chat_template
    拼成 user 消息（含图像占位）-> processor 编码（padding=True）。
    返回 model.generate 可直接吃的 batch dict（原样透传 processor 返回的字段，
    即 input_ids / attention_mask / pixel_values / image_grid_thw，不做字段改名）。
    """
    truncated = [_truncate_text_to_max_len(processor.tokenizer, t, config1.MAX_TEXT_LENGTH)
                 for t in texts]
    messages = [
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": t}]}]
        for t in truncated
    ]
    templated = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages
    ]
    return processor(
        images=images,
        text=templated,
        return_tensors="pt",
        padding=True,
        **_resolution_kwargs(),
    )


def build_train_batch(processor, images, prompts, targets):
    """训练 batch 构建（训练契约，供 data.py 的 collate_fn 调用）。

    参数：
        processor: VLM processor
        images:    PIL 图片列表
        prompts:   指令文本列表（通常为 [INSTRUCTION_PROMPT] * n）
        targets:   目标文本列表（如 "结论：害羞型拉米尔\n分析：..."）

    labels 掩码推导（prompt 部分置 -100，只监督 assistant 回复）：
    * 有图编码时，文本串里的 <|image_pad|> 在 tokenize 前被展开为
      <|vision_start|> + N*<|image_pad|> + <|vision_end|>（N = 该图 patch 数）；
    * 无图编码（prompt_ids）不展开，占位符仍是 1 个 token；
    * 因此 prompt 部分在有图序列中的长度为：
        len(prompt_ids) - n_prompt（纯文本 token 数）+ n_prompt + n_full + 2*n_prompt
        = len(prompt_ids) + n_prompt + n_full
      其中 n_prompt = prompt_ids 中占位符个数（单图 = 1），
      n_full = 有图序列中展开后的 <|image_pad|> 总个数（单图 = N）。
    * 已用 DL 解释器实测（256x256 图，grid 18x18，merge_size=2，N=81）：
      len(prompt_ids)=98, n_prompt=1, n_full=81 -> prompt_part_len=180，
      且该位置恰为 assistant target 的首 token，无 off-by-one。

    返回 {**inputs, "labels": labels}，所有 tensor 仍在 CPU，由 train.py 搬运到 DEVICE。
    """
    batch_size = len(images)
    full_messages = [
        [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": p}]},
            {"role": "assistant", "content": t},
        ]
        for p, t in zip(prompts, targets)
    ]
    prompt_messages = [
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": p}]}]
        for p in prompts
    ]
    full_texts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
        for m in full_messages
    ]
    prompt_texts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in prompt_messages
    ]

    # 训练文本较短（prompt + target 约 150 token），不做截断以免切断 target；
    # 推理侧的 build_inputs 才按 config1.MAX_TEXT_LENGTH 截断。
    inputs = processor(images=images, text=full_texts, return_tensors="pt", padding=True,
                       **_resolution_kwargs())

    image_pad_id = processor.tokenizer.convert_tokens_to_ids(_IMAGE_PAD_TOKEN)
    labels = inputs["input_ids"].clone()
    for i in range(batch_size):
        prompt_ids = processor(text=[prompt_texts[i]], return_tensors="pt")["input_ids"][0]
        n_prompt = int((prompt_ids == image_pad_id).sum())
        n_full = int((inputs["input_ids"][i] == image_pad_id).sum())
        prompt_part_len = int(prompt_ids.shape[0]) + n_prompt + n_full
        labels[i, :prompt_part_len] = -100
        # padding 位置（attention_mask == 0）也掩码，避免 pad token 计入 loss
        seq_len = int(inputs["attention_mask"][i].sum())
        labels[i, seq_len:] = -100

    inputs["labels"] = labels
    return inputs
