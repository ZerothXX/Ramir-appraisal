# -*- coding: utf-8 -*-
"""单图推理（对应 docs/requirements.md 第九章）。

在 PyCharm 中直接改 TEST_IMAGE_PATH 后运行即可完成
"选一张图 -> 生成 -> 后处理 -> 写出 outputs/Ani_type/{图片名}_result.txt"。

流程：加载基座 -> 按 config.CHECKPOINT_SELECT（"best"/"latest"）加载 LoRA adapter
（目录缺失时回退另一个并提示）-> 用与训练完全一致的 INSTRUCTION_PROMPT 构建输入
-> model.generate（采样参数全部来自 config.GEN_*）-> 轻量后处理 -> 写 txt + print。

后处理原则（规格 3.2 / 9.4 / 11.3）：只做格式修补，不编造模型没有表达的内容。
"""

import os
import re
import threading

import torch
from PIL import Image

import config
from model import (
    INSTRUCTION_PROMPT,
    build_inputs,
    load_base_model_and_processor,
    load_lora_checkpoint,
)
from utils import ensure_dir, get_logger

logger = get_logger("test")

# ===== 待测试图片路径：在 PyCharm 里直接改这一行即可 =====
TEST_IMAGE_PATH = os.path.join(config.INPUT_DIR, "test5.png")

# ===== 模型单例（test.py 命令行与 web/app.py 共用，单一来源） =====
# 3B 模型 + LoRA 加载一次要几十秒到几分钟，web 端多次请求必须复用同一份权重。
# 两个全局变量 + 双重检查锁：并发首次调用（如 web 预热线程与首个请求同时到达）也只加载一次。
_MODEL = None
_PROCESSOR = None
_load_lock = threading.Lock()


def load_model_once():
    """加载基座模型 + LoRA adapter（全局单例，多线程安全，重复调用直接返回）。

    与训练完全一致的加载链：load_base_model_and_processor -> 按 config.CHECKPOINT_SELECT
    选 best/latest -> load_lora_checkpoint -> eval + 重开 use_cache（apply_lora 为省显存
    关闭了 cache，推理必须重新打开，否则 generate 无 KV cache 明显变慢）。
    """
    global _MODEL, _PROCESSOR
    if _MODEL is not None:
        return
    with _load_lock:
        if _MODEL is not None:
            return
        logger.info("加载基座模型 + processor ...")
        model, processor = load_base_model_and_processor(config)
        adapter_dir = pick_adapter_dir()
        model = load_lora_checkpoint(model, adapter_dir)
        model.to(config.DEVICE)
        model.eval()
        model.generation_config.use_cache = True
        _MODEL, _PROCESSOR = model, processor


def predict_image(image):
    """对单张 PIL 图片做完整推理，返回后处理后的鉴定文本（web/app.py 复用此管线）。

    流程与训练完全对齐：build_inputs 用同一份 INSTRUCTION_PROMPT（单一来源，见 model.py）
    -> model.generate（采样参数全部来自 config.GEN_*）-> postprocess_output。
    模型未加载时自动触发 load_model_once，因此 web 端直接调用即可。
    """
    load_model_once()
    image = image.convert("RGB")
    # build_inputs 与训练共用同一份 INSTRUCTION_PROMPT（单一来源，见 model.py）
    batch = build_inputs(_PROCESSOR, [image], [INSTRUCTION_PROMPT])
    batch = {k: (v.to(config.DEVICE) if isinstance(v, torch.Tensor) else v)
             for k, v in batch.items()}

    logger.info("生成中 ...（max_new_tokens=%d, temperature=%s, top_p=%s, do_sample=%s）",
                config.GEN_MAX_NEW_TOKENS, config.GEN_TEMPERATURE,
                config.GEN_TOP_P, config.GEN_DO_SAMPLE)
    with torch.no_grad():
        gen_ids = _MODEL.generate(
            **batch,
            max_new_tokens=config.GEN_MAX_NEW_TOKENS,
            temperature=config.GEN_TEMPERATURE,
            top_p=config.GEN_TOP_P,
            do_sample=config.GEN_DO_SAMPLE,
        )

    input_len = batch["input_ids"].shape[1]
    new_ids = gen_ids[0, input_len:]
    raw_text = _PROCESSOR.tokenizer.decode(new_ids, skip_special_tokens=True)
    logger.info("模型原始输出: %r", raw_text)

    result = postprocess_output(raw_text)
    if not result:
        logger.warning("模型输出为空（只补格式不编造内容，请检查生成参数）")
    return result


def main():
    ensure_dir(config.OUTPUT_ANI_TYPE_DIR)
    if not os.path.isfile(TEST_IMAGE_PATH):
        raise FileNotFoundError(f"测试图片不存在: {TEST_IMAGE_PATH}")

    logger.info("开始推理: %s", TEST_IMAGE_PATH)
    image = Image.open(TEST_IMAGE_PATH).convert("RGB")
    result = predict_image(image)

    stem = os.path.splitext(os.path.basename(TEST_IMAGE_PATH))[0]
    out_path = os.path.join(config.OUTPUT_ANI_TYPE_DIR, f"{stem}_result.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result + "\n")

    print("========== 鉴定结果 ==========")
    print(result)
    print(f"已写入: {out_path}")


def pick_adapter_dir():
    """按 config.CHECKPOINT_SELECT 选择 best/latest；对应目录缺失时回退另一个并提示。"""
    if config.CHECKPOINT_SELECT == "latest":
        order = ["latest", "best"]
    else:  # 未知取值一律按 best 优先
        order = ["best", "latest"]
    for key in order:
        adapter_dir = os.path.join(config.OUTPUT_TRAIN_MODELS_DIR, key)
        if os.path.isdir(adapter_dir) and os.path.isfile(
                os.path.join(adapter_dir, "adapter_config.json")):
            if key != config.CHECKPOINT_SELECT:
                logger.warning("CHECKPOINT_SELECT=%s 对应目录不存在，回退加载 %s",
                               config.CHECKPOINT_SELECT, adapter_dir)
            return adapter_dir
    raise FileNotFoundError(
        f"outputs/train_models 下没有可用 LoRA checkpoint（{config.OUTPUT_TRAIN_MODELS_DIR} "
        "下 best/ 与 latest/ 均不存在或缺少 adapter_config.json），请先运行 train.py")


def postprocess_output(raw_text):
    """把模型原始输出整理成规格 3.2 的三种标准形态之一（只补格式，不编造内容）。

    规则：
    * 输出含"未识别到粉毛角色" -> 只输出这一行；
    * 否则按 "结论：" 切分：恰好 1 段 -> 单角色（"结论：" + 段内容）；
    * 多段 -> 每段前加 "角色N" 序号、段间空行；
    * 模型漏写 "结论：" 前缀时（切分后仍只有 1 段）同样补上前缀，即容错格式化。
    """
    text = (raw_text or "").strip()
    if not text:
        return ""
    if "未识别到粉毛角色" in text:
        return "未识别到粉毛角色"
    segments = [seg.strip() for seg in re.split(r"结论：", text)]
    # 若模型自己输出了"角色N"标记（多角色场景），切分后"角色1"落在首段段首、
    # 其余"角色N"落在上一段段尾，分别剥掉再统一编号，避免标记被拼进结论。
    # 只剥"角色+数字"形式的标记，不会误伤"角色扮演型"这类正常词。
    segments = [re.sub(r"^角色\d+[：:、.\s]*", "", seg) for seg in segments]
    segments = [re.sub(r"\s*角色\d+[：:、.\s]*$", "", seg).strip() for seg in segments]
    segments = [seg for seg in segments if seg]
    if len(segments) == 1:
        return "结论：" + segments[0]
    numbered = []
    for i, seg in enumerate(segments):
        numbered.append(f"角色{i + 1}\n结论：" + seg)
    return "\n\n".join(numbered)


def main():
    ensure_dir(config.OUTPUT_ANI_TYPE_DIR)
    if not os.path.isfile(TEST_IMAGE_PATH):
        raise FileNotFoundError(f"测试图片不存在: {TEST_IMAGE_PATH}")

    logger.info("加载基座模型 + processor ...")
    model, processor = load_base_model_and_processor(config)
    adapter_dir = pick_adapter_dir()
    model = load_lora_checkpoint(model, adapter_dir)
    model.to(config.DEVICE)
    model.eval()
    # apply_lora 为省显存把 use_cache 关了，推理需重新打开（否则 generate 无 KV cache，明显变慢）
    model.generation_config.use_cache = True

    image = Image.open(TEST_IMAGE_PATH).convert("RGB")
    # build_inputs 与训练共用同一份 INSTRUCTION_PROMPT（单一来源，见 model.py）
    batch = build_inputs(processor, [image], [INSTRUCTION_PROMPT])
    batch = {k: (v.to(config.DEVICE) if isinstance(v, torch.Tensor) else v)
             for k, v in batch.items()}

    logger.info("生成中 ...（max_new_tokens=%d, temperature=%s, top_p=%s, do_sample=%s）",
                config.GEN_MAX_NEW_TOKENS, config.GEN_TEMPERATURE,
                config.GEN_TOP_P, config.GEN_DO_SAMPLE)
    with torch.no_grad():
        gen_ids = model.generate(
            **batch,
            max_new_tokens=config.GEN_MAX_NEW_TOKENS,
            temperature=config.GEN_TEMPERATURE,
            top_p=config.GEN_TOP_P,
            do_sample=config.GEN_DO_SAMPLE,
        )

    input_len = batch["input_ids"].shape[1]
    new_ids = gen_ids[0, input_len:]
    raw_text = processor.tokenizer.decode(new_ids, skip_special_tokens=True)
    logger.info("模型原始输出: %r", raw_text)

    result = postprocess_output(raw_text)
    if not result:
        logger.warning("模型输出为空，结果文件将为空（只补格式不编造内容，请检查生成参数）")
    stem = os.path.splitext(os.path.basename(TEST_IMAGE_PATH))[0]
    out_path = os.path.join(config.OUTPUT_ANI_TYPE_DIR, f"{stem}_result.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result + "\n")

    print("========== 鉴定结果 ==========")
    print(result)
    print(f"已写入: {out_path}")


if __name__ == "__main__":
    main()
