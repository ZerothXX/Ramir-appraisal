# -*- coding: utf-8 -*-
"""单图推理（对应 docs/requirements.md 第九章）。

在 PyCharm 中直接改 TEST_IMAGE_PATH 后运行即可完成
"选一张图 -> 生成 -> 后处理 -> 写出 outputs/Ani_type/{图片名}_result.txt"。

流程：加载基座 -> 按 config1.CHECKPOINT_SELECT（"best"/"latest"）加载 LoRA adapter
（目录缺失时回退另一个并提示）-> 用与训练完全一致的 INSTRUCTION_PROMPT 构建输入
-> model.generate（采样参数全部来自 config1.GEN_*）-> 轻量后处理 -> 写 txt + print。

后处理原则（规格 3.2 / 9.4 / 11.3）：只做格式修补，不编造模型没有表达的内容。
"""

import os
import re

import torch
from PIL import Image

import config1
from model1 import (
    INSTRUCTION_PROMPT,
    build_inputs,
    load_base_model_and_processor,
    load_lora_checkpoint,
)
from utils1 import ensure_dir, get_logger

logger = get_logger("test")

# ===== 待测试图片路径：在 PyCharm 里直接改这一行即可 =====
TEST_IMAGE_PATH = os.path.join(config1.INPUT_DIR, "example.jpg")


def pick_adapter_dir():
    """按 config1.CHECKPOINT_SELECT 选择 best/latest；对应目录缺失时回退另一个并提示。"""
    if config1.CHECKPOINT_SELECT == "latest":
        order = ["latest", "best"]
    else:  # 未知取值一律按 best 优先
        order = ["best", "latest"]
    for key in order:
        adapter_dir = os.path.join(config1.OUTPUT_TRAIN_MODELS_DIR, key)
        if os.path.isdir(adapter_dir) and os.path.isfile(
                os.path.join(adapter_dir, "adapter_config1.json")):
            if key != config1.CHECKPOINT_SELECT:
                logger.warning("CHECKPOINT_SELECT=%s 对应目录不存在，回退加载 %s",
                               config1.CHECKPOINT_SELECT, adapter_dir)
            return adapter_dir
    raise FileNotFoundError(
        f"outputs/train_models 下没有可用 LoRA checkpoint（{config1.OUTPUT_TRAIN_MODELS_DIR} "
        "下 best/ 与 latest/ 均不存在或缺少 adapter_config1.json），请先运行 train.py")


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
    ensure_dir(config1.OUTPUT_ANI_TYPE_DIR)
    if not os.path.isfile(TEST_IMAGE_PATH):
        raise FileNotFoundError(f"测试图片不存在: {TEST_IMAGE_PATH}")

    logger.info("加载基座模型 + processor ...")
    model, processor = load_base_model_and_processor(config)
    adapter_dir = pick_adapter_dir()
    model = load_lora_checkpoint(model, adapter_dir)
    model.to(config1.DEVICE)
    model.eval()
    # apply_lora 为省显存把 use_cache 关了，推理需重新打开（否则 generate 无 KV cache，明显变慢）
    model.generation_config1.use_cache = True

    image = Image.open(TEST_IMAGE_PATH).convert("RGB")
    # build_inputs 与训练共用同一份 INSTRUCTION_PROMPT（单一来源，见 model.py）
    batch = build_inputs(processor, [image], [INSTRUCTION_PROMPT])
    batch = {k: (v.to(config1.DEVICE) if isinstance(v, torch.Tensor) else v)
             for k, v in batch.items()}

    logger.info("生成中 ...（max_new_tokens=%d, temperature=%s, top_p=%s, do_sample=%s）",
                config1.GEN_MAX_NEW_TOKENS, config1.GEN_TEMPERATURE,
                config1.GEN_TOP_P, config1.GEN_DO_SAMPLE)
    with torch.no_grad():
        gen_ids = model.generate(
            **batch,
            max_new_tokens=config1.GEN_MAX_NEW_TOKENS,
            temperature=config1.GEN_TEMPERATURE,
            top_p=config1.GEN_TOP_P,
            do_sample=config1.GEN_DO_SAMPLE,
        )

    input_len = batch["input_ids"].shape[1]
    new_ids = gen_ids[0, input_len:]
    raw_text = processor.tokenizer.decode(new_ids, skip_special_tokens=True)
    logger.info("模型原始输出: %r", raw_text)

    result = postprocess_output(raw_text)
    if not result:
        logger.warning("模型输出为空，结果文件将为空（只补格式不编造内容，请检查生成参数）")
    stem = os.path.splitext(os.path.basename(TEST_IMAGE_PATH))[0]
    out_path = os.path.join(config1.OUTPUT_ANI_TYPE_DIR, f"{stem}_result.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result + "\n")

    print("========== 鉴定结果 ==========")
    print(result)
    print(f"已写入: {out_path}")


if __name__ == "__main__":
    main()
