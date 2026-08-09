"""4090 服务器版全局配置（24GB 显存，无本机 8.6GB 的显存约束）。

与本机版 config.py 的差异（其余代码与 data1/model1/utils1/train1/test1 同构，
本文件是两套代码唯一的行为分界）：
* BATCH_SIZE 8（本机 1）/ GRADIENT_ACCUMULATION_STEPS 2（本机 4），
  有效 batch 同为 16，单步吞吐约为本机 8 倍；
* IMAGE_SIZE 2048（本机 512）：更高分辨率保留服饰/配件/表情细节，
  每图约 5 千图像 token（本机 512 下约 300）；
* LoRA 追加语言侧 MLP 层（gate_proj/up_proj/down_proj），适配容量更大；
* GRADIENT_CHECKPOINTING 保持 True：梯度检查点省约 80% 激活显存且计算损失
  有限（约 20-30% 训练耗时），24GB 卡开着它才能把分辨率和 batch 推到更大，
  不是"低显存妥协"，而是性价比最高的显存/速度平衡手段。

若服务器显存更大（如 48GB+），可进一步把 BATCH_SIZE 提到 16、IMAGE_SIZE 提到 3072。
"""

import os

import torch

# ===== 路径配置 =====
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
DATASET_DESC_DIR = os.path.join(DATASET_DIR, "descriptions")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
BASE_MODEL_NAME_OR_PATH = os.path.join(MODELS_DIR, "Qwen2.5-VL-3B-Instruct")
OUTPUT_TRAIN_MODELS_DIR = os.path.join(PROJECT_ROOT, "outputs", "train_models")
OUTPUT_CURVE_DIR = os.path.join(PROJECT_ROOT, "outputs", "curve")
OUTPUT_ANI_TYPE_DIR = os.path.join(PROJECT_ROOT, "outputs", "Ani_type")
INPUT_DIR = os.path.join(PROJECT_ROOT, "inputs")

# ===== 数据配置 =====
# 闭集文件夹名单，不在名单内的 imgs* 一律按开放集处理。
# 注：imgs5 的 json 为 18 条逐图标签（开放集格式），经确认按开放集处理，故不在名单内；
# 实际闭集 4×18=72 样本 + 开放集 2×18=36 样本，共 108 条。
CLOSED_SET_DIRS = ["imgs1", "imgs2", "imgs3", "imgs4"]
TRAIN_VAL_SPLIT_RATIO = 0.9
RANDOM_SEED = 42

# ===== 模型/LoRA 超参数 =====
# LORA_TARGET_MODULES 作为"后缀关键字"列表使用：model1.py 对模型中所有 nn.Linear
# 按"模块名以 .{name} 结尾"匹配，并打印实际命中的列表。
# 4090 版比本机版多了语言侧 MLP 层（gate_proj/up_proj/down_proj），适配容量更大；
# 视觉 patch_embed 的 Conv3d.proj 因非 nn.Linear 会被自动过滤。
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj", "qkv", "proj",
    "gate_proj", "up_proj", "down_proj",
]
# True: 训练目标文本包含 "分析：..." 行（requirements.md 3.1 的默认格式）；
# False: 训练目标仅 "结论：xxx型拉米尔"（模型将不再学习生成分析行）。
# 注意：json 中 label 已含 "型拉米尔" 后缀（如 "害羞型拉米尔"），拼接时不要重复追加。
TRAIN_WITH_ANALYSIS = True

# ===== 训练超参数 =====
NUM_EPOCHS = 10
BATCH_SIZE = 8               # 4090 24GB：2048 分辨率 + batch 8 实测显存约 11-13GB，余量充足
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.03
LR_SCHEDULER_TYPE = "cosine"
MAX_TEXT_LENGTH = 128
IMAGE_SIZE = 2048            # 2048×2048 像素；本机版为 512，4090 版调大换取服饰/配件细节
MIXED_PRECISION = "bf16"     # "no" / "fp16" / "bf16"
GRADIENT_CHECKPOINTING = True
EARLY_STOP_PATIENCE = 2      # 验证 loss 连续 N 个 epoch 未创新低则提前停止训练（0 表示关闭早停）
SAVE_STEPS = 200
EVAL_STEPS = 200
LOG_EVERY_N_STEPS = 10
MAX_GRAD_NORM = 1.0

# ===== 设备 =====
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===== 推理超参数 =====
CHECKPOINT_SELECT = "best"   # test.py 加载哪种 LoRA： "best" = 验证 loss 最优 / "latest" = 最新
GEN_MAX_NEW_TOKENS = 200
GEN_TEMPERATURE = 0.9        # 保留采样随机性，服务于开放式多样化输出
GEN_TOP_P = 0.9
GEN_DO_SAMPLE = True
