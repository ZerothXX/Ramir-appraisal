"""全局配置：集中管理所有路径与超参数（对应 docs/requirements.md 第四章）。

所有可调数值/路径都必须在此处修改，train.py / test.py / data.py / model.py
内部不允许硬编码任何数值或路径。开发环境固定为 PyCharm 直接运行。
"""

import os

import torch

# ===== 路径配置 =====
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
DATASET_DESC_DIR = os.path.join(DATASET_DIR, "descriptions")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
# 基座模型实际位于 models/Qwen2.5-VL-3B-Instruct/ 子目录（models/ 下按实际布局指向）
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
# LORA_TARGET_MODULES 作为"后缀关键字"列表使用：model.py 对模型中所有 nn.Linear
# 按"模块名以 .{name} 结尾"匹配（peft 的 endswith 规则），并打印实际命中的列表。
# Qwen2.5-VL：语言侧 q_proj/k_proj/v_proj/o_proj，视觉侧 qkv/proj；
# 视觉 patch_embed 的 Conv3d.proj 因非 nn.Linear 会被自动过滤。
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "qkv", "proj"]
# True: 训练目标文本包含 "分析：..." 行（requirements.md 3.1 的默认格式）；
# False: 训练目标仅 "结论：xxx型拉米尔"（模型将不再学习生成分析行）。
# 注意：json 中 label 已含 "型拉米尔" 后缀（如 "害羞型拉米尔"），拼接时不要重复追加。
TRAIN_WITH_ANALYSIS = True

# ===== 训练超参数 =====
NUM_EPOCHS = 10
BATCH_SIZE = 1              # 冒烟实测：8GB 显存下 batch=4 训练 OOM（batch=1 单图峰值已约 9.9GB，
                            # 含 WDDM 共享内存超卖），降到 1 后训练可跑通（梯度累积不变）
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.03
LR_SCHEDULER_TYPE = "cosine"
MAX_TEXT_LENGTH = 128
IMAGE_SIZE = 512             # None 则用模型自带 processor 默认分辨率；设整数 N 则固定 N×N 像素（min/max_pixels），4090 版可调大到 2048 换取细节。
                             # 冒烟实测（8GB 卡）：默认分辨率下 1760x2336 大图 seq~1500、显存峰值 10.4GB 超物理显存
                             # 导致 OOM/换页巨慢（~120s/step）；512 时峰值 8.1GB、大图 10s/step 可正常训练。
                             # 换 4090 或 16GB+ 显存时可改回 None 或调大换取细节。
MIXED_PRECISION = "bf16"     # "no" / "fp16" / "bf16"
GRADIENT_CHECKPOINTING = True
EARLY_STOP_PATIENCE = 2      # 验证 loss 连续 N 个 epoch 未创新低则提前停止训练（0 表示关闭早停）
SAVE_STEPS = 200             # 每多少步保存一次带 step 编号的 checkpoint
EVAL_STEPS = 200             # 每多少步跑一次验证（epoch 结束时固定再跑一次）
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
