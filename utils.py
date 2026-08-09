# -*- coding: utf-8 -*-
"""通用工具：曲线绘图、指标存取、随机种子、目录创建、日志（对应 docs/requirements.md 第八章）。

约定：
* 图内文字一律英文（Windows 控制台/无中文字体环境下 matplotlib 会渲染成方块）；
* 本模块不依赖 config.py，所有参数由调用方显式传入。
"""

import json
import logging
import os
import random

import matplotlib

matplotlib.use("Agg")  # 无界面后端，供 PyCharm 直接运行
import matplotlib.pyplot as plt

import numpy as np
import torch


def ensure_dir(path):
    """目录不存在则创建（train/test 写文件前统一调用）。"""
    os.makedirs(path, exist_ok=True)
    return path


def set_seed(seed):
    """统一设置 random / numpy / torch 随机种子，保证可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_logger(name="ani_appraisal"):
    """返回输出到控制台的 logger（不写文件日志，保持轻量）。"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.propagate = False
    return logger


def save_metrics_json(metrics_dict, save_path):
    """曲线原始数据落盘为 json（与图片分离保存，便于日后重绘/分析）。"""
    ensure_dir(os.path.dirname(save_path))
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, ensure_ascii=False, indent=2)


def load_metrics_json(save_path):
    """读取 save_metrics_json 保存的原始指标数据。"""
    with open(save_path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_curve(x, y_dict, title, xlabel, ylabel, save_path):
    """多曲线同图画到一张图上并保存 png。

    参数：
        x:        x 轴数据（与每条曲线的 y 数据等长）
        y_dict:   {"曲线名": [y 值, ...], ...}，支持同图多条曲线（如 train/val 双线）
        save_path: 输出 png 路径（父目录不存在会自动创建）
    图内文字一律英文。
    """
    ensure_dir(os.path.dirname(save_path))
    plt.figure(figsize=(8, 5))
    for name, y in y_dict.items():
        plt.plot(x, y, label=name)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
