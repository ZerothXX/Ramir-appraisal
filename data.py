"""data.py — 数据发现、解析、Dataset/DataLoader 构建（对应 docs/requirements.md 第五章）。

职责边界：
* 只负责数据层：扫描校验（scan_dataset）、可复现划分（train_val_split）、
  数据集封装（AniTypeDataset）、batch 组装工厂（make_collate_fn）；
* 不涉及模型结构 / LoRA / 训练循环（属 model.py / train.py）；
* 所有路径与超参数一律从 config.py 读取，本文件不硬编码任何路径或数值；
  唯一例外：图片扩展名集合属于数据格式约定（需求方口径），作为模块常量维护。
"""

import json
import os
import random

import torch  # noqa: F401（保证 torch 环境可用；Dataset 基类来自 torch.utils.data）
from PIL import Image
from torch.utils.data import Dataset

import config

# ===== 数据格式约定 =====
# 图片文件扩展名：jpg/jpeg/png/bmp/webp 均算作图片（格式约定，非可调超参数）。
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _warn(message: str) -> None:
    """统一输出带来源标识的警告（样本级问题：警告并跳过，不中断流程）。"""
    print(f"[data.py 警告] {message}", flush=True)


def _is_image_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS


def _natural_sort_key(name: str):
    """imgs1, imgs2, ..., imgs10 按数字序排列（遍历与警告信息的可读性、确定性）。"""
    digits = name[len("imgs"):] if name.startswith("imgs") else ""
    return (int(digits) if digits.isdigit() else -1, name)


def _image_open_error(image_path: str):
    """打开并校验图片完整性；返回 None 表示可正常打开，否则返回异常对象。"""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return None
    except Exception as e:  # 损坏文件 / 非图片格式误放入等，均由调用方决定警告+跳过
        return e


def _check_entry_fields(folder_dir, json_path, entry, idx=None):
    """校验单条 json 数据必须包含 path/label/analysis 三个字段，否则抛 ValueError。"""
    if not isinstance(entry, dict) or not all(k in entry for k in ("path", "label", "analysis")):
        where = f"第 {idx} 条" if idx is not None else "数据"
        raise ValueError(
            f"描述文件字段缺失: {folder_dir}（描述文件 {json_path}）{where} "
            f"必须包含 path/label/analysis 三个字段，实际内容: {entry!r}"
        )


def _scan_closed_folder(dataset_dir, folder, folder_dir, json_path, entries, image_files):
    """闭集文件夹处理：json 必须恰好 1 条，label/analysis 广播给文件夹内全部图片。

    条数不等于 1 直接抛异常（规格 11.1：禁止静默按开放集逻辑处理）。
    """
    if len(entries) != 1:
        raise ValueError(
            f"闭集文件夹数据条数校验失败: {folder_dir}（描述文件 {json_path}）："
            f"闭集 json 必须恰好 1 条数据（该文件夹全部图片共享同一条 label 并广播），"
            f"实际为 {len(entries)} 条（规格 11.1：禁止静默按开放集逻辑处理）"
        )
    entry = entries[0]
    _check_entry_fields(folder_dir, json_path, entry)
    entry_full = os.path.normpath(os.path.join(dataset_dir, entry["path"]))
    if not os.path.isfile(entry_full):
        raise FileNotFoundError(
            f"闭集文件夹描述路径不存在: {folder_dir} 的 {json_path} 中 "
            f"path={entry['path']!r}，解析后文件不存在: {entry_full}（规格 2.2：图片路径必须真实存在）"
        )
    samples = []
    for fname in image_files:
        image_path = os.path.join(folder_dir, fname)
        err = _image_open_error(image_path)
        if err is not None:
            _warn(f"闭集文件夹 {folder_dir} 图片无法打开，跳过该样本: {image_path}（原因: {err}）")
            continue
        samples.append({
            "image_path": image_path,
            "label_text": entry["label"],
            "analysis_text": entry["analysis"],
            "is_closed_set": True,
            "source_dir": folder_dir,
        })
    return samples


def _scan_open_folder(dataset_dir, folder, folder_dir, json_path, entries, image_files):
    """开放集文件夹处理：json 条数必须与文件夹内图片文件数一致，逐条对齐。

    条数不一致直接抛异常并指明文件夹名与两个数字（规格 2.4，禁止静默跳过/截断）。
    """
    if len(entries) != len(image_files):
        raise ValueError(
            f"开放集文件夹数据条数不匹配: {folder_dir}（描述文件 {json_path}）："
            f"json 条数 {len(entries)} 与文件夹内图片文件数 {len(image_files)} 不一致，"
            f"禁止静默跳过或截断（规格 2.4）"
        )
    disk_names = set(image_files)
    used_names = set()
    samples = []
    for idx, entry in enumerate(entries, start=1):
        _check_entry_fields(folder_dir, json_path, entry, idx=idx)
        entry_full = os.path.normpath(os.path.join(dataset_dir, entry["path"]))
        if not os.path.isfile(entry_full):
            raise FileNotFoundError(
                f"开放集文件夹描述路径不存在: {folder_dir} 的 {json_path} 第 {idx} 条 "
                f"path={entry['path']!r}，解析后文件不存在: {entry_full}（规格 2.2）"
            )
        fname = os.path.basename(entry_full)
        if fname not in disk_names:
            raise ValueError(
                f"开放集文件夹描述路径不属于该文件夹: {folder_dir}（描述文件 {json_path}）"
                f"第 {idx} 条 path={entry['path']!r} 不是 {folder_dir} 内的图片文件"
            )
        if fname in used_names:
            raise ValueError(
                f"开放集文件夹描述路径重复: {folder_dir}（描述文件 {json_path}）"
                f"第 {idx} 条 path={entry['path']!r} 与前面条目指向同一文件"
            )
        used_names.add(fname)
        err = _image_open_error(entry_full)
        if err is not None:
            _warn(f"开放集文件夹 {folder_dir} 图片无法打开，跳过该样本: {entry_full}（原因: {err}）")
            continue
        samples.append({
            "image_path": entry_full,
            "label_text": entry["label"],
            "analysis_text": entry["analysis"],
            "is_closed_set": False,
            "source_dir": folder_dir,
        })
    return samples


def scan_dataset(dataset_dir, desc_dir, closed_set_dirs):
    """扫描 dataset/ 下所有 imgs* 文件夹并生成统一样本列表（规格 5.1 / 第二章）。

    参数：
        dataset_dir:   数据集根目录（如 config.DATASET_DIR）
        desc_dir:      描述 json 目录（如 config.DATASET_DESC_DIR）
        closed_set_dirs: 闭集文件夹名单（如 config.CLOSED_SET_DIRS），不在名单内的
                         imgs* 一律按开放集处理（规格 2.4，不用编号大小等隐式规则猜测）

    返回：
        样本 dict 列表，每条至少包含：
            image_path:    图片绝对路径
            label_text:    目标标签文本（闭集为文件夹共享标签，开放集为该图专属标签）
            analysis_text: 分析文本（与 label 对应）
            is_closed_set: bool，是否来自闭集文件夹
            source_dir:    所属图片文件夹绝对路径

    校验规则（fail fast，异常信息均指明具体文件夹/文件）：
        * 每个 imgs* 文件夹必须有对应的 {folder}.json，缺失即抛 FileNotFoundError；
        * 闭集：json 必须恰好 1 条且含 path/label/analysis 三字段，label 广播到
          文件夹内全部图片文件；条数不等于 1 抛 ValueError（规格 11.1）；
        * 开放集：json 条数必须与文件夹内图片文件数一致，不一致抛 ValueError 并
          指明文件夹名与两个数字（规格 2.4）；json 路径必须真实存在（规格 2.2）；
        * 单个图片损坏/无法打开：打印明确警告并跳过该样本（规格 11.2），不中断流程。
    """
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"数据集目录不存在: {dataset_dir}")
    if not os.path.isdir(desc_dir):
        raise FileNotFoundError(f"描述文件目录不存在: {desc_dir}")

    closed_set = set(closed_set_dirs)
    folders = sorted(
        (n for n in os.listdir(dataset_dir)
         if n.startswith("imgs") and os.path.isdir(os.path.join(dataset_dir, n))),
        key=_natural_sort_key,
    )
    if not folders:
        raise ValueError(f"数据集目录下未找到任何 imgs* 文件夹: {dataset_dir}")

    samples = []
    for folder in folders:
        folder_dir = os.path.join(dataset_dir, folder)
        json_path = os.path.join(desc_dir, f"{folder}.json")
        if not os.path.isfile(json_path):
            raise FileNotFoundError(
                f"缺少描述文件: 文件夹 {folder_dir} 对应 {json_path} 不存在，请检查数据完整性"
            )
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                entries = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(
                f"描述文件解析失败（JSON 损坏或编码非 UTF-8）: {json_path}（{e}）"
            ) from e
        if not isinstance(entries, list):
            raise ValueError(
                f"描述文件格式错误: {json_path} 顶层必须是 JSON 数组，实际为 {type(entries).__name__}"
            )

        image_files = sorted(f for f in os.listdir(folder_dir) if _is_image_file(f))
        if not image_files:
            raise ValueError(f"文件夹内没有任何图片文件: {folder_dir}")

        if folder in closed_set:
            samples.extend(_scan_closed_folder(
                dataset_dir, folder, folder_dir, json_path, entries, image_files))
        else:
            samples.extend(_scan_open_folder(
                dataset_dir, folder, folder_dir, json_path, entries, image_files))

    if not samples:
        raise ValueError(f"扫描完成但没有产生任何样本（dataset_dir={dataset_dir}）")
    return samples


def train_val_split(samples, ratio, seed):
    """按固定随机种子可复现地划分训练/验证集（规格 5.3）。

    使用独立的 random.Random(seed) 实例洗牌，不污染全局随机状态；
    相同 seed + 相同输入顺序必然得到相同划分。不修改传入列表。
    """
    if not 0.0 < ratio < 1.0:
        raise ValueError(f"train/val 划分比例必须位于 (0, 1) 区间，实际为: {ratio}")
    items = list(samples)
    if not items:
        raise ValueError("train_val_split 收到空样本列表，无法划分")
    rng = random.Random(seed)
    rng.shuffle(items)
    n_train = int(len(items) * ratio)
    return items[:n_train], items[n_train:]


class AniTypeDataset(Dataset):
    """拉米尔鉴定样本数据集（规格 5.2 / 5.4 的唯一一致解读）。

    __getitem__ 返回 (PIL 图片, 目标文本)：图像张量化、pad 组装在 collate 阶段
    由 model.build_train_batch 统一完成（规格 5.4），本类不调用 processor 编码。

    目标文本格式（规格 3.1，label 已含"型拉米尔"后缀，直接拼接不重复追加）：
        config.TRAIN_WITH_ANALYSIS=True  -> "结论：{label_text}\n分析：{analysis_text}"
        config.TRAIN_WITH_ANALYSIS=False -> "结论：{label_text}"

    图片读取失败：打印明确警告并跳过该样本（规格 11.2），不让训练崩溃。
    """

    def __init__(self, samples, processor):
        """
        samples:   scan_dataset 输出的样本 dict 列表（或 train_val_split 划分后的子集）
        processor: VLM processor（按契约接收并透传；实际编码在 collate 阶段完成）
        """
        self.processor = processor
        self.with_analysis = config.TRAIN_WITH_ANALYSIS
        self._images = []
        self._texts = []
        skipped = 0
        for sample in samples:
            image = self._safe_load(sample["image_path"])
            if image is None:
                skipped += 1
                continue
            self._images.append(image)
            self._texts.append(self._build_target(sample))
        if skipped > 0:
            _warn(f"AniTypeDataset 初始化时跳过 {skipped} 个无法打开的图片样本")
        if not self._images:
            raise RuntimeError(
                "AniTypeDataset 没有任何可用样本（全部图片读取失败被跳过），无法训练"
            )

    def _safe_load(self, image_path):
        try:
            with Image.open(image_path) as img:
                return img.convert("RGB")
        except Exception as e:
            _warn(f"图片读取失败，跳过该样本: {image_path}（原因: {e}）")
            return None

    def _build_target(self, sample):
        if self.with_analysis:
            return f"结论：{sample['label_text']}\n分析：{sample['analysis_text']}"
        return f"结论：{sample['label_text']}"

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        return self._images[idx], self._texts[idx]


def make_collate_fn(processor):
    """返回 collate_fn 工厂（规格 5.4 / 文件分工契约）。

    collate_fn 接收 [(PIL 图片, 目标文本), ...] 的 batch，
    调用 model.py 的 build_train_batch(processor, images, prompts, targets)
    组装模型输入并返回；prompts 统一使用 model.py 的 INSTRUCTION_PROMPT 常量。

    model 模块采用延迟导入：model.py 与 peft 由并行/后续阶段提供，
    data.py 单独导入（如纯数据冒烟测试）时不依赖它们。
    """
    try:
        import model as _model_mod
    except Exception as e:
        raise RuntimeError(
            "make_collate_fn 需要 model.py 提供 build_train_batch / INSTRUCTION_PROMPT "
            f"契约导出，但导入 model 失败: {type(e).__name__}: {e}"
        ) from e
    build_train_batch = getattr(_model_mod, "build_train_batch", None)
    instruction_prompt = getattr(_model_mod, "INSTRUCTION_PROMPT", None)
    if build_train_batch is None or instruction_prompt is None:
        present = [n for n in ("build_train_batch", "INSTRUCTION_PROMPT")
                   if hasattr(_model_mod, n)]
        raise RuntimeError(
            "model.py 缺少契约导出: 需要 build_train_batch(processor, images, prompts, targets)"
            f" 与 INSTRUCTION_PROMPT 常量，当前 model.py 实际仅有: {present}"
        )

    def collate_fn(batch):
        images = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        prompts = [instruction_prompt] * len(batch)
        return build_train_batch(processor, images, prompts, targets)

    return collate_fn
