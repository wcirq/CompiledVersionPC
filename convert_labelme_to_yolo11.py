# -*- coding: utf-8 -*-
"""
将 datasets/back 下的图片 + Labelme风格json，转换为 YOLOv11 实例分割 数据集

功能：
1. 只保留指定类别
2. 生成 YOLO 实例分割标签 txt（多边形坐标）
3. 随机划分 train / val
4. 尽量保证 train / val 都包含全部类别
5. 若无法满足（例如某类只有1张图），给出告警
6. 自动生成 dataset.yaml

说明：
- 不使用 ArgumentParser，直接修改下面的配置变量即可
- 支持 polygon 形状，直接导出原始多边形坐标做实例分割
- 如果图片没有所需类别目标，可根据 INCLUDE_EMPTY_IMAGES 决定是否保留为空标签图
"""

import json
import random
import shutil
from pathlib import Path
from collections import defaultdict

# =========================
# 配置区：直接改这里即可
# =========================

# 原始数据目录（里面放图片和json）
SOURCE_DIR = Path(f"datasets/train-top/src-null")

# 输出 YOLO 数据集目录
OUTPUT_DIR = Path(f"datasets/train-top/data")

# 需要保留的类别（顺序即类别ID顺序）
SELECTED_CLASSES = [
    "train"
]

# 训练集比例（剩下的是验证集）
TRAIN_RATIO = 0.9

# 随机种子
RANDOM_SEED = 42

# 是否保留“没有目标框但有图片”的空标签图片
# True: 会生成空txt并纳入数据集
# False: 没有选中类别目标的图片直接跳过
INCLUDE_EMPTY_IMAGES = True

# 是否复制图片（True）还是尝试硬链接/失败再复制（False时更快但某些盘不支持）
FORCE_COPY_IMAGES = True

# 是否清空输出目录再生成
CLEAR_OUTPUT_DIR = False

# 支持的图片后缀
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# =========================
# 工具函数
# =========================

def safe_mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def clear_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_or_link_file(src: Path, dst: Path, force_copy: bool = True):
    safe_mkdir(dst.parent)
    if dst.exists():
        dst.unlink()

    if force_copy:
        shutil.copy2(src, dst)
    else:
        try:
            os.link(src, dst)
        except Exception:
            shutil.copy2(src, dst)


def find_image_files(source_dir: Path):
    image_files = []
    for p in source_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            image_files.append(p)
    return sorted(image_files)


def load_json(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# =========================
# 【实例分割核心】多边形坐标归一化
# =========================
def polygon_to_yolo(points, image_w, image_h):
    """
    Labelme 原始多边形点 → YOLO 实例分割归一化坐标
    返回: [x1, y1, x2, y2, ...] （全部 0~1 之间）
    """
    normalized = []
    for (x, y) in points:
        nx = clamp(float(x) / image_w, 0.0, 1.0)
        ny = clamp(float(y) / image_h, 0.0, 1.0)
        normalized.append(nx)
        normalized.append(ny)
    return normalized


def parse_one_json(json_path: Path, selected_class_to_id: dict):
    """
    解析单个json，返回：
    {
        "imagePath": ...,
        "imageWidth": ...,
        "imageHeight": ...,
        "labels": [(cls_id, x1,y1,x2,y2,...), ...],  # 实例分割格式
        "class_ids_in_image": set(...)
    }
    """
    data = load_json(json_path)

    image_w = data.get("imageWidth", None)
    image_h = data.get("imageHeight", None)
    image_path_in_json = data.get("imagePath", "")

    if not image_w or not image_h:
        raise ValueError(f"json缺少 imageWidth/imageHeight: {json_path}")

    yolo_labels = []
    class_ids_in_image = set()

    for shape in data.get("shapes", []):
        label = shape.get("label", "")
        if label not in selected_class_to_id:
            continue

        points = shape.get("points", [])
        if not points or len(points) < 3:  # 分割至少需要3个点
            continue

        cls_id = selected_class_to_id[label]

        try:
            # 直接使用原始多边形做实例分割
            poly = polygon_to_yolo(points, image_w, image_h)
            yolo_labels.append((cls_id, poly))
            class_ids_in_image.add(cls_id)
        except Exception as e:
            print(f"[WARN] 解析分割多边形失败，跳过: {json_path.name}, label={label}, err={e}")

    return {
        "imagePath": image_path_in_json,
        "imageWidth": image_w,
        "imageHeight": image_h,
        "labels": yolo_labels,
        "class_ids_in_image": class_ids_in_image,
    }


def build_samples(source_dir: Path, selected_classes):
    selected_class_to_id = {name: i for i, name in enumerate(selected_classes)}
    image_files = find_image_files(source_dir)
    image_map = {p.stem: p for p in image_files}

    all_samples = []
    missing_json = []
    parse_fail = []

    for stem, image_path in image_map.items():
        json_path = source_dir / f"{stem}.json"
        if not json_path.exists():
            missing_json.append(image_path.name)
            continue

        try:
            parsed = parse_one_json(json_path, selected_class_to_id)
            labels = parsed["labels"]
            class_ids_in_image = parsed["class_ids_in_image"]

            if (not labels) and (not INCLUDE_EMPTY_IMAGES):
                continue

            all_samples.append({
                "image_path": image_path,
                "json_path": json_path,
                "stem": stem,
                "labels": labels,
                "class_ids_in_image": class_ids_in_image,
            })
        except Exception as e:
            parse_fail.append((json_path.name, str(e)))

    return all_samples, missing_json, parse_fail


def ensure_split_contains_classes(samples, selected_classes, train_ratio=0.8, seed=42):
    rng = random.Random(seed)
    samples = samples[:]
    rng.shuffle(samples)

    num_total = len(samples)
    if num_total == 0:
        return [], [], [], []

    target_train_num = int(round(num_total * train_ratio))
    target_train_num = max(1, min(target_train_num, num_total - 1)) if num_total >= 2 else num_total

    class_to_indices = defaultdict(list)
    for idx, s in enumerate(samples):
        for cls_id in s["class_ids_in_image"]:
            class_to_indices[cls_id].append(idx)

    warnings = []
    selected_cls_ids = list(range(len(selected_classes)))

    train_set = set()
    val_set = set()

    for cls_id in selected_cls_ids:
        idxs = class_to_indices.get(cls_id, [])
        if len(idxs) == 0:
            warnings.append(f"[WARN] 类别 '{selected_classes[cls_id]}' 在全部数据中不存在。")
            continue

        rng.shuffle(idxs)

        if len(idxs) == 1:
            only_idx = idxs[0]
            if len(train_set) < target_train_num:
                train_set.add(only_idx)
            else:
                val_set.add(only_idx)
            warnings.append(
                f"[WARN] 类别 '{selected_classes[cls_id]}' 只出现在 1 张图中，无法同时出现在 train 和 val。"
            )
        else:
            train_candidate = idxs[0]
            val_candidate = idxs[1] if len(idxs) > 1 else idxs[0]
            train_set.add(train_candidate)
            val_set.add(val_candidate)

    overlap = train_set & val_set
    for idx in list(overlap):
        if len(train_set) <= target_train_num:
            val_set.discard(idx)
        else:
            train_set.discard(idx)

    all_indices = list(range(num_total))
    remaining = [i for i in all_indices if i not in train_set and i not in val_set]
    rng.shuffle(remaining)

    need_train = max(0, target_train_num - len(train_set))
    for idx in remaining[:need_train]:
        train_set.add(idx)
    for idx in remaining[need_train:]:
        val_set.add(idx)

    if len(train_set) == 0 and len(val_set) > 1:
        idx = next(iter(val_set))
        val_set.remove(idx)
        train_set.add(idx)
    if len(val_set) == 0 and len(train_set) > 1:
        idx = next(iter(train_set))
        train_set.remove(idx)
        val_set.add(idx)

    train_samples = [samples[i] for i in sorted(train_set)]
    val_samples = [samples[i] for i in sorted(val_set)]

    train_classes = set()
    val_classes = set()
    for s in train_samples:
        train_classes |= s["class_ids_in_image"]
    for s in val_samples:
        val_classes |= s["class_ids_in_image"]

    for cls_id, cls_name in enumerate(selected_classes):
        total_count = len(class_to_indices.get(cls_id, []))
        if total_count == 0:
            continue
        if cls_id not in train_classes:
            warnings.append(f"[WARN] 类别 '{cls_name}' 最终未进入 train。")
        if cls_id not in val_classes:
            warnings.append(f"[WARN] 类别 '{cls_name}' 最终未进入 val。")

    return train_samples, val_samples, warnings, samples


# =========================
# 【实例分割】写入标签txt
# =========================
def write_one_label_txt(txt_path: Path, labels):
    safe_mkdir(txt_path.parent)
    with open(txt_path, "w", encoding="utf-8") as f:
        for cls_id, poly_coords in labels:
            # YOLO 实例分割格式：class_id x1 y1 x2 y2 ... xn yn
            line = f"{cls_id} " + " ".join(f"{v:.6f}" for v in poly_coords)
            f.write(line + "\n")


def export_split(samples, split_name: str, output_dir: Path):
    img_out_dir = output_dir / "images" / split_name
    lbl_out_dir = output_dir / "labels" / split_name
    safe_mkdir(img_out_dir)
    safe_mkdir(lbl_out_dir)

    for s in samples:
        img_src = s["image_path"]
        img_dst = img_out_dir / img_src.name
        txt_dst = lbl_out_dir / f"{s['stem']}.txt"

        shutil.copy2(img_src, img_dst)
        write_one_label_txt(txt_dst, s["labels"])


def write_dataset_yaml(output_dir: Path, selected_classes):
    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {output_dir.resolve().as_posix()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(selected_classes)}\n")
        f.write("names:\n")
        for i, name in enumerate(selected_classes):
            f.write(f"  {i}: {name}\n")
    return yaml_path


def collect_split_class_stats(samples, selected_classes):
    image_count_per_class = defaultdict(int)
    seg_count_per_class = defaultdict(int)

    for s in samples:
        cls_set = set()
        for cls_id, *_ in s["labels"]:
            seg_count_per_class[cls_id] += 1
            cls_set.add(cls_id)
        for cls_id in cls_set:
            image_count_per_class[cls_id] += 1

    lines = []
    for cls_id, cls_name in enumerate(selected_classes):
        lines.append(
            f"  - {cls_name}: 图像数={image_count_per_class.get(cls_id, 0)}, 分割实例数={seg_count_per_class.get(cls_id, 0)}"
        )
    return "\n".join(lines)


def main():
    random.seed(RANDOM_SEED)

    print("========== 配置 ==========")
    print(f"SOURCE_DIR         = {SOURCE_DIR}")
    print(f"OUTPUT_DIR         = {OUTPUT_DIR}")
    print(f"SELECTED_CLASSES   = {SELECTED_CLASSES}")
    print(f"TRAIN_RATIO        = {TRAIN_RATIO}")
    print(f"RANDOM_SEED        = {RANDOM_SEED}")
    print(f"INCLUDE_EMPTY_IMAGES = {INCLUDE_EMPTY_IMAGES}")
    print(f"CLEAR_OUTPUT_DIR   = {CLEAR_OUTPUT_DIR}")
    print("==========================\n")

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"源目录不存在: {SOURCE_DIR}")
    if not SELECTED_CLASSES:
        raise ValueError("SELECTED_CLASSES 不能为空")
    if not (0.0 < TRAIN_RATIO < 1.0):
        raise ValueError("TRAIN_RATIO 必须在 (0, 1) 之间")

    if CLEAR_OUTPUT_DIR:
        clear_dir(OUTPUT_DIR)
    else:
        safe_mkdir(OUTPUT_DIR)

    all_samples, missing_json, parse_fail = build_samples(SOURCE_DIR, SELECTED_CLASSES)

    print(f"[INFO] 找到可处理样本数: {len(all_samples)}")
    print(f"[INFO] 缺少json的图片数: {len(missing_json)}")
    if missing_json:
        for name in missing_json[:20]:
            print(f"  [MISSING_JSON] {name}")
        if len(missing_json) > 20:
            print(f"  ... 还有 {len(missing_json) - 20} 个未显示")

    if parse_fail:
        print(f"[WARN] 解析失败文件数: {len(parse_fail)}")
        for name, err in parse_fail[:20]:
            print(f"  [PARSE_FAIL] {name}: {err}")
        if len(parse_fail) > 20:
            print(f"  ... 还有 {len(parse_fail) - 20} 个未显示")

    if len(all_samples) == 0:
        raise RuntimeError("没有可用样本，请检查 SOURCE_DIR、SELECTED_CLASSES 或 json 格式。")

    train_samples, val_samples, split_warnings, used_samples = ensure_split_contains_classes(
        all_samples,
        selected_classes=SELECTED_CLASSES,
        train_ratio=TRAIN_RATIO,
        seed=RANDOM_SEED
    )

    if len(train_samples) == 0 or len(val_samples) == 0:
        raise RuntimeError(f"划分失败：train={len(train_samples)}, val={len(val_samples)}。")

    export_split(train_samples, "train", OUTPUT_DIR)
    export_split(val_samples, "val", OUTPUT_DIR)
    yaml_path = write_dataset_yaml(OUTPUT_DIR, SELECTED_CLASSES)

    print("\n========== 数据集统计 ==========")
    print(f"总样本数: {len(used_samples)}")
    print(f"训练集样本数: {len(train_samples)}")
    print(f"验证集样本数: {len(val_samples)}")
    print("\n[训练集类别统计]")
    print(collect_split_class_stats(train_samples, SELECTED_CLASSES))
    print("\n[验证集类别统计]")
    print(collect_split_class_stats(val_samples, SELECTED_CLASSES))

    if split_warnings:
        print("\n========== 告警 ==========")
        for w in split_warnings:
            print(w)
    else:
        print("\n[INFO] 训练集和验证集类别覆盖检查通过。")

    print("\n========== 输出 ==========")
    print(f"YOLO 实例分割数据集目录: {OUTPUT_DIR.resolve()}")
    print(f"dataset.yaml:   {yaml_path.resolve()}")
    print("\n完成！")


if __name__ == "__main__":
    main()