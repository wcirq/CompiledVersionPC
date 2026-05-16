# -*- coding: utf-8 -*-
"""
YOLOv11 目标检测训练脚本（不使用 ArgumentParser）

使用方式：
1. 修改下面“配置区”的变量
2. 直接运行：
   python train_yolo11_detect.py

依赖：
   pip install ultralytics

说明：
- 本脚本面向 Ultralytics YOLO11/YOLO 系列目标检测训练
- 支持 n / s / m / l / x 模型规格选择
- 支持从官方预训练权重训练，也支持从 yaml 结构训练
- 不使用 argparse，所有参数直接在变量区改
"""
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO


# =========================================================
# 1. 基础配置
# =========================================================

EXP = "exp_20260504"
p_model_name = "last"

# 数据集配置文件（你前面生成的 dataset.yaml）
DATASET_YAML = f"datasets/train-top/data/dataset.yaml"

# 模型规格：可选 "n", "s", "m", "l", "x"
MODEL_SIZE = "n"

# 任务类型：这里固定为目标检测
TASK = "segment"

# 是否使用预训练权重
USE_PRETRAINED = True

# 如果 USE_PRETRAINED=True：
#   将自动加载 yolo11{MODEL_SIZE}.pt
# 如果 USE_PRETRAINED=False：
#   将自动加载 yolo11{MODEL_SIZE}.yaml（从结构开始训练）
MODEL_DIR = ""   # 一般留空即可，如需自定义可填目录，例如 "/data/models"

# 训练输出项目名
PROJECT = "runs/train"
NAME = f"yolo11{MODEL_SIZE}_exp_{datetime.now().strftime('%Y%m%d')}"

# 是否允许覆盖同名实验目录
EXIST_OK = True

# 设备
# 可选：
#   "0"      -> 第0张GPU
#   "0,1"    -> 多卡
#   "cpu"    -> CPU
#   None     -> 自动选择
DEVICE = "0"

# 随机种子
SEED = 42

# 是否确定性训练（更可复现，但可能稍慢）
DETERMINISTIC = True

# 单卡/总 batch size
BATCH = 64

# 输入尺寸
IMGSZ = 640

# 训练轮数
# EPOCHS = 140
EPOCHS = 2000

# 提前停止（连续多少轮指标不提升则停止）
PATIENCE = 100

# 数据加载线程数
WORKERS = 4

# 是否缓存数据到内存/磁盘
# 可选：False, True, "ram", "disk"
CACHE = False

# 断点续训
RESUME = False

# 是否使用 AMP 混合精度
AMP = True

# 训练阶段是否输出更详细日志
VERBOSE = True


# =========================================================
# 2. 优化器与学习率相关
# =========================================================

# 优化器
# 可选：'auto', 'SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp'
OPTIMIZER = "auto"

# 初始学习率
LR0 = 0.01

# 最终学习率系数（最终lr = LR0 * LRF）
LRF = 0.01

# momentum（对 SGD / RMSProp 等有效）
MOMENTUM = 0.937

# 权重衰减
WEIGHT_DECAY = 0.0005

# warmup epoch 数
WARMUP_EPOCHS = 3.0

# warmup 初始 momentum
WARMUP_MOMENTUM = 0.8

# warmup 初始 bias lr
WARMUP_BIAS_LR = 0.1

# 是否使用余弦退火学习率
COS_LR = False

# 是否关闭 mosaic（最后 N 个 epoch 关闭）
CLOSE_MOSAIC = 10

# 冻结前几层
# 例如：
#   None / 0  -> 不冻结
#   10        -> 冻结前10层
#   [0,1,2]   -> 冻结指定层
FREEZE = None


# =========================================================
# 3. 损失相关超参数
# =========================================================

# box loss 权重
BOX = 7.5

# cls loss 权重
CLS = 0.5

# dfl loss 权重（Distribution Focal Loss）
DFL = 1.5

# pose / kobj 对纯目标检测一般不用，但保留写上
POSE = 12.0
KOBJ = 1.0

# label smoothing
LABEL_SMOOTHING = 0.0

# nominal batch size，用于 loss 归一化
NBS = 64


# =========================================================
# 4. 数据增强参数
# =========================================================

# HSV 颜色增强
HSV_H = 0.015
HSV_S = 0.7
HSV_V = 0.4

# 几何增强
DEGREES = 180
TRANSLATE = 0.1
SCALE = 0.5
SHEAR = 0.0
PERSPECTIVE = 0.0

# 上下 / 左右翻转
FLIPUD = 0.5
FLIPLR = 0.5

# 通道顺序扰动（BGR）
BGR = 0.5

# Mosaic / MixUp / CopyPaste
MOSAIC = 0.8
MIXUP = 0.2
COPY_PASTE = 0.0
COPY_PASTE_MODE = "flip"   # 可选值依版本可能不同，常见 "flip"

# 随机擦除（分类更常用，检测里一般较少用）
ERASING = 0.0

# 随机裁剪比例（分类更常见，检测中一般不用）
CROP_FRACTION = 1.0


# =========================================================
# 5. 验证 / 保存 / 推理相关
# =========================================================

# 训练时是否做验证
VAL = True

# 验证集上的置信度阈值（验证/预测时可影响输出）
CONF = None

# NMS IoU 阈值
IOU = 0.7

# 每张图最多检测目标数
MAX_DET = 300

# 是否保存 checkpoint
SAVE = True

# 保存周期，-1表示按默认策略
SAVE_PERIOD = -1

# 是否保存训练图、曲线等
PLOTS = True

# 是否保存 json 评测结果（COCO风格评估常用）
SAVE_JSON = False

# 是否保存混淆矩阵/验证预测图等
# 一般由 PLOTS 控制，这里额外保留
VISUALIZE = False

# 是否单类别训练（把所有类别视为1类）
SINGLE_CLS = False

# 是否矩形训练
RECT = False

# 是否使用多尺度训练
MULTI_SCALE = False

# 是否仅训练，不下载额外内容等
PRETRAINED = True   # 通常与 USE_PRETRAINED 保持一致


# =========================================================
# 6. 分割/姿态/分类通用参数（检测任务里一般不用，但写全）
# =========================================================

OVERLAP_MASK = True
MASK_RATIO = 4
DROPOUT = 0.0


# =========================================================
# 7. 自定义补充参数
# =========================================================

# 你也可以直接指定现成权重路径，而不是自动拼 yolo11s.pt
CUSTOM_MODEL_PATH = f"yolo11{MODEL_SIZE}-seg.pt"

# 训练完成后是否自动做一次 val
RUN_FINAL_VAL = True


def build_model_path():
    """
    构建模型路径：
    1. 如果 CUSTOM_MODEL_PATH 非空，优先用它
    2. 否则根据 MODEL_SIZE + USE_PRETRAINED 自动构建
    """
    if CUSTOM_MODEL_PATH:
        return CUSTOM_MODEL_PATH

    size = MODEL_SIZE.lower().strip()
    if size not in {"n", "s", "m", "l", "x"}:
        raise ValueError(f"MODEL_SIZE 必须是 n/s/m/l/x 之一，当前为: {MODEL_SIZE}")

    if USE_PRETRAINED:
        filename = f"yolo11{size}.pt"
    else:
        filename = f"yolo11{size}.yaml"

    if MODEL_DIR:
        return str(Path(MODEL_DIR) / filename)
    return filename


def main():
    model_path = build_model_path()

    print("=" * 80)
    print("YOLOv11 目标检测训练配置")
    print("=" * 80)
    print(f"DATASET_YAML   : {DATASET_YAML}")
    print(f"MODEL_SIZE     : {MODEL_SIZE}")
    print(f"USE_PRETRAINED : {USE_PRETRAINED}")
    print(f"MODEL_PATH     : {model_path}")
    print(f"PROJECT/NAME   : {PROJECT}/{NAME}")
    print(f"DEVICE         : {DEVICE}")
    print(f"EPOCHS         : {EPOCHS}")
    print(f"BATCH          : {BATCH}")
    print(f"IMGSZ          : {IMGSZ}")
    print("=" * 80)

    model = YOLO(model_path)

    # model.load(f"runs/detect/runs/train/yolo11{MODEL_SIZE}_{DATASET_TYPE}_{EXP}/weights/{p_model_name}.pt")

    results = model.train(
        # -------------------------
        # 基础
        # -------------------------
        data=DATASET_YAML,
        task=TASK,
        epochs=EPOCHS,
        batch=BATCH,
        imgsz=IMGSZ,
        project=PROJECT,
        name=NAME,
        exist_ok=EXIST_OK,
        device=DEVICE,
        workers=WORKERS,
        cache=CACHE,
        seed=SEED,
        deterministic=DETERMINISTIC,
        resume=RESUME,
        amp=AMP,
        verbose=VERBOSE,

        # -------------------------
        # 保存/验证
        # -------------------------
        val=VAL,
        save=SAVE,
        save_period=SAVE_PERIOD,
        plots=PLOTS,
        save_json=SAVE_JSON,
        conf=CONF,
        iou=IOU,
        max_det=MAX_DET,

        # -------------------------
        # 训练控制
        # -------------------------
        patience=PATIENCE,
        pretrained=PRETRAINED,
        single_cls=SINGLE_CLS,
        rect=RECT,
        multi_scale=MULTI_SCALE,
        freeze=FREEZE,

        # -------------------------
        # 优化器
        # -------------------------
        optimizer=OPTIMIZER,
        lr0=LR0,
        lrf=LRF,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        warmup_epochs=WARMUP_EPOCHS,
        warmup_momentum=WARMUP_MOMENTUM,
        warmup_bias_lr=WARMUP_BIAS_LR,
        cos_lr=COS_LR,
        close_mosaic=CLOSE_MOSAIC,
        nbs=NBS,

        # -------------------------
        # 损失权重
        # -------------------------
        box=BOX,
        cls=CLS,
        dfl=DFL,
        pose=POSE,
        kobj=KOBJ,
        label_smoothing=LABEL_SMOOTHING,

        # -------------------------
        # 数据增强
        # -------------------------
        hsv_h=HSV_H,
        hsv_s=HSV_S,
        hsv_v=HSV_V,
        degrees=DEGREES,
        translate=TRANSLATE,
        scale=SCALE,
        shear=SHEAR,
        perspective=PERSPECTIVE,
        flipud=FLIPUD,
        fliplr=FLIPLR,
        bgr=BGR,
        mosaic=MOSAIC,
        mixup=MIXUP,
        copy_paste=COPY_PASTE,
        copy_paste_mode=COPY_PASTE_MODE,
        erasing=ERASING,
        crop_fraction=CROP_FRACTION,

        # -------------------------
        # 其他任务通用参数（检测任务里通常不生效或无需关心）
        # -------------------------
        overlap_mask=OVERLAP_MASK,
        mask_ratio=MASK_RATIO,
        dropout=DROPOUT,
    )

    print("\n训练完成。")
    print("train() 返回结果对象：")
    print(results)

    if RUN_FINAL_VAL:
        print("\n开始执行最终验证...")
        metrics = model.val(
            data=DATASET_YAML,
            imgsz=IMGSZ,
            batch=BATCH,
            device=DEVICE,
            conf=CONF,
            iou=IOU,
            max_det=MAX_DET,
            plots=PLOTS,
            save_json=SAVE_JSON,
        )
        print("\n最终验证完成。metrics：")
        print(metrics)


if __name__ == "__main__":
    main()