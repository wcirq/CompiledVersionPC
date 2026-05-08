import os
import sys
import numpy as np
import cv2
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm

# ====================== 【配置：修改这里即可】 ======================
# 训练好的模型路径（best.pt）
MODEL_PATH = "runs/segment/runs/train/yolo11n_exp_20260507/weights/best.pt"

# 图片文件夹路径
IMAGE_FOLDER = "/media/wcirq/data/datasets/成都大弯镇/火车俯拍分割模型数据集/null"

# 置信度阈值
CONF_THRES = 0.8
# =================================================================
# 加载模型
model = YOLO(MODEL_PATH)

# 获取图片
image_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
image_paths = [p for p in Path(IMAGE_FOLDER).iterdir() if p.suffix.lower() in image_exts]
image_paths.sort()

if not image_paths:
    print("未找到图片")
    sys.exit()

print(f"找到 {len(image_paths)} 张图片")

if not os.path.exists("result"):
    os.mkdir("result")

# 遍历推理
for img_path in tqdm(image_paths):
    # 读取中文路径图片
    img_np = np.fromfile(str(img_path), dtype=np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    if img is None:
        print(f"读取失败: {img_path.name}")
        continue

    # 推理
    results = model(img, conf=CONF_THRES, verbose=False)  # verbose=False 关闭多余打印
    res_img = results[0].plot()

    # 显示
    cv2.imwrite(f"result/{os.path.basename(img_path)}", res_img)

cv2.destroyAllWindows()
print("\n浏览完成！")