import os
import json
from PIL import Image
import shutil
import random

def generate_empty_labels(source_dir: str, target_dir: str):
    """
    为源目录所有图片生成空标注json，并将图片+json复制到目标目录
    """
    # 支持的图片格式
    image_exts = ['.jpg', '.jpeg', '.png', '.bmp']

    # 自动创建目标文件夹
    os.makedirs(target_dir, exist_ok=True)

    files = os.listdir(source_dir)
    random.shuffle(files)

    # 遍历源目录所有文件
    for filename in files[:120]:
        file_path = os.path.join(source_dir, filename)

        # 跳过文件夹
        if not os.path.isfile(file_path):
            continue

        name, ext = os.path.splitext(filename)
        ext = ext.lower()

        # 只处理图片
        if ext in image_exts:
            try:
                # 1. 读取图片尺寸（宽、高）
                with Image.open(file_path) as img:
                    width, height = img.size

                # 2. 构建空标注json内容（完全按你给的格式）
                label_content = {
                    "version": "6.1.3",
                    "flags": {},
                    "shapes": [],  # 空标注
                    "imagePath": filename,
                    "imageData": None,
                    "imageHeight": height,
                    "imageWidth": width
                }

                # 3. 目标路径
                target_img_path = os.path.join(target_dir, filename)
                target_json_path = os.path.join(target_dir, f"{name}.json")

                # 4. 复制图片
                shutil.copy2(file_path, target_img_path)

                # 5. 保存空标注json
                with open(target_json_path, 'w', encoding='utf-8') as f:
                    json.dump(label_content, f, indent=2, ensure_ascii=False)

                print(f"✅ 处理完成：{filename} | 尺寸：{width}x{height}")

            except Exception as e:
                print(f"❌ 处理失败 {filename}：{str(e)}")

# ====================== 【在这里修改路径】 ======================
if __name__ == "__main__":
    # 源目录：存放原始图片的文件夹
    SOURCE_DIR = r"/media/wcirq/data/datasets/成都大弯镇/火车俯拍分割模型数据集/null"

    # 目标目录A：图片和空标注json都保存到这里
    TARGET_DIR = r"datasets/train-top/src-null"

    # 开始执行
    generate_empty_labels(SOURCE_DIR, TARGET_DIR)
    print("\n🎉 全部处理完成！图片 + 空标注已生成到目标目录。")