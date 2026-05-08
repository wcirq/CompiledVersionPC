import os
import shutil
import random

def copy_labeled_images(
        source_dir: str,    # 源文件夹（图片+json都在这里）
        target_dir: str,    # 目标文件夹（复制到这里）
        image_exts: list = None
):
    """
    只复制有对应json标注的图片和标签文件到目标目录
    """
    # 默认支持的图片格式
    if image_exts is None:
        image_exts = ['.jpg', '.jpeg', '.png', '.bmp']

    # 自动创建目标文件夹
    os.makedirs(target_dir, exist_ok=True)

    # 遍历源目录所有文件
    for filename in os.listdir(source_dir):
        file_path = os.path.join(source_dir, filename)

        # 只处理文件，跳过文件夹
        if not os.path.isfile(file_path):
            continue

        # 分离文件名和后缀
        name, ext = os.path.splitext(filename)
        ext = ext.lower()  # 统一小写

        # 如果是图片文件
        if ext in image_exts:
            # 检查是否存在同名json
            json_file = os.path.join(source_dir, f"{name}.json")

            # 只有同时存在图片和json才复制
            if os.path.exists(json_file):
                # 复制图片
                shutil.copy2(file_path, os.path.join(target_dir, filename))
                # 复制标签
                shutil.copy2(json_file, os.path.join(target_dir, f"{name}.json"))
                print(f"✅ 已复制：{filename} 和 {name}.json")

# ====================== 【在这里修改路径】 ======================
if __name__ == "__main__":
    # 源目录：你的图片和json所在文件夹
    SOURCE_FOLDER = r"/media/wcirq/data/datasets/成都大弯镇/火车俯拍分割模型数据集/no-null"

    # 目标目录：复制到哪里
    TARGET_FOLDER = r"datasets/train-top/src"

    # 开始复制
    copy_labeled_images(SOURCE_FOLDER, TARGET_FOLDER)
    print("\n🎉 复制完成！只保留了有标注的图片和标签。")