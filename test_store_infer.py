"""`store_infer` 推理接口示例脚本。

这个文件用于验证“通用检测模型推理注册表”相关能力，重点不是训练和模型仓库，
而是已经注册好的检测模型能否直接推理。

当前脚本包含两个最常用的演示：

1. `test_list_inference_models()`
   列出当前环境中可用的检测模型，确认注册表加载是否正常。

2. `test_run_fire_smoke()`
   调用名为 `fire_smoke` 的烟火识别模型对单张图做推理，并把可视化结果从 base64 解码出来。

这个脚本适合：
- 检查 `store_infer` 的模型发现机制是否正常
- 验证某个推理模型是否能跑通
- 对接前端推理页面前，先在本地脚本里做一次最小闭环验证
"""

import cv2
import base64
import numpy as np
from store_infer import list_models, run_inference


def _resize_for_display(image, max_width=1280, max_height=720):
    """按显示器尺寸缩放图片，仅用于本地调试显示。"""
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)

    if scale < 1.0:
        return cv2.resize(
            image,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return image


def test_list_inference_models():
    """打印当前所有已注册的推理模型。

    适合在以下场景快速确认：
    - 推理模型是否被正确发现
    - 模型注册表是否加载成功
    - 前端“检测测试”页面理论上应该能看到哪些模型
    """
    print(list_models())


def test_run_fire_smoke():
    """运行一个 `fire_smoke` 烟火识别示例。

    这个函数演示了 `run_inference()` 的最基本调用方式：
    - 指定模型名
    - 指定图片路径
    - 指定置信度阈值
    - 要求返回可视化结果的 base64

    返回结果里除了检测框和类别信息外，还会包含一张可视化图。
    这里把它解码成 OpenCV 图像，方便后续保存或显示。
    """
    result = run_inference(
        model_name="fire_smoke",
        # image_path="weights/imgs/cf177220111084e5238e303fce537a20.jpg",
        image_path="cf177220111084e5238e303fce537a20.jpg",
        conf_threshold=0.25,
        include_visualization_base64=True,
    )
    visualization_base64 = result["visualization_base64"]
    result['visualization_base64'] = ""
    print(result)
    image_bytes = base64.b64decode(visualization_base64)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    visualization_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    # 如果要本地弹窗查看推理可视化结果，可以取消下面两行的注释。
    # cv2.imshow("visualization_image", _resize_for_display(visualization_image))
    # cv2.waitKey(0)


if __name__ == "__main__":
    # 默认先列出模型，再跑一个最小推理示例。
    test_list_inference_models()
    test_run_fire_smoke()
