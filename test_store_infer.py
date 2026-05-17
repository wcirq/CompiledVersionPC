import cv2
import base64
import numpy as np
from store_infer import list_models, run_inference


def _resize_for_display(image, max_width=1280, max_height=720):
    # 仅缩放显示窗口中的图像，避免大图弹窗超出屏幕。
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
    print(list_models())


def test_run_fire_smoke():
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
    # cv2.imshow("visualization_image", _resize_for_display(visualization_image))
    # cv2.waitKey(0)


if __name__ == "__main__":
    test_list_inference_models()
    test_run_fire_smoke()
