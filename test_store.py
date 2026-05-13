import base64
import cv2
import os
import numpy as np
from store_core import TrainRoofAnomalyStore
from store_core.schemas import RuntimeOptions


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


def _get_annotation_style(image):
    height, width = image.shape[:2]
    base = max(height, width)
    line_thickness = max(2, int(round(base / 500)))
    box_thickness = max(2, int(round(base / 450)))
    font_scale = max(0.7, base / 1400.0)
    font_thickness = max(2, int(round(base / 700)))
    return {
        "roof_thickness": line_thickness,
        "anomaly_thickness": line_thickness,
        "box_thickness": box_thickness,
        "font_scale": font_scale,
        "font_thickness": font_thickness,
    }


def test_store_init():
    store = TrainRoofAnomalyStore(
        root_dir="./store_data",
        autostart_service=True,
        service_port=55555,
        yolo_conf_threshold=0.8
    )

    print(store.service_info)
    return store


def test_store_train(store: TrainRoofAnomalyStore):

    def on_progress(event: dict):
        print(event)

    options = RuntimeOptions(
        device="cuda",
        knn_backend="faiss",
        crop_size=(640, 640),
        stride=(512, 512),
        threshold_quantile=0.99,
    )

    result = store.train_model(
        model_name="综合列车顶异物检测",
        image_dir="templates2",
        runtime_options=options,
        calibrate_dir=None,
        progress_callback=on_progress,
    )

    print(result["model_id"])
    return result["model_id"]


def test_store_predict(store: TrainRoofAnomalyStore, model_id=None, image_path=None):
    if image_path is None:
        image_path = "test_imgs2/test_h7x_DBXS_BAD/test5.jpg"
    result = store.detect_image(
        model_id="model_e8ad3cf30cd14a10" if model_id is None else model_id,
        image_path=image_path,
        include_heatmap_base64=True,
        threshold=22
    )

    heatmap_base64 = result.get("heatmap_base64")
    if not heatmap_base64:
        print(f"detect_image 未返回 heatmap_base64, {result['message']}")
        return

    # base64 热力图解码为 OpenCV 可显示的 BGR 图像。
    result["heatmap_base64"] = ""
    image_bytes = base64.b64decode(heatmap_base64)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    heatmap_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if heatmap_image is None:
        print("heatmap_base64 解码失败")
        return

    annotated_image = cv2.imread(image_path)
    if annotated_image is None:
        print(f"原图读取失败: {image_path}")
        return

    style = _get_annotation_style(annotated_image)

    # 绿色绘制车顶分割轮廓。
    for contour in result.get("roof_contours", []):
        contour_array = np.array(contour, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(
            annotated_image,
            [contour_array],
            True,
            (0, 255, 0),
            style["roof_thickness"],
        )

    # 红色绘制异常区域轮廓，黄色绘制外接框和分数。
    for index, region in enumerate(result.get("anomaly_regions", []), start=1):
        contour = region.get("contour", [])
        if contour:
            contour_array = np.array(contour, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(
                annotated_image,
                [contour_array],
                True,
                (0, 0, 255),
                style["anomaly_thickness"],
            )

        box = region.get("box", [])
        if len(box) == 4:
            x1, y1, x2, y2 = box
            cv2.rectangle(
                annotated_image,
                (x1, y1),
                (x2, y2),
                (0, 255, 255),
                style["box_thickness"],
            )
            score = region.get("score")
            label = f"A{index}"
            if score is not None:
                label = f"{label}:{score:.3f}"
            text_y = max(y1 - 8, 20)
            cv2.putText(
                annotated_image,
                label,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                style["font_scale"],
                (0, 255, 255),
                style["font_thickness"],
                cv2.LINE_AA,
            )

    print(result)
    # 分别显示热力图和标注结果，两个窗口都按屏幕范围缩小显示。
    cv2.imshow("detect_image heatmap", _resize_for_display(heatmap_image))
    cv2.imshow("detect_image annotations", _resize_for_display(annotated_image))
    print("按任意键关闭图片窗口")
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    return annotated_image, heatmap_base64


def test_store_batch_predict(store, model_id):
    root = "test_imgs2/test_h7x_DBXS_BAD"
    files = os.listdir(root)
    for i, file in enumerate(files):
        annotated_image, heatmap_base64 = test_store_predict(store, model_id=model_id, image_path=os.path.join(root, file))
        # cv2.imwrite(os.path.join(root, f"{str(i).zfill(5)}_heatmap.jpg"), heatmap_base64)
        cv2.imwrite(os.path.join(root, f"{str(i).zfill(5)}_annotated.jpg"), annotated_image)


if __name__ == '__main__':
    model_id = "model_67dd6bdb82a346f6"
    store = TrainRoofAnomalyStore(
        root_dir="./store_data",
        autostart_service=True,
        service_port=55555,
        yolo_conf_threshold=0.2,
    )
    # model_id = test_store_train(store)
    test_store_predict(store, model_id=model_id)
    test_store_batch_predict(store, model_id=model_id)

    store.serve_forever()
