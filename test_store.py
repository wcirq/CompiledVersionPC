"""`store_core` 端到端示例测试脚本。

这个文件主要用于手工验证“异物检测模型仓库”相关能力是否可用，覆盖三类典型场景：

1. 初始化 `TrainRoofAnomalyStore`
   用于确认本地模型仓库、后台服务、分割模型等是否能正常启动。

2. 训练模型
   使用一批正常车顶模板图训练一个新的异物检测模型，并把进度事件打印到终端。

3. 检测与结果可视化
   对单张图、整目录图片做异物检测，并把热力图、轮廓、框、批量输出文件保存出来。

这个脚本更偏“开发调试/人工验收”，不是自动化单元测试：
- 会直接打印大量结果
- 会弹 OpenCV 窗口
- 默认依赖本地图片、模型目录和 `store_data`

如果只是想快速验证一个已经训练好的模型，通常直接运行：

```bash
python test_store.py
```

再按需取消 `__main__` 里对应函数的注释即可。
"""

import base64
import cv2
import os
import numpy as np
from store_core import TrainRoofAnomalyStore
from store_core.schemas import RuntimeOptions


def _resize_for_display(image, max_width=1280, max_height=720):
    """按显示器尺寸缩放图片，仅用于 OpenCV 弹窗预览。

    这个函数不会影响检测结果，只是避免原图或热力图太大时窗口超出屏幕。
    """
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
    """根据图像分辨率动态生成标注样式。

    目的是让大图和小图在显示轮廓、框、文字时都保持相对合理的线宽和字号。
    """
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
    """初始化 `TrainRoofAnomalyStore` 并打印后台服务信息。

    适合在排查以下问题时单独调用：
    - `store_data` 目录是否正常
    - 后台 FastAPI 服务是否能自动拉起
    - 基础依赖是否安装完整
    """
    store = TrainRoofAnomalyStore(
        root_dir="./store_data",
        autostart_service=True,
        service_port=55555,
        yolo_conf_threshold=0.5
    )

    print(store.service_info)
    return store


def test_store_train(store: TrainRoofAnomalyStore):
    """训练一个新的异物检测模型。

    使用说明：
    - `image_dir` 指向训练图片目录，通常是正常车顶模板图
    - `runtime_options` 可按需修改切图、阈值、后处理等参数
    - `on_progress` 会把训练进度事件直接打印出来，方便排查训练卡在哪个阶段

    返回值：
    - 训练成功后返回新模型的 `model_id`
    """

    def on_progress(event: dict):
        print(event)

    options = RuntimeOptions(
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
    """对单张图片做异物检测，并显示热力图与标注结果。

    这个函数主要用于人工观察：
    - 车顶分割轮廓是否正确
    - 异常热力图位置是否合理
    - 异常框与轮廓是否贴合

    参数：
    - `model_id`: 要加载的模型 ID；留空时使用脚本里的默认示例模型
    - `image_path`: 待检测图片路径；留空时使用当前目录下的 `test5.jpg`

    返回值：
    - `(annotated_image, heatmap_base64)`
      方便外部继续保存或复用结果
    """
    if image_path is None:
        # image_path = "test_imgs2/test_h7x_DBXS_BAD/test5.jpg"
        image_path = "test5.jpg"
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

    # 后端返回的热力图是 base64 JPEG；这里解码成 OpenCV BGR 图像方便显示。
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

    # 绿色：车顶分割轮廓，用于确认 ROI 是否合理。
    for contour in result.get("roof_contours", []):
        contour_array = np.array(contour, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(
            annotated_image,
            [contour_array],
            True,
            (0, 255, 0),
            style["roof_thickness"],
        )

    # 红色：异常区域轮廓；黄色：异常外接框和分数标签。
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
    # 分别显示热力图和标注结果，显示时按屏幕范围缩小，不改变原始结果。
    cv2.imshow("detect_image heatmap", _resize_for_display(heatmap_image))
    cv2.imshow("detect_image annotations", _resize_for_display(annotated_image))
    print("按任意键关闭图片窗口")
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    return annotated_image, heatmap_base64


def test_store_batch_predict(store, model_id):
    """对一个目录下的全部图片逐张检测，并把标注图保存回目录。

    这是最简单的批量人工验收脚本：
    - 输入目录：`test_imgs2/test_h7x_DBXS_BAD`
    - 输出：把每张图的标注结果另存为 `00000_annotated.jpg` 这类文件

    适合快速扫一遍一批测试图的检测效果。
    """
    root = "test_imgs2/test_h7x_DBXS_BAD"
    files = os.listdir(root)
    for i, file in enumerate(files):
        annotated_image, heatmap_base64 = test_store_predict(store, model_id=model_id, image_path=os.path.join(root, file))
        # cv2.imwrite(os.path.join(root, f"{str(i).zfill(5)}_heatmap.jpg"), heatmap_base64)
        cv2.imwrite(os.path.join(root, f"{str(i).zfill(5)}_annotated.jpg"), annotated_image)


def test_store_detect_and_save_results(store: TrainRoofAnomalyStore, model_id=None):
    """调用 `detect_and_save_results()` 做目录级批量检测并保存标准输出文件。

    和 `test_store_batch_predict()` 的区别：
    - 这里直接调用后端封装好的批量保存接口
    - 会额外输出热力图、叠加图、异常裁剪图、标注 txt 等文件
    - 更适合验证部署输出格式是否符合预期
    """
    result = store.detect_and_save_results(
        model_id="model_e8ad3cf30cd14a10" if model_id is None else model_id,
        image_dir="test_imgs2/test_h7x_DBXS_BAD",
        output_dir="test_imgs2/test_h7x_DBXS_BAD_store_results",
        threshold=22,
        use_segmentation=False,
        save_process_files=True,
    )
    print(f"批量检测完成，共处理 {result['count']} 张图片，输出目录: {result['output_dir']}")
    for item in result["items"]:
        print(item["image_path"])
        for saved in item["saved_files"]:
            print("  ->", saved)
    return result


if __name__ == '__main__':
    # `__main__` 保留一套最常用的人工测试入口。
    #
    # 默认行为：
    # - 启动 `TrainRoofAnomalyStore`
    # - 直接对一张测试图做检测并弹窗显示结果
    #
    # 如果要测试训练、批量检测或批量保存，可以取消下面对应函数的注释。
    model_id = "model_f8e53c97ce004b22"
    store = TrainRoofAnomalyStore(
        root_dir="./store_data",
        autostart_service=True,
        service_port=55555,
        yolo_conf_threshold=0.2,
    )
    # model_id = test_store_train(store)
    test_store_predict(store, model_id=model_id)
    # test_store_detect_and_save_results(store, model_id=model_id)
    # test_store_batch_predict(store, model_id=model_id)

    store.serve_forever()
