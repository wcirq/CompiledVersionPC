import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
STORE_SRC = ROOT / "store" / "src"
if str(STORE_SRC) not in sys.path:
    sys.path.insert(0, str(STORE_SRC))

from store_core.io_utils import read_image_bgr, write_image_bgr
from store_core.package_data import get_default_bm_yolo_weight_path
from store_core.segmentation_sophon import SophonTrainRoofSegmenter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="测试 BM 列车顶部分割模型是否能正常输出轮廓。")
    parser.add_argument("image", help="待测试图片路径")
    parser.add_argument(
        "--bmodel",
        default=get_default_bm_yolo_weight_path(),
        help="BM 分割 bmodel 路径",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="分割置信度阈值",
    )
    parser.add_argument(
        "--dev-id",
        type=int,
        default=0,
        help="BM 设备 ID",
    )
    parser.add_argument(
        "--save",
        default="bm_seg_result.jpg",
        help="可视化结果保存路径",
    )
    return parser


def draw_result(image_bgr: np.ndarray, roofs: list[dict]) -> np.ndarray:
    annotated = image_bgr.copy()
    base = max(annotated.shape[:2])
    line_thickness = max(2, int(round(base / 500)))
    box_thickness = max(2, int(round(base / 450)))
    font_scale = max(0.7, base / 1400.0)
    font_thickness = max(2, int(round(base / 700)))

    for index, roof in enumerate(roofs, start=1):
        contour = np.asarray(roof["contour"], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(annotated, [contour], True, (0, 255, 0), line_thickness)
        x1, y1, x2, y2 = roof["bbox"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), box_thickness)
        label = f"R{index}:{float(roof['confidence']):.3f}"
        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )
    return annotated


def main() -> int:
    args = build_parser().parse_args()
    image_bgr = read_image_bgr(args.image)
    segmenter = SophonTrainRoofSegmenter(
        weight_path=args.bmodel,
        conf_threshold=args.conf,
        dev_id=args.dev_id,
    )

    roofs = segmenter.segment_image(image_bgr)
    print(f"image: {args.image}")
    print(f"bmodel: {args.bmodel}")
    print(f"conf_threshold: {args.conf}")
    print(f"dev_id: {args.dev_id}")
    print(f"roof_count: {len(roofs)}")

    if not roofs:
        print("result: no roof contour detected")
        return 1

    for index, roof in enumerate(roofs, start=1):
        print(
            f"roof[{index}]: "
            f"confidence={float(roof['confidence']):.4f}, "
            f"bbox={roof['bbox']}, "
            f"points={len(roof['contour'])}"
        )

    annotated = draw_result(image_bgr, roofs)
    write_image_bgr(args.save, annotated)
    print(f"saved: {args.save}")
    print("result: success")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
