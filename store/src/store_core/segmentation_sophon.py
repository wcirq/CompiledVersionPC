from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .package_data import get_default_sophon_seg_demo_path
from .runtime_backend import sophon_sail_available


class SophonTrainRoofSegmenter:
    def __init__(
        self,
        weight_path: str,
        conf_threshold: float = 0.4,
        dev_id: int = 0,
        demo_path: Optional[str] = None,
    ):
        self.weight_path = str(weight_path)
        self.conf_threshold = float(conf_threshold)
        self.dev_id = int(dev_id)
        self.demo_path = str(demo_path or get_default_sophon_seg_demo_path())
        self._module = None
        self._sail = None
        self._model = None
        self._model_conf_threshold: Optional[float] = None

    def _load_demo_module(self):
        if self._module is not None and self._sail is not None:
            return self._module, self._sail
        if not sophon_sail_available():
            raise RuntimeError("Sophon runtime is unavailable.")
        module_dir = Path(self.demo_path)
        module_file = module_dir / "yolov8_bmcv.py"
        if not module_dir.exists():
            raise ValueError(f"Sophon segmentation demo path not found: {module_dir}")
        if not module_file.exists():
            raise ValueError(f"Sophon segmentation demo entry not found: {module_file}")
        if str(module_dir) not in sys.path:
            sys.path.insert(0, str(module_dir))
        import sophon.sail as sail  # type: ignore

        spec = importlib.util.spec_from_file_location("store_core_sophon_yolov8_seg_bmcv", module_file)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load Sophon segmentation module: {module_file}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._module = module
        self._sail = sail
        return module, sail

    def _ensure_model(self, conf_threshold: Optional[float] = None):
        module, _ = self._load_demo_module()
        active_conf_threshold = self.conf_threshold if conf_threshold is None else float(conf_threshold)
        if self._model is None or self._model_conf_threshold != active_conf_threshold:
            args = argparse.Namespace(
                bmodel=self.weight_path,
                dev_id=self.dev_id,
                conf_thresh=active_conf_threshold,
                nms_thresh=0.7,
            )
            self._model = module.Yolov8Seg(args)
            self._model_conf_threshold = active_conf_threshold
        return self._model

    def _decode_bm_image(self, input_path: str, conf_threshold: Optional[float] = None):
        _, sail = self._load_demo_module()
        model = self._ensure_model(conf_threshold=conf_threshold)
        decoder = sail.Decoder(input_path, True, self.dev_id)
        bmimg = sail.BMImage()
        ret = decoder.read(model.handle, bmimg)
        if ret != 0:
            raise RuntimeError(f"Sophon decoder failed to read image: {input_path}")
        return bmimg

    @staticmethod
    def _normalize_segment_polygon(polygon: object) -> Optional[np.ndarray]:
        contour = np.asarray(polygon, dtype=np.float32)
        if contour.ndim == 1 and contour.size >= 6 and contour.size % 2 == 0:
            contour = contour.reshape(-1, 2)
        elif contour.ndim == 2 and contour.shape[0] == 1 and contour.shape[1] >= 6 and contour.shape[1] % 2 == 0:
            contour = contour.reshape(-1, 2)
        elif contour.ndim == 3 and contour.shape[0] == 1 and contour.shape[2] == 2:
            contour = contour.reshape(-1, 2)
        elif contour.ndim == 2 and contour.shape[1] == 2:
            contour = contour.reshape(-1, 2)
        else:
            return None
        if contour.shape[0] < 3:
            return None
        return contour

    def segment_image(self, image_bgr: np.ndarray, conf_threshold: Optional[float] = None) -> List[dict]:
        model = self._ensure_model(conf_threshold=conf_threshold)
        temp_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                temp_path = temp_file.name
            if not cv2.imwrite(temp_path, image_bgr):
                raise RuntimeError("Failed to write temporary image for Sophon segmentation.")
            bmimg = self._decode_bm_image(temp_path, conf_threshold=conf_threshold)
            results = model([bmimg])[0]
        finally:
            if temp_path:
                Path(temp_path).unlink(missing_ok=True)

        boxes, segments, _ = results
        segment_items = segments if segments is not None else []
        payloads: List[dict] = []
        image_h, image_w = image_bgr.shape[:2]
        for index, polygon in enumerate(segment_items):
            contour = self._normalize_segment_polygon(polygon)
            if contour is None:
                continue
            contour = np.round(contour).astype(np.int32)
            contour[:, 0] = np.clip(contour[:, 0], 0, image_w - 1)
            contour[:, 1] = np.clip(contour[:, 1], 0, image_h - 1)
            x, y, w, h = cv2.boundingRect(contour)
            if w <= 1 or h <= 1:
                continue

            mask_full = np.zeros((image_h, image_w), dtype=np.uint8)
            cv2.fillPoly(mask_full, [contour], 255)
            masked_full = np.full_like(image_bgr, 255)
            masked_full[mask_full > 0] = image_bgr[mask_full > 0]

            crop = image_bgr[y:y + h, x:x + w].copy()
            mask_crop = mask_full[y:y + h, x:x + w].copy()
            masked_crop = np.full_like(crop, 255)
            masked_crop[mask_crop > 0] = crop[mask_crop > 0]

            confidence = 0.0
            if boxes is not None and len(boxes) > index:
                row = np.asarray(boxes[index]).reshape(-1)
                if row.size >= 5:
                    confidence = float(row[4])
            payloads.append(
                {
                    "contour": contour.astype(int).tolist(),
                    "bbox": [int(x), int(y), int(x + w), int(y + h)],
                    "crop_bgr": crop,
                    "masked_full_bgr": masked_full,
                    "masked_crop_bgr": masked_crop,
                    "mask_full": mask_full,
                    "mask_crop": mask_crop,
                    "confidence": confidence,
                }
            )
        payloads.sort(key=lambda item: (item["bbox"][2] - item["bbox"][0]) * (item["bbox"][3] - item["bbox"][1]), reverse=True)
        return payloads
