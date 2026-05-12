from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
from ultralytics import YOLO

from .io_utils import read_image_bgr
from .package_data import get_default_yolo_weight_path


@dataclass
class SegmentedRoof:
    contour: List[List[int]]
    bbox: List[int]
    crop_bgr: np.ndarray
    masked_full_bgr: np.ndarray
    masked_crop_bgr: np.ndarray
    mask_full: np.ndarray
    mask_crop: np.ndarray
    confidence: float


class TrainRoofSegmenter:
    def __init__(
        self,
        weight_path: Optional[str] = None,
        conf_threshold: float = 0.8,
        device: Optional[str] = None,
    ):
        self.weight_path = weight_path or get_default_yolo_weight_path()
        self.conf_threshold = float(conf_threshold)
        self.device = device
        self.model = YOLO(self.weight_path)

    def segment_image_path(self, image_path: str) -> List[SegmentedRoof]:
        image_bgr = read_image_bgr(image_path)
        return self.segment_image(image_bgr)

    def segment_image(self, image_bgr: np.ndarray) -> List[SegmentedRoof]:
        result = self.model.predict(
            source=image_bgr,
            conf=self.conf_threshold,
            verbose=False,
            device=self.device,
            retina_masks=True,
        )[0]
        if result.masks is None or result.masks.xy is None:
            return []

        roofs: List[SegmentedRoof] = []
        scores = result.boxes.conf.tolist() if result.boxes is not None and result.boxes.conf is not None else []
        image_h, image_w = image_bgr.shape[:2]

        for idx, poly in enumerate(result.masks.xy):
            contour = np.asarray(poly, dtype=np.float32)
            if contour.shape[0] < 3:
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

            roofs.append(
                SegmentedRoof(
                    contour=contour.astype(int).tolist(),
                    bbox=[int(x), int(y), int(x + w), int(y + h)],
                    crop_bgr=crop,
                    masked_full_bgr=masked_full,
                    masked_crop_bgr=masked_crop,
                    mask_full=mask_full,
                    mask_crop=mask_crop,
                    confidence=float(scores[idx]) if idx < len(scores) else 0.0,
                )
            )

        roofs.sort(key=lambda item: (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1]), reverse=True)
        return roofs
