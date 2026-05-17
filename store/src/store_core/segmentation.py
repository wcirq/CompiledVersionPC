from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
from ultralytics import YOLO

from .io_utils import read_image_bgr
from .package_data import get_default_bm_yolo_weight_path, get_default_yolo_weight_path
from .runtime_backend import resolve_runtime_backend
from .segmentation_sophon import SophonTrainRoofSegmenter


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
        torch_weight_path: Optional[str] = None,
        bm_weight_path: Optional[str] = None,
        conf_threshold: float = 0.4,
    ):
        runtime = resolve_runtime_backend()
        self.runtime_backend = runtime.backend
        self.torch_weight_path = torch_weight_path or get_default_yolo_weight_path()
        self.bm_weight_path = bm_weight_path or get_default_bm_yolo_weight_path()
        self.conf_threshold = float(conf_threshold)
        self.device = runtime.torch_device
        self.model = None
        self._bm_segmenter = None

    def segment_image_path(self, image_path: str, conf_threshold: Optional[float] = None) -> List[SegmentedRoof]:
        image_bgr = read_image_bgr(image_path)
        return self.segment_image(image_bgr, conf_threshold=conf_threshold)

    def _build_segmented_roof(self, image_bgr: np.ndarray, contour: np.ndarray, confidence: float) -> Optional[SegmentedRoof]:
        image_h, image_w = image_bgr.shape[:2]
        if contour.shape[0] < 3:
            return None
        contour = np.round(contour).astype(np.int32)
        contour[:, 0] = np.clip(contour[:, 0], 0, image_w - 1)
        contour[:, 1] = np.clip(contour[:, 1], 0, image_h - 1)
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 1 or h <= 1:
            return None

        mask_full = np.zeros((image_h, image_w), dtype=np.uint8)
        cv2.fillPoly(mask_full, [contour], 255)
        masked_full = np.full_like(image_bgr, 255)
        masked_full[mask_full > 0] = image_bgr[mask_full > 0]

        crop = image_bgr[y:y + h, x:x + w].copy()
        mask_crop = mask_full[y:y + h, x:x + w].copy()
        masked_crop = np.full_like(crop, 255)
        masked_crop[mask_crop > 0] = crop[mask_crop > 0]

        return SegmentedRoof(
            contour=contour.astype(int).tolist(),
            bbox=[int(x), int(y), int(x + w), int(y + h)],
            crop_bgr=crop,
            masked_full_bgr=masked_full,
            masked_crop_bgr=masked_crop,
            mask_full=mask_full,
            mask_crop=mask_crop,
            confidence=float(confidence),
        )

    def _segment_with_torch(self, image_bgr: np.ndarray, conf_threshold: Optional[float] = None) -> List[SegmentedRoof]:
        if self.model is None:
            self.model = YOLO(self.torch_weight_path)
        active_conf_threshold = self.conf_threshold if conf_threshold is None else float(conf_threshold)
        result = self.model.predict(
            source=image_bgr,
            conf=active_conf_threshold,
            verbose=False,
            device=self.device,
            retina_masks=True,
        )[0]
        if result.masks is None or result.masks.xy is None:
            return []

        roofs: List[SegmentedRoof] = []
        scores = result.boxes.conf.tolist() if result.boxes is not None and result.boxes.conf is not None else []
        for idx, poly in enumerate(result.masks.xy):
            roof = self._build_segmented_roof(
                image_bgr=image_bgr,
                contour=np.asarray(poly, dtype=np.float32),
                confidence=float(scores[idx]) if idx < len(scores) else 0.0,
            )
            if roof is not None:
                roofs.append(roof)
        return roofs

    def _segment_with_bm(self, image_bgr: np.ndarray, conf_threshold: Optional[float] = None) -> List[SegmentedRoof]:
        if self._bm_segmenter is None:
            self._bm_segmenter = SophonTrainRoofSegmenter(
                weight_path=self.bm_weight_path,
                conf_threshold=self.conf_threshold,
            )
        roofs: List[SegmentedRoof] = []
        for item in self._bm_segmenter.segment_image(image_bgr, conf_threshold=conf_threshold):
            roofs.append(
                SegmentedRoof(
                    contour=item["contour"],
                    bbox=item["bbox"],
                    crop_bgr=item["crop_bgr"],
                    masked_full_bgr=item["masked_full_bgr"],
                    masked_crop_bgr=item["masked_crop_bgr"],
                    mask_full=item["mask_full"],
                    mask_crop=item["mask_crop"],
                    confidence=float(item["confidence"]),
                )
            )
        return roofs

    def segment_image(self, image_bgr: np.ndarray, conf_threshold: Optional[float] = None) -> List[SegmentedRoof]:
        if self.runtime_backend == "bm":
            roofs = self._segment_with_bm(image_bgr, conf_threshold=conf_threshold)
        else:
            roofs = self._segment_with_torch(image_bgr, conf_threshold=conf_threshold)

        roofs.sort(key=lambda item: (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1]), reverse=True)
        return roofs
