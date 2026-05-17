from __future__ import annotations

import cv2
import logging
from typing import Any, Dict, Optional

from ultralytics import YOLO

from store_core.io_utils import image_to_base64, maybe_load_image_bgr

from .base import BaseInferenceRunner

LOGGER = logging.getLogger(__name__)


class YoloInferenceRunner(BaseInferenceRunner):
    def __init__(self, config, backend_config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.backend_config = dict(backend_config or {})
        self.runtime_backend = "ultralytics"
        self._model: Optional[YOLO] = None

    @property
    def model(self) -> YOLO:
        if self._model is None:
            weight_path = self.backend_config.get("weight_path") or self.config.weight_path
            if not weight_path:
                raise ValueError(f"Ultralytics backend weight_path is not configured for model: {self.config.name}")
            self._model = YOLO(weight_path)
        return self._model

    def predict(
        self,
        image_path: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        image_bgr=None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        image_bgr = maybe_load_image_bgr(image_path=image_path, image_bytes=image_bytes, image_bgr=image_bgr)
        conf_value = kwargs.get("conf_threshold")
        iou_value = kwargs.get("iou_threshold")
        imgsz_value = kwargs.get("imgsz")
        max_det_value = kwargs.get("max_det")
        device_value = kwargs.get("device")
        include_visualization_base64 = bool(kwargs.get("include_visualization_base64", False))
        conf = float(self.config.conf_threshold if conf_value is None else conf_value)
        iou = float(self.config.iou_threshold if iou_value is None else iou_value)
        imgsz = int(self.config.imgsz if imgsz_value is None else imgsz_value)
        max_det = int(self.config.max_det if max_det_value is None else max_det_value)
        device = self.config.device if device_value is None else device_value
        weight_path = self.backend_config.get("weight_path") or self.config.weight_path
        LOGGER.info(
            "Running inference: model=%s backend=%s device=%s weight=%s conf=%.3f iou=%.3f imgsz=%d max_det=%d",
            self.config.name,
            self.runtime_backend,
            device,
            weight_path,
            conf,
            iou,
            imgsz,
            max_det,
        )
        result = self.model.predict(
            source=image_bgr,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            max_det=max_det,
            device=device,
            verbose=False,
        )[0]

        names = self.config.class_names or [str(name) for name in getattr(result, "names", {}).values()]
        detections = []
        boxes = result.boxes
        if boxes is not None:
            xyxy_list = boxes.xyxy.tolist() if boxes.xyxy is not None else []
            conf_list = boxes.conf.tolist() if boxes.conf is not None else []
            cls_list = boxes.cls.tolist() if boxes.cls is not None else []
            for index, box in enumerate(xyxy_list):
                class_id = int(cls_list[index]) if index < len(cls_list) else -1
                class_name = (
                    names[class_id]
                    if 0 <= class_id < len(names)
                    else str(class_id)
                )
                detections.append(
                    {
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": float(conf_list[index]) if index < len(conf_list) else 0.0,
                        "box": [int(round(value)) for value in box[:4]],
                    }
                )

        image_height, image_width = image_bgr.shape[:2]
        payload = {
            "model_name": self.config.name,
            "backend": self.runtime_backend,
            "task_type": self.config.task_type,
            "image_width": int(image_width),
            "image_height": int(image_height),
            "count": len(detections),
            "detections": detections,
            "conf_threshold": conf,
            "iou_threshold": iou,
            "imgsz": imgsz,
            "max_det": max_det,
        }
        if include_visualization_base64:
            payload["visualization_base64"] = image_to_base64(self._draw_detections(image_bgr, detections))
        return payload

    def _draw_detections(self, image_bgr, detections):
        annotated = image_bgr.copy()
        base_size = max(annotated.shape[:2])
        line_width = max(2, int(round(base_size / 500)))
        font_scale = max(0.7, base_size / 1400.0)
        font_thickness = max(2, int(round(base_size / 700)))
        for index, item in enumerate(detections, start=1):
            x1, y1, x2, y2 = item["box"]
            color = (35, 124, 255) if item["class_name"] == "fire" else (71, 196, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_width)
            label = f'{index}. {item["class_name"]} {item["confidence"]:.3f}'
            text_y = max(y1 - 10, 24)
            cv2.putText(
                annotated,
                label,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                font_thickness,
                cv2.LINE_AA,
            )
        return annotated
