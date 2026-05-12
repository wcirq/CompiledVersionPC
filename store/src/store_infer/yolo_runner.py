from __future__ import annotations

import cv2
from typing import Any, Dict, Optional

from ultralytics import YOLO

from store_core.io_utils import image_to_base64, maybe_load_image_bgr

from .base import BaseInferenceRunner


class YoloInferenceRunner(BaseInferenceRunner):
    def __init__(self, config):
        super().__init__(config)
        self._model: Optional[YOLO] = None

    @property
    def model(self) -> YOLO:
        if self._model is None:
            self._model = YOLO(self.config.weight_path)
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
        max_det_value = kwargs.get("max_det")
        device_value = kwargs.get("device")
        include_visualization_base64 = bool(kwargs.get("include_visualization_base64", False))
        conf = float(self.config.conf_threshold if conf_value is None else conf_value)
        iou = float(self.config.iou_threshold if iou_value is None else iou_value)
        max_det = int(self.config.max_det if max_det_value is None else max_det_value)
        device = self.config.device if device_value is None else device_value
        result = self.model.predict(
            source=image_bgr,
            conf=conf,
            iou=iou,
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
            "backend": self.config.backend,
            "task_type": self.config.task_type,
            "image_width": int(image_width),
            "image_height": int(image_height),
            "count": len(detections),
            "detections": detections,
            "conf_threshold": conf,
            "iou_threshold": iou,
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
