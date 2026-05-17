from __future__ import annotations

import argparse
import cv2
import importlib.util
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from store_core.io_utils import image_to_base64, maybe_load_image_bgr

from .base import BaseInferenceRunner

LOGGER = logging.getLogger(__name__)


class SophonYoloInferenceRunner(BaseInferenceRunner):
    def __init__(self, config, backend_config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.backend_config = dict(backend_config or {})
        self.runtime_backend = "sophon_bmcv"
        self._module = None
        self._sail = None
        self._model = None
        self._dev_id: Optional[int] = None

    @property
    def weight_path(self) -> str:
        weight_path = self.backend_config.get("weight_path")
        if not weight_path:
            raise ValueError(f"Sophon backend weight_path is not configured for model: {self.config.name}")
        return str(weight_path)

    @property
    def python_path(self) -> str:
        python_path = self.backend_config.get("python_path")
        if not python_path:
            raise ValueError(f"Sophon backend python_path is not configured for model: {self.config.name}")
        return str(python_path)

    def _load_demo_module(self):
        if self._module is not None and self._sail is not None:
            return self._module, self._sail
        module_dir = Path(self.python_path)
        if not module_dir.exists():
            raise ValueError(f"Sophon demo python path not found: {module_dir}")
        module_file = module_dir / "yolov8_bmcv.py"
        if not module_file.exists():
            raise ValueError(f"Sophon demo entry not found: {module_file}")
        if str(module_dir) not in sys.path:
            sys.path.insert(0, str(module_dir))
        try:
            import sophon.sail as sail  # type: ignore
        except Exception as exc:
            raise RuntimeError(f"Sophon runtime is unavailable. Please install sophon.sail or use ultralytics backend.{exc}") from exc
        spec = importlib.util.spec_from_file_location("store_infer_sophon_yolov8_bmcv", module_file)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load Sophon module: {module_file}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._module = module
        self._sail = sail
        return module, sail

    def validate_environment(self) -> None:
        _ = self.weight_path
        _ = self.python_path
        self._load_demo_module()

    def _resolve_dev_id(self, device_value: Any = None) -> int:
        if device_value is not None and str(device_value).strip() != "":
            device_text = str(device_value).strip()
            if device_text.isdigit():
                return int(device_text)
        backend_dev_id = self.backend_config.get("dev_id", self.config.device if self.config.device is not None else 0)
        return int(backend_dev_id)

    def _ensure_model(self, device_value: Any = None):
        module, _ = self._load_demo_module()
        dev_id = self._resolve_dev_id(device_value)
        if self._model is None or self._dev_id != dev_id:
            args = argparse.Namespace(
                bmodel=self.weight_path,
                dev_id=dev_id,
                conf_thresh=float(self.config.conf_threshold),
                nms_thresh=float(self.config.iou_threshold),
            )
            self._model = module.YOLOv8(args)
            self._dev_id = dev_id
        return self._model

    def _decode_bm_image(self, input_path: str, dev_id: int):
        _, sail = self._load_demo_module()
        decoder = sail.Decoder(input_path, True, dev_id)
        bmimg = sail.BMImage()
        ret = decoder.read(self._model.handle, bmimg)
        if ret != 0:
            raise RuntimeError(f"Sophon decoder failed to read image: {input_path}")
        return bmimg

    def predict(
        self,
        image_path: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        image_bgr=None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        model = self._ensure_model(kwargs.get("device"))
        image_bgr = maybe_load_image_bgr(image_path=image_path, image_bytes=image_bytes, image_bgr=image_bgr)
        conf_value = kwargs.get("conf_threshold")
        iou_value = kwargs.get("iou_threshold")
        max_det_value = kwargs.get("max_det")
        include_visualization_base64 = bool(kwargs.get("include_visualization_base64", False))
        conf = float(self.config.conf_threshold if conf_value is None else conf_value)
        iou = float(self.config.iou_threshold if iou_value is None else iou_value)
        max_det = int(self.config.max_det if max_det_value is None else max_det_value)
        LOGGER.info(
            "Running inference: model=%s backend=%s dev_id=%d bmodel=%s conf=%.3f iou=%.3f max_det=%d",
            self.config.name,
            self.runtime_backend,
            int(self._dev_id or self._resolve_dev_id(kwargs.get("device"))),
            self.weight_path,
            conf,
            iou,
            max_det,
        )
        model.postprocess.conf_thresh = conf
        model.postprocess.nms_thresh = iou
        model.postprocess.max_det = max_det

        temp_path: Optional[str] = None
        input_path = image_path
        if not input_path:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                temp_path = temp_file.name
            if not cv2.imwrite(temp_path, image_bgr):
                raise RuntimeError("Failed to write temporary image for Sophon inference.")
            input_path = temp_path

        try:
            bmimg = self._decode_bm_image(str(input_path), int(self._dev_id or 0))
            det = model([bmimg])[0]
        finally:
            if temp_path:
                Path(temp_path).unlink(missing_ok=True)

        names = list(self.config.class_names)
        detections = []
        for row in det.tolist() if det is not None else []:
            x1, y1, x2, y2, score, category_id = row[:6]
            class_id = int(category_id)
            class_name = names[class_id] if 0 <= class_id < len(names) else str(class_id)
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": float(score),
                    "box": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
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
            "imgsz": int(model.net_w),
            "max_det": max_det,
            "device": int(self._dev_id or 0),
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
