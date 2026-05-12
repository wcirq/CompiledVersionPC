import base64
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def read_image_bgr(image_path: str) -> np.ndarray:
    data = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    return image


def write_image_bgr(image_path: str, image_bgr: np.ndarray) -> None:
    path = Path(image_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".png"
    ext = suffix if suffix.startswith(".") else f".{suffix}"
    ok, buf = cv2.imencode(ext, image_bgr)
    if not ok:
        raise ValueError(f"Failed to encode image for {image_path}")
    buf.tofile(str(path))


def image_to_base64(image_bgr: np.ndarray, image_format: str = ".jpg") -> str:
    ok, buf = cv2.imencode(image_format, image_bgr)
    if not ok:
        raise ValueError("Failed to encode image to base64.")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def image_bytes_to_bgr(payload: bytes) -> np.ndarray:
    data = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode uploaded image bytes.")
    return image


def ensure_image_bgr(image_bgr: np.ndarray) -> np.ndarray:
    if not isinstance(image_bgr, np.ndarray):
        raise ValueError("image_bgr must be a numpy.ndarray.")
    if image_bgr.ndim == 2:
        return cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    if image_bgr.ndim != 3:
        raise ValueError("image_bgr must be a 2D grayscale image or a 3D BGR image.")
    if image_bgr.shape[2] == 1:
        return cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    if image_bgr.shape[2] != 3:
        raise ValueError("image_bgr must have 3 channels in BGR order.")
    if image_bgr.dtype != np.uint8:
        image_bgr = np.clip(image_bgr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image_bgr)


def maybe_load_image_bgr(
    image_path: Optional[str] = None,
    image_bytes: Optional[bytes] = None,
    image_bgr: Optional[np.ndarray] = None,
) -> np.ndarray:
    if image_bgr is not None:
        return ensure_image_bgr(image_bgr)
    if image_bytes is not None:
        return image_bytes_to_bgr(image_bytes)
    if image_path:
        return read_image_bgr(image_path)
    raise ValueError("Either image_path, image_bytes, or image_bgr must be provided.")
