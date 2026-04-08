from typing import Tuple

try:
    import cv2
    import numpy as np
except Exception as exc:
    from .debug_utils import print_exception_details

    print_exception_details(exc, context="engine.augment import failed")
    raise


def vertical_flip(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.flip(image_rgb, 0)


def rotate_image(image_rgb: np.ndarray, angle_range: Tuple[float, float], border_value: int = 255) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    angle = np.random.uniform(angle_range[0], angle_range[1])
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(
        image_rgb,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(border_value, border_value, border_value),
    )


def adjust_brightness(image_rgb: np.ndarray, brightness_limit: float = 0.08) -> np.ndarray:
    img = image_rgb.astype(np.float32)
    beta = 255.0 * np.random.uniform(-brightness_limit, brightness_limit)
    return np.clip(img + beta, 0, 255).astype(np.uint8)


def adjust_contrast(image_rgb: np.ndarray, contrast_limit: float = 0.08) -> np.ndarray:
    img = image_rgb.astype(np.float32)
    alpha = 1.0 + np.random.uniform(-contrast_limit, contrast_limit)
    return np.clip(img * alpha, 0, 255).astype(np.uint8)


def adjust_saturation(image_rgb: np.ndarray, saturation_limit: float = 0.08) -> np.ndarray:
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    scale = 1.0 + np.random.uniform(-saturation_limit, saturation_limit)
    hsv[..., 1] = np.clip(hsv[..., 1] * scale, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def gaussian_blur(image_rgb: np.ndarray, ksize_options: Tuple[int, ...] = (3, 5), sigma_range: Tuple[float, float] = (0.1, 1.2)) -> np.ndarray:
    ksize = int(np.random.choice(ksize_options))
    if ksize % 2 == 0:
        ksize += 1
    sigma = np.random.uniform(sigma_range[0], sigma_range[1])
    return cv2.GaussianBlur(image_rgb, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)


def gaussian_noise(image_rgb: np.ndarray, sigma_range: Tuple[float, float] = (2.0, 8.0)) -> np.ndarray:
    sigma = np.random.uniform(sigma_range[0], sigma_range[1])
    noise = np.random.normal(0, sigma, image_rgb.shape).astype(np.float32)
    return np.clip(image_rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def color_shift(image_rgb: np.ndarray, shift_limit: int = 8) -> np.ndarray:
    img = image_rgb.astype(np.int16)
    shifts = np.random.randint(-shift_limit, shift_limit + 1, size=(1, 1, 3), dtype=np.int16)
    return np.clip(img + shifts, 0, 255).astype(np.uint8)


def gamma_adjust(image_rgb: np.ndarray, gamma_range: Tuple[float, float] = (0.95, 1.05)) -> np.ndarray:
    gamma = max(np.random.uniform(gamma_range[0], gamma_range[1]), 1e-6)
    table = np.array([(i / 255.0) ** gamma * 255.0 for i in range(256)], dtype=np.float32)
    return cv2.LUT(image_rgb, np.clip(table, 0, 255).astype(np.uint8))


def channel_swap(image_rgb: np.ndarray) -> np.ndarray:
    return image_rgb[..., ::-1].copy()


def perspective_warp(image_rgb: np.ndarray, distortion_scale: float = 0.04, border_value: int = 255) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    if h < 2 or w < 2:
        return image_rgb.copy()

    dx = w * distortion_scale
    dy = h * distortion_scale
    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    dst = src.copy()
    dst[0] += [np.random.uniform(-dx, dx), np.random.uniform(-dy, dy)]
    dst[1] += [np.random.uniform(-dx, dx), np.random.uniform(-dy, dy)]
    dst[2] += [np.random.uniform(-dx, dx), np.random.uniform(-dy, dy)]
    dst[3] += [np.random.uniform(-dx, dx), np.random.uniform(-dy, dy)]
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        image_rgb,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(border_value, border_value, border_value),
    )
