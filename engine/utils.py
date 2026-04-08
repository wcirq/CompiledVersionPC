import json
import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import cv2
    import numpy as np
    import torch
except Exception as exc:
    from .debug_utils import print_exception_details

    print_exception_details(exc, context="engine.utils import failed")
    raise

from .debug_utils import guarded


@guarded("engine.utils.list_images failed")
def list_images(image_dir: str) -> List[str]:
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"]
    image_paths = []
    root = Path(image_dir)
    for ext in image_extensions:
        image_paths.extend(root.glob(f"**/*{ext}"))
        image_paths.extend(root.glob(f"**/*{ext.upper()}"))
    return sorted(str(p) for p in image_paths)


@guarded("engine.utils.read_image_rgb failed")
def read_image_rgb(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


@guarded("engine.utils.ensure_dir failed")
def ensure_dir(path: Optional[str]):
    if path:
        os.makedirs(path, exist_ok=True)


@guarded("engine.utils.to_bgr failed")
def to_bgr(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


@guarded("engine.utils.generate_positions failed")
def generate_positions(length: int, crop_len: int, stride: int) -> List[int]:
    if length <= crop_len:
        return [0]
    positions = list(range(0, length - crop_len + 1, stride))
    last = length - crop_len
    if positions[-1] != last:
        positions.append(last)
    return positions


@guarded("engine.utils.normalize_fixed failed")
def normalize_fixed(arr: np.ndarray, vmin: float, vmax: float, eps: float = 1e-12) -> np.ndarray:
    arr = np.clip(arr, vmin, vmax)
    return ((arr - vmin) / max(vmax - vmin, eps)).astype(np.float32)


@guarded("engine.utils.resize_long_side failed")
def resize_long_side(image: np.ndarray, target_long_side: int) -> Tuple[np.ndarray, float]:
    if target_long_side is None or target_long_side <= 0:
        return image, 1.0
    h, w = image.shape[:2]
    long_side = max(h, w)
    if long_side <= target_long_side:
        return image, 1.0
    scale = target_long_side / float(long_side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


@guarded("engine.utils.select_roi_with_tk failed")
def select_roi_with_tk(
    image_bgr: np.ndarray,
    window_title: str = "Select ROI",
    max_w: int = 1600,
    max_h: int = 900,
) -> Optional[Tuple[int, int, int, int]]:
    import tkinter as tk
    from PIL import Image, ImageTk

    h0, w0 = image_bgr.shape[:2]
    scale = min(max_w / w0, max_h / h0, 1.0)
    disp_w = max(1, int(round(w0 * scale)))
    disp_h = max(1, int(round(h0 * scale)))

    disp_bgr = cv2.resize(image_bgr, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
    disp_rgb = cv2.cvtColor(disp_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(disp_rgb)

    result = {"roi": None}
    root = tk.Tk()
    root.title(window_title)

    canvas = tk.Canvas(root, width=disp_w, height=disp_h, cursor="cross")
    canvas.pack()

    tk_img = ImageTk.PhotoImage(pil_img)
    canvas.create_image(0, 0, anchor="nw", image=tk_img)

    start_x = None
    start_y = None
    rect_id = None

    info_var = tk.StringVar()
    info_var.set("鼠标拖拽框选 ROI，回车确认，C 清空，Esc 取消")
    info_label = tk.Label(root, textvariable=info_var, anchor="w")
    info_label.pack(fill="x")

    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    def on_mouse_down(event):
        nonlocal start_x, start_y, rect_id
        start_x = clamp(event.x, 0, disp_w - 1)
        start_y = clamp(event.y, 0, disp_h - 1)
        if rect_id is not None:
            canvas.delete(rect_id)
            rect_id = None
        rect_id = canvas.create_rectangle(start_x, start_y, start_x, start_y, outline="lime", width=2)

    def on_mouse_move(event):
        if start_x is None or start_y is None or rect_id is None:
            return
        cur_x = clamp(event.x, 0, disp_w - 1)
        cur_y = clamp(event.y, 0, disp_h - 1)
        canvas.coords(rect_id, start_x, start_y, cur_x, cur_y)

    def on_mouse_up(event):
        nonlocal start_x, start_y, rect_id
        if start_x is None or start_y is None or rect_id is None:
            return

        end_x = clamp(event.x, 0, disp_w - 1)
        end_y = clamp(event.y, 0, disp_h - 1)

        x1 = min(start_x, end_x)
        y1 = min(start_y, end_y)
        x2 = max(start_x, end_x)
        y2 = max(start_y, end_y)

        if (x2 - x1) < 5 or (y2 - y1) < 5:
            info_var.set("ROI 太小，请重新框选")
            canvas.delete(rect_id)
            rect_id = None
            start_x = None
            start_y = None
            result["roi"] = None
            return

        result["roi"] = (x1, y1, x2, y2)
        info_var.set(f"已选择 ROI: {(x1, y1, x2, y2)}，按回车确认")

    def clear_roi(event=None):
        nonlocal start_x, start_y, rect_id
        if rect_id is not None:
            canvas.delete(rect_id)
            rect_id = None
        start_x = None
        start_y = None
        result["roi"] = None
        info_var.set("已清空，重新拖拽框选 ROI")

    def confirm(event=None):
        if result["roi"] is None:
            info_var.set("还没有选择 ROI")
            return
        x1, y1, x2, y2 = result["roi"]
        ox1 = int(round(x1 / scale))
        oy1 = int(round(y1 / scale))
        ox2 = int(round(x2 / scale))
        oy2 = int(round(y2 / scale))
        ox1 = max(0, min(w0 - 1, ox1))
        oy1 = max(0, min(h0 - 1, oy1))
        ox2 = max(0, min(w0, ox2))
        oy2 = max(0, min(h0, oy2))
        result["roi"] = None if (ox2 <= ox1 or oy2 <= oy1) else (ox1, oy1, ox2, oy2)
        root.destroy()

    def cancel(event=None):
        result["roi"] = None
        root.destroy()

    canvas.bind("<ButtonPress-1>", on_mouse_down)
    canvas.bind("<B1-Motion>", on_mouse_move)
    canvas.bind("<ButtonRelease-1>", on_mouse_up)
    root.bind("<Return>", confirm)
    root.bind("<space>", confirm)
    root.bind("<Escape>", cancel)
    root.bind("<Key-c>", clear_roi)
    root.bind("<Key-C>", clear_roi)
    root.mainloop()
    return result["roi"]


@guarded("engine.utils.round_to_multiple failed")
def round_to_multiple(x: int, multiple: int) -> int:
    if multiple <= 1:
        return int(x)
    return max(multiple, int(round(x / multiple) * multiple))


@guarded("engine.utils.sample_random_crop_size failed")
def sample_random_crop_size(
    base_crop_size: Tuple[int, int],
    scale_range: Tuple[float, float] = (1.0, 1.0),
    round_multiple: int = 32,
    min_crop_size: int = 64,
) -> Tuple[int, int]:
    base_h, base_w = base_crop_size
    min_scale, max_scale = scale_range
    if min_scale <= 0 or max_scale <= 0:
        raise ValueError("scale_range must be positive.")
    if min_scale > max_scale:
        raise ValueError("scale_range[0] must <= scale_range[1].")

    scale_h = np.random.uniform(min_scale, max_scale)
    scale_w = np.random.uniform(min_scale, max_scale)
    if base_h == base_w:
        scale_h = scale_w

    crop_h = round_to_multiple(max(min_crop_size, int(round(base_h * scale_h))), round_multiple)
    crop_w = round_to_multiple(max(min_crop_size, int(round(base_w * scale_w))), round_multiple)
    return max(min_crop_size, crop_h), max(min_crop_size, crop_w)


@guarded("engine.utils.scale_stride_with_crop failed")
def scale_stride_with_crop(
    base_crop_size: Tuple[int, int],
    new_crop_size: Tuple[int, int],
    base_stride: Tuple[int, int],
    round_multiple: int = 1,
) -> Tuple[int, int]:
    base_h, base_w = base_crop_size
    new_h, new_w = new_crop_size
    stride_h, stride_w = base_stride
    sh = max(1, int(round(stride_h * new_h / max(base_h, 1))))
    sw = max(1, int(round(stride_w * new_w / max(base_w, 1))))
    if round_multiple > 1:
        sh = round_to_multiple(sh, round_multiple)
        sw = round_to_multiple(sw, round_multiple)
    return max(1, sh), max(1, sw)


@guarded("engine.utils.save_embeddings_stream_init failed")
def save_embeddings_stream_init(save_dir: str, embed_dim: int):
    os.makedirs(save_dir, exist_ok=True)
    meta = {"embed_dim": int(embed_dim), "total_embeddings": 0, "num_chunks": 0}
    with open(os.path.join(save_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


@guarded("engine.utils.save_embeddings_stream_append failed")
def save_embeddings_stream_append(save_dir: str, embeddings: torch.Tensor):
    meta_path = os.path.join(save_dir, "meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    chunk_id = int(meta["num_chunks"])
    np.save(os.path.join(save_dir, f"chunk_{chunk_id:06d}.npy"), embeddings.detach().cpu().numpy().astype(np.float32))
    meta["num_chunks"] += 1
    meta["total_embeddings"] += int(embeddings.shape[0])
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


@guarded("engine.utils.load_random_embeddings_from_chunks failed")
def load_random_embeddings_from_chunks(save_dir: str, max_embeddings: int, seed: int = 42) -> torch.Tensor:
    with open(os.path.join(save_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    total = int(meta["total_embeddings"])
    embed_dim = int(meta["embed_dim"])
    num_chunks = int(meta["num_chunks"])
    if total <= 0:
        raise ValueError("No embeddings found on disk.")

    if max_embeddings is None or max_embeddings <= 0 or max_embeddings >= total:
        keep_global = np.arange(total, dtype=np.int64)
    else:
        rng = np.random.default_rng(seed)
        keep_global = np.sort(rng.choice(total, size=max_embeddings, replace=False).astype(np.int64))

    chunk_ranges = []
    start = 0
    for i in range(num_chunks):
        chunk_path = os.path.join(save_dir, f"chunk_{i:06d}.npy")
        arr = np.load(chunk_path, mmap_mode="r")
        if arr.ndim != 2 or int(arr.shape[1]) != embed_dim:
            raise ValueError(f"Invalid chunk shape in {chunk_path}: {arr.shape}")
        end = start + int(arr.shape[0])
        chunk_ranges.append((chunk_path, start, end))
        start = end

    result_parts = []
    ptr = 0
    n_keep = len(keep_global)
    for chunk_path, start, end in chunk_ranges:
        local_indices = []
        while ptr < n_keep and keep_global[ptr] < end:
            local_indices.append(int(keep_global[ptr] - start))
            ptr += 1
        if local_indices:
            arr = np.load(chunk_path, mmap_mode="r")
            result_parts.append(torch.from_numpy(np.asarray(arr[local_indices], dtype=np.float32)))

    if not result_parts:
        raise ValueError("No embeddings sampled from disk.")
    sampled = torch.cat(result_parts, dim=0).float()
    print(f"Randomly loaded embeddings from disk: {total} -> {sampled.shape[0]}")
    return sampled


@guarded("engine.utils.cleanup_dir failed")
def cleanup_dir(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)


@guarded("engine.utils.parse_tuple2 failed")
def parse_tuple2(values, name: str) -> Tuple[int, int]:
    if values is None or len(values) != 2:
        raise ValueError(f"{name} must contain exactly 2 integers.")
    return int(values[0]), int(values[1])


@guarded("engine.utils.parse_float_tuple2 failed")
def parse_float_tuple2(values, name: str) -> Tuple[float, float]:
    if values is None or len(values) != 2:
        raise ValueError(f"{name} must contain exactly 2 numbers.")
    return float(values[0]), float(values[1])
