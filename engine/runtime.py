import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .augment import (
    adjust_brightness,
    adjust_contrast,
    adjust_saturation,
    channel_swap,
    color_shift,
    gamma_adjust,
    gaussian_blur,
    gaussian_noise,
    perspective_warp,
    rotate_image,
    vertical_flip,
)
from .backbone import FeatureBackbone
from .indexing import MemoryIndex
from .utils import (
    cleanup_dir,
    ensure_dir,
    generate_positions,
    list_images,
    load_random_embeddings_from_chunks,
    normalize_fixed,
    read_image_rgb,
    resize_long_side,
    sample_random_crop_size,
    save_embeddings_stream_append,
    save_embeddings_stream_init,
    scale_stride_with_crop,
    to_bgr,
)


class VisionMemoryEngine:
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        backbone: str = "resnet50",
        input_size: Tuple[int, int] = (240, 240),
        memory_ratio: float = 0.1,
        target_embed_dimension: int = 1024,
        local_kernel: int = 3,
        knn_neighbors: int = 1,
        knn_backend: str = "auto",
        knn_query_chunk_size: int = 8192,
        use_amp: bool = False,
        enable_train_augment: bool = True,
        aug_keep_original_count: int = 1,
        aug_vflip_count: int = 0,
        aug_rotate_count: int = 0,
        aug_brightness_count: int = 0,
        aug_contrast_count: int = 0,
        aug_saturation_count: int = 0,
        aug_blur_count: int = 0,
        aug_noise_count: int = 0,
        aug_color_shift_count: int = 0,
        aug_gamma_count: int = 0,
        aug_channel_swap_count: int = 0,
        aug_perspective_count: int = 0,
        aug_rotate_range: Tuple[float, float] = (-10.0, 10.0),
        aug_brightness_limit: float = 0.08,
        aug_contrast_limit: float = 0.08,
        aug_saturation_limit: float = 0.08,
        aug_blur_sigma_min: float = 0.1,
        aug_blur_sigma_max: float = 1.3,
        aug_noise_sigma_min: float = 2.0,
        aug_noise_sigma_max: float = 8.0,
        aug_color_shift_limit: int = 8,
        aug_gamma_range: Tuple[float, float] = (0.95, 1.05),
        aug_perspective_distortion: float = 0.04,
    ):
        if backbone != "resnet50":
            raise ValueError("Currently only resnet50 is supported.")

        self.device = device
        self.backbone_name = backbone
        self.input_size = tuple(input_size)
        self.memory_ratio = float(memory_ratio)
        self.target_embed_dimension = int(target_embed_dimension)
        self.local_kernel = int(local_kernel)
        self.knn_neighbors = int(knn_neighbors)
        self.knn_backend = knn_backend
        self.knn_query_chunk_size = int(knn_query_chunk_size)
        self.use_amp = bool(use_amp and torch.cuda.is_available() and "cuda" in str(device))

        self.enable_train_augment = bool(enable_train_augment)
        self.aug_keep_original_count = int(aug_keep_original_count)
        self.aug_vflip_count = int(aug_vflip_count)
        self.aug_rotate_count = int(aug_rotate_count)
        self.aug_brightness_count = int(aug_brightness_count)
        self.aug_contrast_count = int(aug_contrast_count)
        self.aug_saturation_count = int(aug_saturation_count)
        self.aug_blur_count = int(aug_blur_count)
        self.aug_noise_count = int(aug_noise_count)
        self.aug_color_shift_count = int(aug_color_shift_count)
        self.aug_gamma_count = int(aug_gamma_count)
        self.aug_channel_swap_count = int(aug_channel_swap_count)
        self.aug_perspective_count = int(aug_perspective_count)

        self.aug_rotate_range = tuple(aug_rotate_range)
        self.aug_brightness_limit = float(aug_brightness_limit)
        self.aug_contrast_limit = float(aug_contrast_limit)
        self.aug_saturation_limit = float(aug_saturation_limit)
        self.aug_blur_sigma_min = float(aug_blur_sigma_min)
        self.aug_blur_sigma_max = float(aug_blur_sigma_max)
        self.aug_noise_sigma_min = float(aug_noise_sigma_min)
        self.aug_noise_sigma_max = float(aug_noise_sigma_max)
        self.aug_color_shift_limit = int(aug_color_shift_limit)
        self.aug_gamma_range = tuple(aug_gamma_range)
        self.aug_perspective_distortion = float(aug_perspective_distortion)

        self.feature_extractor = FeatureBackbone().to(self.device).eval()
        self.raw_embed_dim = 512 + 1024
        self.project_matrix = self._init_projector()

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        self.memory_bank: Optional[torch.Tensor] = None
        self.memory_index: Optional[MemoryIndex] = None
        self.train_image_paths: List[str] = []

        self.score_mean: Optional[float] = None
        self.score_std: Optional[float] = None
        self.recommended_threshold: Optional[float] = None
        self.heatmap_mean: Optional[float] = None
        self.heatmap_std: Optional[float] = None
        self.heatmap_vis_min: Optional[float] = None
        self.heatmap_vis_max: Optional[float] = None

    def _init_projector(self):
        if self.target_embed_dimension <= 0:
            return None
        gen = torch.Generator(device="cpu")
        gen.manual_seed(42)
        mat = torch.randn(self.raw_embed_dim, self.target_embed_dimension, generator=gen, dtype=torch.float32)
        return F.normalize(mat, dim=0)

    def _images_to_tensor_batch(self, images_rgb: List[np.ndarray]) -> torch.Tensor:
        h, w = self.input_size
        arrs = []
        for img in images_rgb:
            resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            x = resized.astype(np.float32) / 255.0
            x = (x - self.mean) / self.std
            arrs.append(np.transpose(x, (2, 0, 1)))
        return torch.from_numpy(np.stack(arrs, axis=0)).float()

    def _extract_sliding_crops(
        self,
        image_rgb: np.ndarray,
        crop_size: Tuple[int, int],
        stride: Tuple[int, int],
        pad_value: int = 255,
    ):
        crop_h, crop_w = crop_size
        stride_h, stride_w = stride
        orig_h, orig_w = image_rgb.shape[:2]

        padded = cv2.copyMakeBorder(
            image_rgb,
            top=0,
            bottom=max(0, crop_h - orig_h),
            left=0,
            right=max(0, crop_w - orig_w),
            borderType=cv2.BORDER_CONSTANT,
            value=[pad_value, pad_value, pad_value],
        )
        h, w = padded.shape[:2]
        ys = generate_positions(h, crop_h, stride_h)
        xs = generate_positions(w, crop_w, stride_w)

        crops = []
        boxes = []
        for y in ys:
            for x in xs:
                crops.append(padded[y:y + crop_h, x:x + crop_w])
                boxes.append((y, y + crop_h, x, x + crop_w))
        return crops, boxes, (orig_h, orig_w), padded

    def _merge_features(self, feat2: torch.Tensor, feat3: torch.Tensor):
        feat3_h = int(feat3.shape[2])
        feat3_w = int(feat3.shape[3])
        feat2_aligned = F.adaptive_avg_pool2d(feat2, output_size=(feat3_h, feat3_w))
        feat = torch.cat([feat2_aligned, feat3], dim=1)
        if self.local_kernel > 1:
            pad = self.local_kernel // 2
            feat = F.avg_pool2d(feat, kernel_size=self.local_kernel, stride=1, padding=pad)

        b, _, h, w = feat.shape
        patches = feat.permute(0, 2, 3, 1).reshape(b, h * w, -1)
        if self.project_matrix is not None:
            patches = patches @ self.project_matrix.to(patches.device)
        return patches, (h, w)

    @torch.no_grad()
    def _extract_embeddings_batch(self, images: torch.Tensor):
        images = images.to(self.device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=self.use_amp):
            feat2, feat3 = self.feature_extractor(images)
            return self._merge_features(feat2, feat3)

    def _compress_memory(self, features: torch.Tensor, sampling_ratio: float, start_points: int = 10) -> torch.Tensor:
        n_total = features.shape[0]
        n_select = max(1, int(n_total * sampling_ratio))
        if n_select >= n_total:
            return features

        device = torch.device(self.device if torch.cuda.is_available() and "cuda" in self.device else "cpu")
        feats = features.float().to(device)
        rng = np.random.default_rng(42)
        start_indices = torch.tensor(
            rng.choice(n_total, size=min(start_points, n_total), replace=False),
            dtype=torch.long,
            device=device,
        )
        min_dist = torch.cdist(feats, feats[start_indices], p=2).min(dim=1).values
        selected = [int(start_indices[0].item())]
        selected_mask = torch.zeros(n_total, dtype=torch.bool, device=device)
        selected_mask[selected[0]] = True

        pbar = tqdm(total=n_select - 1, desc="Memory compression")
        for _ in range(n_select - 1):
            next_idx = int(torch.argmax(min_dist).item())
            selected.append(next_idx)
            selected_mask[next_idx] = True
            new_dist = torch.cdist(feats, feats[next_idx:next_idx + 1], p=2).squeeze(1)
            min_dist = torch.minimum(min_dist, new_dist)
            min_dist[selected_mask] = -1.0
            pbar.update(1)
        pbar.close()
        return feats[torch.tensor(selected, dtype=torch.long, device=device)].cpu()

    def _build_index(self):
        if self.memory_bank is None or len(self.memory_bank) == 0:
            raise ValueError("Memory bank is empty.")
        self.memory_index = MemoryIndex(
            backend=self.knn_backend,
            n_neighbors=self.knn_neighbors,
            device=self.device,
            query_chunk_size=self.knn_query_chunk_size,
        )
        self.memory_index.fit(self.memory_bank)

    def _generate_augmented_crops(self, crop_rgb: np.ndarray) -> List[np.ndarray]:
        outputs = [crop_rgb.copy() for _ in range(max(0, self.aug_keep_original_count if self.enable_train_augment else 1))]
        if not self.enable_train_augment:
            return outputs
        outputs.extend(vertical_flip(crop_rgb) for _ in range(max(0, self.aug_vflip_count)))
        outputs.extend(rotate_image(crop_rgb, self.aug_rotate_range, border_value=255) for _ in range(max(0, self.aug_rotate_count)))
        outputs.extend(adjust_brightness(crop_rgb, self.aug_brightness_limit) for _ in range(max(0, self.aug_brightness_count)))
        outputs.extend(adjust_contrast(crop_rgb, self.aug_contrast_limit) for _ in range(max(0, self.aug_contrast_count)))
        outputs.extend(adjust_saturation(crop_rgb, self.aug_saturation_limit) for _ in range(max(0, self.aug_saturation_count)))
        outputs.extend(gaussian_blur(crop_rgb, (3, 5), (self.aug_blur_sigma_min, self.aug_blur_sigma_max)) for _ in range(max(0, self.aug_blur_count)))
        outputs.extend(gaussian_noise(crop_rgb, (self.aug_noise_sigma_min, self.aug_noise_sigma_max)) for _ in range(max(0, self.aug_noise_count)))
        outputs.extend(color_shift(crop_rgb, self.aug_color_shift_limit) for _ in range(max(0, self.aug_color_shift_count)))
        outputs.extend(gamma_adjust(crop_rgb, self.aug_gamma_range) for _ in range(max(0, self.aug_gamma_count)))
        outputs.extend(channel_swap(crop_rgb) for _ in range(max(0, self.aug_channel_swap_count)))
        outputs.extend(perspective_warp(crop_rgb, self.aug_perspective_distortion, border_value=255) for _ in range(max(0, self.aug_perspective_count)))
        return outputs

    def build_memory_bank(
        self,
        image_dir: str,
        crop_size: Tuple[int, int] = (160, 160),
        stride: Optional[Tuple[int, int]] = None,
        batch_size: int = 32,
        max_embeddings: int = 0,
        train_crop_scale_range: Tuple[float, float] = (1.0, 1.0),
        train_crop_round_multiple: int = 32,
        train_min_crop_size: int = 64,
        random_seed: int = 42,
        stream_to_disk: bool = True,
        stream_dir: str = "./embedding_cache",
        cleanup_stream_dir: bool = True,
        infer_long_side: int = 0,
    ) -> int:
        image_paths = list_images(image_dir)
        if not image_paths:
            raise ValueError(f"No images found in {image_dir}")

        np.random.seed(random_seed)
        self.train_image_paths = image_paths
        all_embeddings = []
        batch_crops = []
        total_embeddings_written = 0

        if stream_to_disk:
            if cleanup_stream_dir:
                cleanup_dir(stream_dir)
            save_embeddings_stream_init(stream_dir, embed_dim=self.target_embed_dimension)

        def flush_batch():
            nonlocal batch_crops, total_embeddings_written
            if not batch_crops:
                return
            batch = self._images_to_tensor_batch(batch_crops)
            embs, _ = self._extract_embeddings_batch(batch)
            embed_dim = int(embs.shape[2])
            embs = embs.reshape(-1, embed_dim).cpu().float()
            total_embeddings_written += int(embs.shape[0])
            if stream_to_disk:
                save_embeddings_stream_append(stream_dir, embs)
            else:
                all_embeddings.append(embs)
            batch_crops = []

        for img_path in tqdm(image_paths, desc="Building memory"):
            try:
                image_rgb = read_image_rgb(img_path)
                image_rgb, _ = resize_long_side(image_rgb, infer_long_side)
                cur_crop_size = sample_random_crop_size(
                    base_crop_size=crop_size,
                    scale_range=train_crop_scale_range,
                    round_multiple=train_crop_round_multiple,
                    min_crop_size=train_min_crop_size,
                )
                cur_stride = (
                    (max(1, cur_crop_size[0] // 2), max(1, cur_crop_size[1] // 2))
                    if stride is None
                    else scale_stride_with_crop(crop_size, cur_crop_size, stride, round_multiple=1)
                )
                crops, _, _, _ = self._extract_sliding_crops(image_rgb=image_rgb, crop_size=cur_crop_size, stride=cur_stride)
                for crop in crops:
                    batch_crops.extend(self._generate_augmented_crops(crop))
                    if len(batch_crops) >= batch_size:
                        flush_batch()
            except Exception as exc:
                print(f"[WARN] Failed processing {img_path}: {exc}")
        flush_batch()

        if stream_to_disk:
            if total_embeddings_written <= 0:
                raise ValueError("No embeddings extracted from training images.")
            embeddings_tensor = load_random_embeddings_from_chunks(stream_dir, max_embeddings=max_embeddings, seed=random_seed)
        else:
            if not all_embeddings:
                raise ValueError("No embeddings extracted from training images.")
            embeddings_tensor = torch.cat(all_embeddings, dim=0).float()
            if max_embeddings and embeddings_tensor.shape[0] > max_embeddings:
                rng = np.random.default_rng(random_seed)
                idx = np.sort(rng.choice(embeddings_tensor.shape[0], size=max_embeddings, replace=False))
                embeddings_tensor = embeddings_tensor[idx]

        self.memory_bank = (
            self._compress_memory(embeddings_tensor, sampling_ratio=self.memory_ratio)
            if self.memory_ratio < 1.0
            else embeddings_tensor
        )
        self._build_index()
        return int(self.memory_bank.shape[0])

    @torch.no_grad()
    def _compute_score_map(
        self,
        image_rgb: np.ndarray,
        crop_size: Tuple[int, int] = (160, 160),
        stride: Optional[Tuple[int, int]] = None,
        detect_batch_size: int = 8,
        infer_long_side: int = 0,
    ):
        if self.memory_bank is None or len(self.memory_bank) == 0:
            raise ValueError("Memory bank is empty.")
        if self.memory_index is None:
            self._build_index()

        orig_h0, orig_w0 = image_rgb.shape[:2]
        work_image, infer_scale = resize_long_side(image_rgb, infer_long_side)
        if stride is None:
            stride = (crop_size[0] // 2, crop_size[1] // 2)

        crops, boxes, (orig_h, orig_w), padded = self._extract_sliding_crops(work_image, crop_size, stride)
        full_heatmap = np.zeros(padded.shape[:2], dtype=np.float32)
        count_map = np.zeros(padded.shape[:2], dtype=np.float32)
        global_score = -1e18

        for st in range(0, len(crops), max(1, int(detect_batch_size))):
            ed = min(st + max(1, int(detect_batch_size)), len(crops))
            batch = self._images_to_tensor_batch(crops[st:ed])
            embeddings, (feat_h, feat_w) = self._extract_embeddings_batch(batch)
            embed_dim = int(embeddings.shape[2])
            distances, _ = self.memory_index.kneighbors(embeddings.reshape(-1, embed_dim).contiguous())
            patch_scores = distances[:, 0].reshape(ed - st, feat_h, feat_w).astype(np.float32)
            global_score = max(global_score, float(patch_scores.max()))

            for i, (y1, y2, x1, x2) in enumerate(boxes[st:ed]):
                score_map = cv2.resize(patch_scores[i], (crop_size[1], crop_size[0]), interpolation=cv2.INTER_CUBIC)
                full_heatmap[y1:y2, x1:x2] += score_map
                count_map[y1:y2, x1:x2] += 1.0

        count_map[count_map == 0] = 1.0
        full_heatmap = (full_heatmap / count_map)[:orig_h, :orig_w]
        if infer_scale != 1.0:
            full_heatmap = cv2.resize(full_heatmap, (orig_w0, orig_h0), interpolation=cv2.INTER_CUBIC)
        return float(global_score), full_heatmap.astype(np.float32)

    def calibrate_threshold(
        self,
        image_dir: str,
        crop_size: Tuple[int, int] = (160, 160),
        stride: Optional[Tuple[int, int]] = None,
        quantile: float = 0.99,
        heatmap_std_scale: float = 3.0,
        heatmap_quantile: float = 0.999,
        max_heatmap_samples: int = 2_000_000,
        detect_batch_size: int = 8,
        infer_long_side: int = 0,
    ) -> float:
        image_paths = list_images(image_dir)
        if not image_paths:
            raise ValueError(f"No images found in {image_dir}")

        scores = []
        sampled_heat_values = []
        rng = np.random.default_rng(42)

        for img_path in tqdm(image_paths, desc="Calibrating threshold"):
            image_rgb = read_image_rgb(img_path)
            score, heatmap = self._compute_score_map(image_rgb, crop_size, stride, detect_batch_size, infer_long_side)
            scores.append(score)
            flat = heatmap.reshape(-1).astype(np.float32)
            if flat.size > 0:
                sampled_heat_values.append(flat)

        scores = np.array(scores, dtype=np.float32)
        self.score_mean = float(scores.mean())
        self.score_std = float(scores.std())
        self.recommended_threshold = float(np.quantile(scores, quantile))

        if sampled_heat_values:
            heat_values = np.concatenate(sampled_heat_values, axis=0)
            if heat_values.size > max_heatmap_samples:
                heat_values = heat_values[rng.choice(heat_values.size, size=max_heatmap_samples, replace=False)]
            self.heatmap_mean = float(heat_values.mean())
            self.heatmap_std = float(heat_values.std())
            qv = float(np.quantile(heat_values, heatmap_quantile))
            sv = float(self.heatmap_mean + heatmap_std_scale * self.heatmap_std)
            self.heatmap_vis_min = 0.0
            self.heatmap_vis_max = max(qv, sv, max(float(self.recommended_threshold) * 0.5, 1e-6))
        else:
            self.heatmap_mean = None
            self.heatmap_std = None
            self.heatmap_vis_min = 0.0
            self.heatmap_vis_max = max(float(self.recommended_threshold or 1.0), 1e-6)

        return self.recommended_threshold

    def detect_image(
        self,
        image_rgb: np.ndarray,
        crop_size: Tuple[int, int] = (160, 160),
        stride: Optional[Tuple[int, int]] = None,
        threshold: float = 1.0,
        save_heatmap: Optional[str] = None,
        overlay_alpha: float = 0.45,
        original_image_for_overlay: Optional[np.ndarray] = None,
        roi_xyxy: Optional[Tuple[int, int, int, int]] = None,
        detect_batch_size: int = 8,
        infer_long_side: int = 0,
        heatmap_zero_below_threshold: bool = False,
    ):
        global_score, full_heatmap = self._compute_score_map(image_rgb, crop_size, stride, detect_batch_size, infer_long_side)
        is_anomaly = bool(global_score > threshold)

        if heatmap_zero_below_threshold:
            full_heatmap = full_heatmap.copy()
            full_heatmap[full_heatmap < threshold] = 0.0

        if save_heatmap:
            vis_min = float(self.heatmap_vis_min) if self.heatmap_vis_min is not None else 0.0
            vis_max = float(self.heatmap_vis_max) if self.heatmap_vis_max is not None else max(float(threshold), 1e-6)
            heat_u8 = (normalize_fixed(full_heatmap, vis_min, vis_max) * 255).astype(np.uint8)
            heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
            base_bgr = to_bgr(image_rgb)
            overlay = cv2.addWeighted(base_bgr, 1.0 - overlay_alpha, heat_color, overlay_alpha, 0)

            src_path = str(Path(save_heatmap).with_name(Path(save_heatmap).stem + "_src.jpg"))
            heatmap_only_path = str(Path(save_heatmap).with_name(Path(save_heatmap).stem + "_heatmap.jpg"))
            score_json_path = str(Path(save_heatmap).with_name(Path(save_heatmap).stem + "_score.json"))

            cv2.imwrite(src_path, base_bgr)
            cv2.imwrite(save_heatmap, overlay)
            cv2.imwrite(heatmap_only_path, heat_color)
            with open(score_json_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "global_score": global_score,
                        "is_anomaly": is_anomaly,
                        "threshold": threshold,
                        "roi_xyxy": roi_xyxy,
                        "heatmap_vis_min": vis_min,
                        "heatmap_vis_max": vis_max,
                        "heatmap_zero_below_threshold": heatmap_zero_below_threshold,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            if original_image_for_overlay is not None and roi_xyxy is not None:
                x1, y1, x2, y2 = roi_xyxy
                full_bgr = original_image_for_overlay.copy()
                full_bgr[y1:y2, x1:x2] = overlay
                color = (0, 255, 255) if is_anomaly else (0, 255, 0)
                cv2.rectangle(full_bgr, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    full_bgr,
                    f"score={global_score:.4f} {'ANOMALY' if is_anomaly else 'NORMAL'}",
                    (x1, max(25, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                    cv2.LINE_AA,
                )
                cv2.imwrite(str(Path(save_heatmap).with_name(Path(save_heatmap).stem + "_full.jpg")), full_bgr)

        return is_anomaly, global_score, full_heatmap

    def detect(self, image_path: str, **kwargs):
        image_rgb = read_image_rgb(image_path)
        return self.detect_image(image_rgb=image_rgb, **kwargs)

    def detect_batch(
        self,
        image_dir: str,
        crop_size: Tuple[int, int] = (160, 160),
        stride: Optional[Tuple[int, int]] = None,
        threshold: float = 1.0,
        output_dir: Optional[str] = None,
        detect_batch_size: int = 8,
        infer_long_side: int = 0,
        heatmap_zero_below_threshold: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        image_paths = list_images(image_dir)
        if not image_paths:
            raise ValueError(f"No images found in {image_dir}")
        ensure_dir(output_dir)
        results = {}
        for img_path in tqdm(image_paths, desc="Batch detection"):
            try:
                heatmap_path = os.path.join(output_dir, f"{Path(img_path).stem}_overlay.jpg") if output_dir else None
                is_anomaly, score, _ = self.detect(
                    image_path=img_path,
                    crop_size=crop_size,
                    stride=stride,
                    threshold=threshold,
                    save_heatmap=heatmap_path,
                    detect_batch_size=detect_batch_size,
                    infer_long_side=infer_long_side,
                    heatmap_zero_below_threshold=heatmap_zero_below_threshold,
                )
                results[img_path] = {"is_anomaly": bool(is_anomaly), "score": float(score)}
            except Exception as exc:
                results[img_path] = {"is_anomaly": None, "score": None, "error": str(exc)}
        if output_dir:
            with open(os.path.join(output_dir, "detection_results.json"), "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        return results

    def save(self, save_path: str):
        torch.save(
            {
                "config": {
                    "device": self.device,
                    "backbone_name": self.backbone_name,
                    "input_size": self.input_size,
                    "memory_ratio": self.memory_ratio,
                    "target_embed_dimension": self.target_embed_dimension,
                    "local_kernel": self.local_kernel,
                    "knn_neighbors": self.knn_neighbors,
                    "knn_backend": self.knn_backend,
                    "knn_query_chunk_size": self.knn_query_chunk_size,
                    "use_amp": self.use_amp,
                    "enable_train_augment": self.enable_train_augment,
                    "aug_keep_original_count": self.aug_keep_original_count,
                    "aug_vflip_count": self.aug_vflip_count,
                    "aug_rotate_count": self.aug_rotate_count,
                    "aug_brightness_count": self.aug_brightness_count,
                    "aug_contrast_count": self.aug_contrast_count,
                    "aug_saturation_count": self.aug_saturation_count,
                    "aug_blur_count": self.aug_blur_count,
                    "aug_noise_count": self.aug_noise_count,
                    "aug_color_shift_count": self.aug_color_shift_count,
                    "aug_gamma_count": self.aug_gamma_count,
                    "aug_channel_swap_count": self.aug_channel_swap_count,
                    "aug_perspective_count": self.aug_perspective_count,
                    "aug_rotate_range": self.aug_rotate_range,
                    "aug_brightness_limit": self.aug_brightness_limit,
                    "aug_contrast_limit": self.aug_contrast_limit,
                    "aug_saturation_limit": self.aug_saturation_limit,
                    "aug_blur_sigma_min": self.aug_blur_sigma_min,
                    "aug_blur_sigma_max": self.aug_blur_sigma_max,
                    "aug_noise_sigma_min": self.aug_noise_sigma_min,
                    "aug_noise_sigma_max": self.aug_noise_sigma_max,
                    "aug_color_shift_limit": self.aug_color_shift_limit,
                    "aug_gamma_range": self.aug_gamma_range,
                    "aug_perspective_distortion": self.aug_perspective_distortion,
                },
                "memory_bank": self.memory_bank,
                "train_image_paths": self.train_image_paths,
                "project_matrix": self.project_matrix,
                "score_mean": self.score_mean,
                "score_std": self.score_std,
                "recommended_threshold": self.recommended_threshold,
                "heatmap_mean": self.heatmap_mean,
                "heatmap_std": self.heatmap_std,
                "heatmap_vis_min": self.heatmap_vis_min,
                "heatmap_vis_max": self.heatmap_vis_max,
            },
            save_path,
        )
        print(f"Model saved to: {save_path}")

    def _sync_runtime_state(self):
        self.use_amp = bool(self.use_amp and torch.cuda.is_available() and "cuda" in str(self.device))
        self.feature_extractor = self.feature_extractor.to(self.device).float().eval()
        if self.project_matrix is not None:
            self.project_matrix = self.project_matrix.float()
        if self.memory_bank is not None:
            self.memory_bank = self.memory_bank.cpu().float()

    def load(self, load_path: str):
        try:
            data = torch.load(load_path, map_location="cpu", weights_only=False)
        except TypeError:
            data = torch.load(load_path, map_location="cpu")

        config = data.get("config", {})
        self.device = "cpu"
        self.backbone_name = config.get("backbone_name", self.backbone_name)
        self.input_size = tuple(config.get("input_size", self.input_size))
        self.memory_ratio = config.get("memory_ratio", self.memory_ratio)
        self.target_embed_dimension = config.get("target_embed_dimension", self.target_embed_dimension)
        self.local_kernel = config.get("local_kernel", self.local_kernel)
        self.knn_neighbors = config.get("knn_neighbors", self.knn_neighbors)
        self.knn_backend = config.get("knn_backend", self.knn_backend)
        self.knn_query_chunk_size = config.get("knn_query_chunk_size", self.knn_query_chunk_size)
        self.use_amp = bool(config.get("use_amp", self.use_amp))

        self.enable_train_augment = bool(config.get("enable_train_augment", self.enable_train_augment))
        self.aug_keep_original_count = int(config.get("aug_keep_original_count", self.aug_keep_original_count))
        self.aug_vflip_count = int(config.get("aug_vflip_count", self.aug_vflip_count))
        self.aug_rotate_count = int(config.get("aug_rotate_count", self.aug_rotate_count))
        self.aug_brightness_count = int(config.get("aug_brightness_count", self.aug_brightness_count))
        self.aug_contrast_count = int(config.get("aug_contrast_count", self.aug_contrast_count))
        self.aug_saturation_count = int(config.get("aug_saturation_count", self.aug_saturation_count))
        self.aug_blur_count = int(config.get("aug_blur_count", self.aug_blur_count))
        self.aug_noise_count = int(config.get("aug_noise_count", self.aug_noise_count))
        self.aug_color_shift_count = int(config.get("aug_color_shift_count", self.aug_color_shift_count))
        self.aug_gamma_count = int(config.get("aug_gamma_count", self.aug_gamma_count))
        self.aug_channel_swap_count = int(config.get("aug_channel_swap_count", self.aug_channel_swap_count))
        self.aug_perspective_count = int(config.get("aug_perspective_count", self.aug_perspective_count))

        self.aug_rotate_range = tuple(config.get("aug_rotate_range", self.aug_rotate_range))
        self.aug_brightness_limit = float(config.get("aug_brightness_limit", self.aug_brightness_limit))
        self.aug_contrast_limit = float(config.get("aug_contrast_limit", self.aug_contrast_limit))
        self.aug_saturation_limit = float(config.get("aug_saturation_limit", self.aug_saturation_limit))
        self.aug_blur_sigma_min = float(config.get("aug_blur_sigma_min", self.aug_blur_sigma_min))
        self.aug_blur_sigma_max = float(config.get("aug_blur_sigma_max", self.aug_blur_sigma_max))
        self.aug_noise_sigma_min = float(config.get("aug_noise_sigma_min", self.aug_noise_sigma_min))
        self.aug_noise_sigma_max = float(config.get("aug_noise_sigma_max", self.aug_noise_sigma_max))
        self.aug_color_shift_limit = int(config.get("aug_color_shift_limit", self.aug_color_shift_limit))
        self.aug_gamma_range = tuple(config.get("aug_gamma_range", self.aug_gamma_range))
        self.aug_perspective_distortion = float(config.get("aug_perspective_distortion", self.aug_perspective_distortion))

        self.memory_bank = data["memory_bank"]
        self.train_image_paths = data.get("train_image_paths", [])
        self.project_matrix = data.get("project_matrix", self.project_matrix)
        self.score_mean = data.get("score_mean")
        self.score_std = data.get("score_std")
        self.recommended_threshold = data.get("recommended_threshold")
        self.heatmap_mean = data.get("heatmap_mean")
        self.heatmap_std = data.get("heatmap_std")
        self.heatmap_vis_min = data.get("heatmap_vis_min")
        self.heatmap_vis_max = data.get("heatmap_vis_max")

        self._sync_runtime_state()
        self._build_index()
        print(f"Model loaded from: {load_path}")
        print(f"Memory bank shape: {tuple(self.memory_bank.shape)}")

    def extract_image_embeddings(
        self,
        image_rgb: np.ndarray,
        crop_size: Tuple[int, int] = (160, 160),
        stride: Optional[Tuple[int, int]] = None,
        detect_batch_size: int = 8,
        infer_long_side: int = 0,
    ) -> torch.Tensor:
        work_image, _ = resize_long_side(image_rgb, infer_long_side)
        if stride is None:
            stride = (crop_size[0] // 2, crop_size[1] // 2)
        crops, _, _, _ = self._extract_sliding_crops(work_image, crop_size, stride)
        if not crops:
            raise ValueError("No valid crops extracted from image.")
        outputs = []
        for st in range(0, len(crops), max(1, int(detect_batch_size))):
            ed = min(st + max(1, int(detect_batch_size)), len(crops))
            batch = self._images_to_tensor_batch(crops[st:ed])
            embeddings, _ = self._extract_embeddings_batch(batch)
            embed_dim = int(embeddings.shape[2])
            outputs.append(embeddings.reshape(-1, embed_dim).cpu().float())
        return torch.cat(outputs, dim=0).float()

    def append_positive_embeddings(
        self,
        embeddings: torch.Tensor,
        max_append_embeddings: int = 0,
        random_seed: int = 42,
        recompress_ratio: float = 1.0,
    ) -> int:
        if embeddings is None or embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise ValueError("Invalid embeddings to append.")
        embeddings = embeddings.cpu().float()
        if max_append_embeddings and embeddings.shape[0] > max_append_embeddings:
            rng = np.random.default_rng(random_seed)
            idx = np.sort(rng.choice(embeddings.shape[0], size=max_append_embeddings, replace=False))
            embeddings = embeddings[idx]
        merged = embeddings if self.memory_bank is None or len(self.memory_bank) == 0 else torch.cat([self.memory_bank.cpu().float(), embeddings], dim=0)
        if recompress_ratio is not None and recompress_ratio < 1.0:
            merged = self._compress_memory(merged, sampling_ratio=recompress_ratio)
        self.memory_bank = merged.cpu().float()
        self._build_index()
        return int(self.memory_bank.shape[0])
