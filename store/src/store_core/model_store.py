from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from .engine import VisionMemoryEngine
from .engine.utils import resize_long_side
from .io_utils import maybe_load_image_bgr, read_image_bgr, write_image_bgr, image_to_base64
from .schemas import ModelInfo, ModelVersionInfo, RuntimeOptions, SampleRecord, new_id, utc_now
from .segmentation import TrainRoofSegmenter


ProgressCallback = Optional[Callable[[Dict[str, Any]], None]]
DEFAULT_APPEND_MAX_VECTORS = 20


class ModelStoreManager:
    def __init__(
        self,
        root_dir: str,
        yolo_weight_path: Optional[str] = None,
        yolo_conf_threshold: float = 0.25,
        yolo_device: Optional[str] = None,
    ):
        self.root_dir = Path(root_dir).resolve()
        self.models_dir = self.root_dir / "models"
        self.registry_path = self.root_dir / "registry.json"
        self.tmp_dir = self.root_dir / "tmp"
        self.yolo_weight_path = yolo_weight_path
        self.yolo_conf_threshold = float(yolo_conf_threshold)
        self.yolo_device = yolo_device
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.segmenter = TrainRoofSegmenter(
            weight_path=yolo_weight_path,
            conf_threshold=yolo_conf_threshold,
            device=yolo_device,
        )
        self._ensure_registry()

    def _ensure_registry(self) -> None:
        if not self.registry_path.exists():
            self._write_json(self.registry_path, {"models": []})

    @staticmethod
    def _write_json(path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _read_json(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))

    def _read_registry(self) -> Dict[str, Any]:
        return self._read_json(self.registry_path, {"models": []})

    def _save_registry(self, payload: Dict[str, Any]) -> None:
        self._write_json(self.registry_path, payload)

    def _model_dir(self, model_id: str) -> Path:
        return self.models_dir / model_id

    def _model_meta_path(self, model_id: str) -> Path:
        return self._model_dir(model_id) / "model.json"

    def _version_dir(self, model_id: str, version_id: str) -> Path:
        return self._model_dir(model_id) / "versions" / version_id

    def _version_meta_path(self, model_id: str, version_id: str) -> Path:
        return self._version_dir(model_id, version_id) / "version.json"

    def _samples_path(self, model_id: str, version_id: str) -> Path:
        return self._version_dir(model_id, version_id) / "samples.json"

    def _engine_path(self, model_id: str, version_id: str) -> Path:
        return self._version_dir(model_id, version_id) / "memory_model.pt"

    def _sample_tile_dir(self, model_id: str, version_id: str, sample_id: str) -> Path:
        return self._version_dir(model_id, version_id) / "tiles" / sample_id

    def _sample_tile_meta_path(self, model_id: str, version_id: str, sample_id: str) -> Path:
        return self._sample_tile_dir(model_id, version_id, sample_id) / "tiles.json"

    def _emit(self, callback: ProgressCallback, stage: str, **extra: Any) -> None:
        if callback is None:
            return
        payload = {"stage": stage, "timestamp": utc_now()}
        payload.update(extra)
        callback(payload)

    def _get_model_storage_status(self, model_meta: Dict[str, Any]) -> Dict[str, Any]:
        model_id = model_meta.get("model_id")
        version_id = model_meta.get("current_version_id")
        if not model_id or not version_id:
            return {
                "sample_assets_available": False,
                "raw_dir_exists": False,
                "processed_dir_exists": False,
                "tiles_dir_exists": False,
                "samples_file_exists": False,
                "message": "模型元数据不完整，无法判断样本文件状态。",
            }

        version_dir = self._version_dir(model_id, version_id)
        raw_dir = version_dir / "raw"
        processed_dir = version_dir / "processed"
        tiles_dir = version_dir / "tiles"
        samples_path = self._samples_path(model_id, version_id)
        sample_assets_available = raw_dir.exists() and processed_dir.exists() and tiles_dir.exists()
        return {
            "sample_assets_available": sample_assets_available,
            "raw_dir_exists": raw_dir.exists(),
            "processed_dir_exists": processed_dir.exists(),
            "tiles_dir_exists": tiles_dir.exists(),
            "samples_file_exists": samples_path.exists(),
            "message": (
                "样本文件完整，可查看和维护向量库样本。"
                if sample_assets_available
                else "当前模型缺少 raw / processed / tiles 等样本文件，只能用于部署检测，无法查看或维护向量库样本。"
            ),
        }

    def _with_model_storage_status(self, model_meta: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(model_meta)
        payload["storage_status"] = self._get_model_storage_status(model_meta)
        return payload

    def _get_sample_file_status(self, model_id: str, version_id: str, sample: Dict[str, Any]) -> Dict[str, Any]:
        raw_exists = Path(sample.get("raw_image_path", "")).exists()
        processed_exists = Path(sample.get("processed_image_path", "")).exists()
        tiles_meta = self._load_sample_tiles(model_id, version_id, sample["sample_id"])
        tile_image_paths = [Path(tile.get("image_path", "")) for tile in tiles_meta.get("tiles", []) if tile.get("image_path")]
        tile_images_complete = bool(tile_image_paths) and all(path.exists() for path in tile_image_paths)
        sample_assets_available = raw_exists and processed_exists and tile_images_complete
        return {
            "sample_assets_available": sample_assets_available,
            "raw_exists": raw_exists,
            "processed_exists": processed_exists,
            "tile_images_complete": tile_images_complete,
            "message": (
                "样本文件完整，可查看和编辑。"
                if sample_assets_available
                else "当前样本缺少原图、处理图或子图文件，无法查看和编辑。"
            ),
        }

    def list_models(self) -> List[Dict[str, Any]]:
        registry = self._read_registry()
        result = []
        for item in registry["models"]:
            model_meta = self._read_json(self._model_meta_path(item["model_id"]), {})
            if model_meta:
                result.append(self._with_model_storage_status(model_meta))
        return result

    def get_model(self, model_id: str) -> Dict[str, Any]:
        path = self._model_meta_path(model_id)
        if not path.exists():
            raise ValueError(f"Unknown model_id: {model_id}")
        return self._with_model_storage_status(self._read_json(path, {}))

    def export_model_archive(
        self,
        model_id: str,
        deployment_only: bool = False,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> str:
        model_meta = self.get_model(model_id)
        model_dir = Path(model_meta["model_dir"])
        if not model_dir.exists():
            raise ValueError(f"Model directory not found: {model_dir}")
        export_dir = self.tmp_dir / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        archive_base = export_dir / (f"{model_id}-deploy" if deployment_only else model_id)
        archive_path = f"{archive_base}.zip"
        if Path(archive_path).exists():
            Path(archive_path).unlink()
        if deployment_only:
            current_version_id = model_meta["current_version_id"]
            files = [
                self._model_meta_path(model_id),
                self._version_meta_path(model_id, current_version_id),
                self._engine_path(model_id, current_version_id),
            ]
            files = [path for path in files if path.exists() and path.is_file()]
            if not files:
                raise ValueError("No deployable model files found.")
        else:
            files = sorted([path for path in model_dir.rglob("*") if path.is_file()])
        total = max(1, len(files))
        with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for index, file_path in enumerate(files, start=1):
                arcname = str(Path(model_dir.name) / file_path.relative_to(model_dir))
                zf.write(file_path, arcname=arcname)
                if progress_callback is not None:
                    progress_callback(index, total, arcname)
        return archive_path

    def get_export_package_summary(self, model_id: str) -> Dict[str, Any]:
        model_meta = self.get_model(model_id)
        model_dir = Path(model_meta["model_dir"])
        if not model_dir.exists():
            raise ValueError(f"Model directory not found: {model_dir}")

        def summarize(paths: List[Path]) -> Dict[str, Any]:
            files = [path for path in paths if path.exists() and path.is_file()]
            total_size = sum(path.stat().st_size for path in files)
            return {
                "file_count": len(files),
                "total_size_bytes": int(total_size),
                "items": [str(path.relative_to(model_dir)) for path in files],
            }

        current_version_id = model_meta["current_version_id"]
        deploy_paths = [
            self._model_meta_path(model_id),
            self._version_meta_path(model_id, current_version_id),
            self._engine_path(model_id, current_version_id),
        ]
        full_paths = sorted([path for path in model_dir.rglob("*") if path.is_file()])
        return {
            "model_id": model_id,
            "full": summarize(full_paths),
            "deploy": summarize(deploy_paths),
        }

    def import_model_archive(self, archive_path: str) -> Dict[str, Any]:
        source_archive = Path(archive_path)
        if not source_archive.exists():
            raise ValueError(f"Archive not found: {archive_path}")

        registry = self._read_registry()
        existing_model_ids = {item["model_id"] for item in registry["models"]}
        existing_model_names = {item["model_name"] for item in registry["models"]}

        with tempfile.TemporaryDirectory(dir=str(self.tmp_dir)) as tmp_dir:
            extract_root = Path(tmp_dir) / "imported_model"
            extract_root.mkdir(parents=True, exist_ok=True)
            shutil.unpack_archive(str(source_archive), str(extract_root), format="zip")

            model_dirs = [item for item in extract_root.iterdir() if item.is_dir()]
            if len(model_dirs) != 1:
                raise ValueError("Model archive must contain exactly one top-level model directory.")

            imported_model_dir = model_dirs[0]
            model_meta_path = imported_model_dir / "model.json"
            if not model_meta_path.exists():
                raise ValueError("model.json not found in imported archive.")

            model_meta = self._read_json(model_meta_path, {})
            model_id = model_meta.get("model_id")
            model_name = model_meta.get("model_name")
            if not model_id or not model_name:
                raise ValueError("Imported model metadata missing model_id or model_name.")
            if model_id in existing_model_ids:
                raise ValueError(f"model_id already exists: {model_id}")
            if model_name in existing_model_names:
                raise ValueError(f"model_name already exists: {model_name}")

            target_dir = self._model_dir(model_id)
            if target_dir.exists():
                raise ValueError(f"Target model directory already exists: {target_dir}")

            shutil.copytree(str(imported_model_dir), str(target_dir))

        registry["models"].append({"model_id": model_id, "model_name": model_name})
        registry["models"].sort(key=lambda item: item["model_name"])
        self._save_registry(registry)
        return self.get_model(model_id)

    def _load_version_meta(self, model_id: str, version_id: str) -> Dict[str, Any]:
        path = self._version_meta_path(model_id, version_id)
        if not path.exists():
            raise ValueError(f"Unknown version: {model_id}/{version_id}")
        return self._read_json(path, {})

    def _load_samples(self, model_id: str, version_id: str) -> List[Dict[str, Any]]:
        path = self._samples_path(model_id, version_id)
        return self._read_json(path, {"samples": []}).get("samples", [])

    def _save_samples(self, model_id: str, version_id: str, samples: List[Dict[str, Any]]) -> None:
        self._write_json(self._samples_path(model_id, version_id), {"samples": samples})

    def _create_engine(self, options: RuntimeOptions) -> VisionMemoryEngine:
        return VisionMemoryEngine(**options.to_engine_kwargs())

    @staticmethod
    def _normalize_contours_payload(contour: Optional[Sequence[Any]]) -> List[List[List[int]]]:
        if contour is None:
            return []
        if not isinstance(contour, Sequence) or len(contour) == 0:
            return []
        first = contour[0]
        if isinstance(first, Sequence) and len(first) > 0 and isinstance(first[0], Sequence):
            return [[list(map(int, point)) for point in polygon] for polygon in contour]  # type: ignore[arg-type]
        return [[list(map(int, point)) for point in contour]]  # type: ignore[arg-type]

    @staticmethod
    def _get_preview_style(image: np.ndarray) -> Dict[str, Any]:
        height, width = image.shape[:2]
        base = max(height, width)
        line_thickness = max(2, int(round(base / 500)))
        box_thickness = max(2, int(round(base / 450)))
        font_scale = max(0.7, base / 1400.0)
        font_thickness = max(2, int(round(base / 700)))
        return {
            "line_thickness": line_thickness,
            "box_thickness": box_thickness,
            "font_scale": font_scale,
            "font_thickness": font_thickness,
        }

    def _build_processed_sample(
        self,
        source_image_name: str,
        source_image_path: str,
        image_bgr: np.ndarray,
        contour: Sequence[Sequence[int]],
        target_raw_path: Path,
        target_processed_path: Path,
        source_type: str,
        note: str = "",
    ) -> Dict[str, Any]:
        contour_np = np.asarray(contour, dtype=np.int32)
        if contour_np.ndim != 2 or contour_np.shape[0] < 3 or contour_np.shape[1] != 2:
            raise ValueError("Contour must be an Nx2 polygon with at least 3 points.")

        image_h, image_w = image_bgr.shape[:2]
        contour_np[:, 0] = np.clip(contour_np[:, 0], 0, image_w - 1)
        contour_np[:, 1] = np.clip(contour_np[:, 1], 0, image_h - 1)
        x, y, w, h = cv2.boundingRect(contour_np)
        if w <= 1 or h <= 1:
            raise ValueError("Contour bounding box is invalid.")

        mask = np.zeros((image_h, image_w), dtype=np.uint8)
        cv2.fillPoly(mask, [contour_np], 255)
        crop = image_bgr[y:y + h, x:x + w].copy()
        crop_mask = mask[y:y + h, x:x + w].copy()
        masked_crop = np.full_like(crop, 255)
        masked_crop[crop_mask > 0] = crop[crop_mask > 0]

        write_image_bgr(str(target_raw_path), image_bgr)
        write_image_bgr(str(target_processed_path), masked_crop)

        record = SampleRecord(
            sample_id=target_processed_path.stem,
            source_image_name=source_image_name,
            source_image_path=source_image_path,
            raw_image_path=str(target_raw_path),
            processed_image_path=str(target_processed_path),
            contour=contour_np.astype(int).tolist(),
            bbox=[int(x), int(y), int(x + w), int(y + h)],
            source_type=source_type,
            note=note,
        )
        return record.to_dict()

    def _preprocess_dir(
        self,
        image_dir: str,
        version_dir: Path,
        progress_callback: ProgressCallback = None,
        source_type: str = "train",
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        raw_dir = version_dir / "raw"
        processed_dir = version_dir / "processed"
        raw_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

        image_paths = sorted([p for p in Path(image_dir).glob("**/*") if p.is_file()])
        samples: List[Dict[str, Any]] = []
        failed_images: List[str] = []

        for idx, image_path in enumerate(image_paths, start=1):
            try:
                image_bgr = read_image_bgr(str(image_path))
                roofs = self.segmenter.segment_image(image_bgr)
                if not roofs:
                    failed_images.append(str(image_path))
                    self._emit(progress_callback, "segment_failed", image=str(image_path), index=idx, total=len(image_paths))
                    continue
                for roof_idx, roof in enumerate(roofs):
                    sample_id = new_id("sample")
                    raw_path = raw_dir / f"{sample_id}.png"
                    processed_path = processed_dir / f"{sample_id}.png"
                    samples.append(
                        self._build_processed_sample(
                            source_image_name=image_path.name,
                            source_image_path=str(image_path),
                            image_bgr=image_bgr,
                            contour=roof.contour,
                            target_raw_path=raw_path,
                            target_processed_path=processed_path,
                            source_type=source_type,
                            note=f"auto segmented roof #{roof_idx + 1}",
                        )
                    )
                self._emit(
                    progress_callback,
                    "segment_ok",
                    image=str(image_path),
                    index=idx,
                    total=len(image_paths),
                    roof_count=len(roofs),
                )
            except Exception as exc:
                failed_images.append(str(image_path))
                self._emit(progress_callback, "segment_failed", image=str(image_path), error=str(exc), index=idx, total=len(image_paths))
        return samples, failed_images

    def _extract_sample_tiles(
        self,
        engine: VisionMemoryEngine,
        model_id: str,
        version_id: str,
        sample: Dict[str, Any],
        options: RuntimeOptions,
    ) -> List[Dict[str, Any]]:
        processed_bgr = read_image_bgr(sample["processed_image_path"])
        processed_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
        work_image, _ = resize_long_side(processed_rgb, options.infer_long_side)
        stride = options.stride if options.stride is not None else (options.crop_size[0] // 2, options.crop_size[1] // 2)
        crops, boxes, (orig_h, orig_w), _ = engine._extract_sliding_crops(work_image, options.crop_size, stride)
        tile_dir = self._sample_tile_dir(model_id, version_id, sample["sample_id"])
        images_dir = tile_dir / "images"
        embeds_dir = tile_dir / "embeddings"
        images_dir.mkdir(parents=True, exist_ok=True)
        embeds_dir.mkdir(parents=True, exist_ok=True)

        if tile_dir.exists():
            for child in images_dir.glob("*"):
                if child.is_file():
                    child.unlink()
            for child in embeds_dir.glob("*"):
                if child.is_file():
                    child.unlink()

        tiles: List[Dict[str, Any]] = []
        gap = 18
        tile_col_size = max(1, options.crop_size[1])
        tile_row_size = max(1, options.crop_size[0])

        for st in range(0, len(crops), max(1, int(options.detect_batch_size))):
            ed = min(st + max(1, int(options.detect_batch_size)), len(crops))
            batch = engine._images_to_tensor_batch(crops[st:ed])
            embeddings, _ = engine._extract_embeddings_batch(batch)
            embed_dim = int(embeddings.shape[2])
            for local_idx, crop in enumerate(crops[st:ed]):
                global_idx = st + local_idx
                tile_id = f"tile_{global_idx:06d}"
                tile_image_path = images_dir / f"{tile_id}.png"
                tile_embed_path = embeds_dir / f"{tile_id}.pt"
                write_image_bgr(str(tile_image_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                tile_embeddings = embeddings[local_idx].reshape(-1, embed_dim).cpu().float()
                torch.save(tile_embeddings, tile_embed_path)
                y1, y2, x1, x2 = boxes[global_idx]
                display_x1 = int(x1 + gap * round(x1 / max(tile_col_size, 1)))
                display_y1 = int(y1 + gap * round(y1 / max(tile_row_size, 1)))
                display_x2 = display_x1 + int(x2 - x1)
                display_y2 = display_y1 + int(y2 - y1)
                tiles.append(
                    {
                        "tile_id": tile_id,
                        "box": [int(x1), int(y1), int(x2), int(y2)],
                        "sample_bbox": list(sample["bbox"]),
                        "display_box": [display_x1, display_y1, display_x2, display_y2],
                        "image_path": str(tile_image_path),
                        "embedding_path": str(tile_embed_path),
                        "enabled": True,
                    }
                )

        meta = {
            "sample_id": sample["sample_id"],
            "canvas_size": [
                max((tile["display_box"][2] for tile in tiles), default=orig_w) + gap,
                max((tile["display_box"][3] for tile in tiles), default=orig_h) + gap,
            ],
            "tiles": tiles,
        }
        self._write_json(self._sample_tile_meta_path(model_id, version_id, sample["sample_id"]), meta)
        sample["tile_count"] = len(tiles)
        sample["tile_meta_path"] = str(self._sample_tile_meta_path(model_id, version_id, sample["sample_id"]))
        return tiles

    def _load_sample_tiles(self, model_id: str, version_id: str, sample_id: str) -> Dict[str, Any]:
        return self._read_json(self._sample_tile_meta_path(model_id, version_id, sample_id), {"tiles": [], "canvas_size": [0, 0]})

    def _save_sample_tiles(self, model_id: str, version_id: str, sample_id: str, payload: Dict[str, Any]) -> None:
        self._write_json(self._sample_tile_meta_path(model_id, version_id, sample_id), payload)

    def _collect_enabled_embeddings_for_sample(self, model_id: str, version_id: str, sample_id: str) -> torch.Tensor:
        tiles_meta = self._load_sample_tiles(model_id, version_id, sample_id)
        chunks: List[torch.Tensor] = []
        for tile in tiles_meta.get("tiles", []):
            if not tile.get("enabled", True):
                continue
            emb_path = tile.get("embedding_path")
            if not emb_path:
                continue
            emb = torch.load(emb_path, map_location="cpu").cpu().float()
            if emb.ndim == 1:
                emb = emb.unsqueeze(0)
            chunks.append(emb)
        if not chunks:
            raise ValueError(f"No enabled tile embeddings found for sample_id: {sample_id}")
        return torch.cat(chunks, dim=0).float()

    def _collect_enabled_embeddings(self, model_id: str, version_id: str, samples: List[Dict[str, Any]]) -> torch.Tensor:
        chunks: List[torch.Tensor] = []
        for sample in samples:
            tiles_meta = self._load_sample_tiles(model_id, version_id, sample["sample_id"])
            for tile in tiles_meta.get("tiles", []):
                if not tile.get("enabled", True):
                    continue
                emb = torch.load(tile["embedding_path"], map_location="cpu")
                chunks.append(emb.cpu().float())
        if not chunks:
            raise ValueError("No enabled tile embeddings found.")
        return torch.cat(chunks, dim=0).float()

    def _rebuild_engine_from_tiles(
        self,
        model_id: str,
        version_id: str,
        options: RuntimeOptions,
        preserve_threshold: bool = True,
        calibrate: bool = False,
        compress_memory: bool = True,
    ) -> Dict[str, Any]:
        engine_path = self._engine_path(model_id, version_id)
        engine = self._create_engine(options)
        previous_threshold = None
        previous_stats: Dict[str, Any] = {}
        if engine_path.exists():
            engine.load(str(engine_path))
            previous_threshold = engine.recommended_threshold
            previous_stats = {
                "score_mean": engine.score_mean,
                "score_std": engine.score_std,
                "heatmap_mean": engine.heatmap_mean,
                "heatmap_std": engine.heatmap_std,
                "heatmap_vis_min": engine.heatmap_vis_min,
                "heatmap_vis_max": engine.heatmap_vis_max,
            }

        samples = self._load_samples(model_id, version_id)
        embeddings = self._collect_enabled_embeddings(model_id, version_id, samples)
        engine.memory_bank = engine._compress_memory(embeddings, sampling_ratio=options.memory_ratio) if compress_memory and options.memory_ratio < 1.0 else embeddings
        engine._build_index()

        if calibrate:
            threshold = engine.calibrate_threshold(
                image_dir=str(self._version_dir(model_id, version_id) / "processed"),
                crop_size=options.crop_size,
                stride=options.stride,
                quantile=options.threshold_quantile,
                heatmap_std_scale=options.heatmap_std_scale,
                heatmap_quantile=options.heatmap_quantile,
                max_heatmap_samples=options.max_heatmap_samples,
                detect_batch_size=options.detect_batch_size,
                infer_long_side=options.infer_long_side,
                fast_calibrate=options.fast_calibrate,
            )
        else:
            threshold = previous_threshold
            if preserve_threshold:
                engine.recommended_threshold = previous_threshold
                engine.score_mean = previous_stats.get("score_mean")
                engine.score_std = previous_stats.get("score_std")
                engine.heatmap_mean = previous_stats.get("heatmap_mean")
                engine.heatmap_std = previous_stats.get("heatmap_std")
                engine.heatmap_vis_min = previous_stats.get("heatmap_vis_min")
                engine.heatmap_vis_max = previous_stats.get("heatmap_vis_max")

        engine.save(str(engine_path))
        return {"engine": engine, "threshold": threshold}

    def _preprocess_calibrate_dir(
        self,
        image_dir: str,
        version_dir: Path,
        progress_callback: ProgressCallback = None,
    ) -> Tuple[Path, List[str]]:
        out_dir = version_dir / "calibrate_processed"
        out_dir.mkdir(parents=True, exist_ok=True)
        failed_images: List[str] = []

        image_paths = sorted([p for p in Path(image_dir).glob("**/*") if p.is_file()])
        for idx, image_path in enumerate(image_paths, start=1):
            try:
                image_bgr = read_image_bgr(str(image_path))
                roofs = self.segmenter.segment_image(image_bgr)
                if not roofs:
                    failed_images.append(str(image_path))
                    continue
                for roof in roofs:
                    sample_id = new_id("cal")
                    processed_path = out_dir / f"{sample_id}.png"
                    contour_np = np.asarray(roof.contour, dtype=np.int32)
                    x, y, w, h = cv2.boundingRect(contour_np)
                    mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask, [contour_np], 255)
                    crop = image_bgr[y:y + h, x:x + w].copy()
                    crop_mask = mask[y:y + h, x:x + w]
                    masked_crop = np.full_like(crop, 255)
                    masked_crop[crop_mask > 0] = crop[crop_mask > 0]
                    write_image_bgr(str(processed_path), masked_crop)
                self._emit(progress_callback, "calibrate_segment_ok", image=str(image_path), index=idx, total=len(image_paths))
            except Exception as exc:
                failed_images.append(str(image_path))
                self._emit(progress_callback, "calibrate_segment_failed", image=str(image_path), error=str(exc), index=idx, total=len(image_paths))
        return out_dir, failed_images

    def _save_version_meta(
        self,
        model_id: str,
        version_id: str,
        status: str,
        threshold: Optional[float],
        samples: List[Dict[str, Any]],
        failed_images: List[str],
        options: RuntimeOptions,
    ) -> Dict[str, Any]:
        version_meta = {
            "version_id": version_id,
            "version_dir": str(self._version_dir(model_id, version_id)),
            "created_at": utc_now(),
            "status": status,
            "threshold": threshold,
            "sample_count": len(samples),
            "failed_image_count": len(failed_images),
            "failed_images": failed_images,
            "runtime_options": asdict(options),
        }
        self._write_json(self._version_meta_path(model_id, version_id), version_meta)
        self._save_samples(model_id, version_id, samples)
        return version_meta

    def train_model(
        self,
        model_name: str,
        image_dir: str,
        runtime_options: Optional[RuntimeOptions] = None,
        save_root_dir: Optional[str] = None,
        calibrate_dir: Optional[str] = None,
        progress_callback: ProgressCallback = None,
    ) -> Dict[str, Any]:
        if save_root_dir and Path(save_root_dir).resolve() != self.root_dir:
            raise ValueError(
                "save_root_dir differs from manager root_dir. "
                "Please instantiate TrainRoofAnomalyStore with the target root_dir first."
            )
        runtime_options = runtime_options or RuntimeOptions()
        registry = self._read_registry()
        if any(item["model_name"] == model_name for item in registry["models"]):
            raise ValueError(f"model_name already exists: {model_name}")

        model_id = new_id("model")
        version_id = new_id("v")
        model_dir = self.root_dir / "models" / model_id
        version_dir = model_dir / "versions" / version_id
        version_dir.mkdir(parents=True, exist_ok=True)

        self._emit(progress_callback, "prepare_training", model_id=model_id, version_id=version_id)
        samples, failed_images = self._preprocess_dir(image_dir=image_dir, version_dir=version_dir, progress_callback=progress_callback)
        if not samples:
            raise ValueError("No valid train roof samples found after YOLO segmentation.")

        engine = self._create_engine(runtime_options)
        self._emit(progress_callback, "build_tiles_start", sample_count=len(samples))
        for sample in samples:
            self._extract_sample_tiles(engine, model_id, version_id, sample, runtime_options)
        self._save_samples(model_id, version_id, samples)
        rebuild = self._rebuild_engine_from_tiles(model_id, version_id, runtime_options, preserve_threshold=False, calibrate=False)
        engine = rebuild["engine"]
        self._emit(progress_callback, "build_memory_done", memory_bank_size=int(engine.memory_bank.shape[0]))

        calibrate_source_dir = str(version_dir / "processed")
        if calibrate_dir:
            calibrate_processed_dir, calibrate_failed = self._preprocess_calibrate_dir(calibrate_dir, version_dir, progress_callback)
            failed_images.extend(calibrate_failed)
            calibrate_source_dir = str(calibrate_processed_dir)

        self._emit(progress_callback, "calibrate_start", calibrate_dir=calibrate_source_dir)
        threshold = engine.calibrate_threshold(
            image_dir=calibrate_source_dir,
            crop_size=runtime_options.crop_size,
            stride=runtime_options.stride,
            quantile=runtime_options.threshold_quantile,
            heatmap_std_scale=runtime_options.heatmap_std_scale,
            heatmap_quantile=runtime_options.heatmap_quantile,
            max_heatmap_samples=runtime_options.max_heatmap_samples,
            detect_batch_size=runtime_options.detect_batch_size,
            infer_long_side=runtime_options.infer_long_side,
            fast_calibrate=runtime_options.fast_calibrate,
        )
        self._emit(progress_callback, "calibrate_done", threshold=threshold)

        engine_path = self._engine_path(model_id, version_id)
        engine.save(str(engine_path))
        version_meta = self._save_version_meta(
            model_id=model_id,
            version_id=version_id,
            status="ready",
            threshold=threshold,
            samples=samples,
            failed_images=failed_images,
            options=runtime_options,
        )

        model_meta = {
            "model_id": model_id,
            "model_name": model_name,
            "model_dir": str(model_dir),
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "current_version_id": version_id,
            "versions": [version_meta],
        }
        self._write_json(self._model_meta_path(model_id), model_meta)
        registry["models"].append({"model_id": model_id, "model_name": model_name})
        self._save_registry(registry)
        self._emit(progress_callback, "train_done", model_id=model_id, version_id=version_id, threshold=threshold)
        return model_meta

    def _load_engine_for_model(self, model_id: str) -> Tuple[VisionMemoryEngine, Dict[str, Any], Dict[str, Any]]:
        model_meta = self.get_model(model_id)
        version_id = model_meta["current_version_id"]
        version_meta = self._load_version_meta(model_id, version_id)
        options = RuntimeOptions(**version_meta["runtime_options"])
        engine = self._create_engine(options)
        engine.load(str(self._engine_path(model_id, version_id)))
        return engine, model_meta, version_meta

    @staticmethod
    def _heatmap_to_regions(
        heatmap: np.ndarray,
        threshold: float,
        offset_xy: Tuple[int, int],
    ) -> List[Dict[str, Any]]:
        binary = (heatmap >= threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions: List[Dict[str, Any]] = []
        offset_x, offset_y = offset_xy
        for contour in contours:
            if contour.shape[0] < 3:
                continue
            contour = contour.reshape(-1, 2)
            x, y, w, h = cv2.boundingRect(contour.astype(np.int32))
            region_heat = heatmap[y:y + h, x:x + w]
            score = float(region_heat[region_heat >= threshold].mean()) if np.any(region_heat >= threshold) else float(region_heat.mean())
            regions.append(
                {
                    "contour": [[int(px + offset_x), int(py + offset_y)] for px, py in contour.tolist()],
                    "box": [int(x + offset_x), int(y + offset_y), int(x + w + offset_x), int(y + h + offset_y)],
                    "score": score,
                }
            )
        regions.sort(key=lambda item: item["score"], reverse=True)
        return regions

    def detect_image(
        self,
        model_id: str,
        image_path: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        image_bgr: Optional[np.ndarray] = None,
        include_heatmap_base64: bool = False,
        threshold: Optional[float] = None,
        heatmap_include_background: bool = True,
        heatmap_zero_below_threshold: Optional[bool] = None,
    ) -> Dict[str, Any]:
        engine, model_meta, version_meta = self._load_engine_for_model(model_id)
        runtime_options = RuntimeOptions(**version_meta["runtime_options"])
        active_threshold = float(threshold if threshold is not None else (version_meta.get("threshold") or engine.recommended_threshold or 1.0))
        zero_below_threshold = (
            runtime_options.heatmap_zero_below_threshold
            if heatmap_zero_below_threshold is None
            else bool(heatmap_zero_below_threshold)
        )
        image_bgr = maybe_load_image_bgr(image_path=image_path, image_bytes=image_bytes, image_bgr=image_bgr)
        roofs = self.segmenter.segment_image(image_bgr)
        if not roofs:
            return {
                "model_id": model_id,
                "model_name": model_meta["model_name"],
                "threshold": active_threshold,
                "is_anomaly": False,
                "score": 0.0,
                "roof_contours": [],
                "anomaly_regions": [],
                "heatmap_include_background": bool(heatmap_include_background),
                "heatmap_zero_below_threshold": bool(zero_below_threshold),
                "message": "No train roof contour detected.",
            }

        full_heatmap = np.zeros(image_bgr.shape[:2], dtype=np.float32)
        score_values: List[float] = []
        anomaly_regions: List[Dict[str, Any]] = []
        roof_contours: List[List[List[int]]] = []

        for roof in roofs:
            roof_contours.append(roof.contour)
            x1, y1, x2, y2 = roof.bbox
            crop_rgb = cv2.cvtColor(roof.masked_crop_bgr, cv2.COLOR_BGR2RGB)
            _, _, crop_heatmap = engine.detect_image(
                image_rgb=crop_rgb,
                crop_size=runtime_options.crop_size,
                stride=runtime_options.stride,
                threshold=active_threshold,
                detect_batch_size=runtime_options.detect_batch_size,
                infer_long_side=runtime_options.infer_long_side,
                heatmap_zero_below_threshold=zero_below_threshold,
            )
            crop_mask = roof.mask_crop > 0
            if crop_heatmap.shape[:2] != crop_mask.shape[:2]:
                crop_heatmap = cv2.resize(crop_heatmap, (crop_mask.shape[1], crop_mask.shape[0]), interpolation=cv2.INTER_CUBIC)
            crop_heatmap = crop_heatmap.astype(np.float32)
            crop_heatmap[~crop_mask] = 0.0
            full_heatmap[y1:y2, x1:x2] = np.maximum(full_heatmap[y1:y2, x1:x2], crop_heatmap)
            if np.any(crop_mask):
                score_values.extend(crop_heatmap[crop_mask].reshape(-1).tolist())
            anomaly_regions.extend(self._heatmap_to_regions(crop_heatmap, active_threshold, (x1, y1)))

        score = float(np.mean(score_values)) if score_values else 0.0
        is_anomaly = any(region["score"] >= active_threshold for region in anomaly_regions)

        response = {
            "model_id": model_id,
            "model_name": model_meta["model_name"],
            "version_id": model_meta["current_version_id"],
            "threshold": active_threshold,
            "is_anomaly": is_anomaly,
            "score": score,
            "roof_contours": roof_contours,
            "anomaly_regions": anomaly_regions,
            "heatmap_include_background": bool(heatmap_include_background),
            "heatmap_zero_below_threshold": bool(zero_below_threshold),
        }

        if include_heatmap_base64:
            vis_min = float(engine.heatmap_vis_min) if engine.heatmap_vis_min is not None else 0.0
            vis_max = float(engine.heatmap_vis_max) if engine.heatmap_vis_max is not None else max(active_threshold, 1e-6)
            heat_u8 = np.clip((full_heatmap - vis_min) / max(vis_max - vis_min, 1e-6), 0.0, 1.0)
            heat_u8 = (heat_u8 * 255).astype(np.uint8)
            heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
            heatmap_image = cv2.addWeighted(image_bgr, 0.55, heat_color, 0.45, 0) if heatmap_include_background else heat_color
            response["heatmap_base64"] = image_to_base64(heatmap_image, ".jpg")
        return response

    def list_samples(self, model_id: str, page: int = 1, page_size: int = 20) -> Dict[str, Any]:
        model_meta = self.get_model(model_id)
        version_id = model_meta["current_version_id"]
        samples = self._load_samples(model_id, version_id)
        page = max(1, int(page))
        page_size = max(1, min(200, int(page_size)))
        start = (page - 1) * page_size
        end = start + page_size
        return {
            "model_id": model_id,
            "version_id": version_id,
            "page": page,
            "page_size": page_size,
            "total": len(samples),
            "items": samples[start:end],
        }

    def get_sample_detail(self, model_id: str, sample_id: str) -> Dict[str, Any]:
        model_meta = self.get_model(model_id)
        version_id = model_meta["current_version_id"]
        samples = self._load_samples(model_id, version_id)
        for sample in samples:
            if sample["sample_id"] != sample_id:
                continue
            tiles_meta = self._load_sample_tiles(model_id, version_id, sample_id)
            return {
                "model_id": model_id,
                "version_id": version_id,
                "sample": sample,
                "sample_file_status": self._get_sample_file_status(model_id, version_id, sample),
                "tiles": tiles_meta.get("tiles", []),
                "canvas_size": tiles_meta.get("canvas_size", [0, 0]),
            }
        raise ValueError(f"Unknown sample_id: {sample_id}")

    def scan_samples_for_anomalies(self, model_id: str, threshold: Optional[float] = None) -> Dict[str, Any]:
        engine, model_meta, version_meta = self._load_engine_for_model(model_id)
        active_threshold = float(threshold if threshold is not None else (version_meta.get("threshold") or engine.recommended_threshold or 1.0))
        samples = self._load_samples(model_id, model_meta["current_version_id"])
        flagged = []

        for sample in samples:
            image_rgb = cv2.cvtColor(read_image_bgr(sample["processed_image_path"]), cv2.COLOR_BGR2RGB)
            is_anomaly, score, _ = engine.detect_image(
                image_rgb=image_rgb,
                crop_size=tuple(version_meta["runtime_options"]["crop_size"]),
                stride=tuple(version_meta["runtime_options"]["stride"]),
                threshold=active_threshold,
                detect_batch_size=int(version_meta["runtime_options"]["detect_batch_size"]),
                infer_long_side=int(version_meta["runtime_options"]["infer_long_side"]),
                heatmap_zero_below_threshold=bool(version_meta["runtime_options"]["heatmap_zero_below_threshold"]),
            )
            sample["last_scan_score"] = float(score)
            sample["last_scan_is_anomaly"] = bool(is_anomaly)
            sample["updated_at"] = utc_now()
            if is_anomaly:
                flagged.append(sample)

        self._save_samples(model_id, model_meta["current_version_id"], samples)
        return {
            "model_id": model_id,
            "version_id": model_meta["current_version_id"],
            "threshold": active_threshold,
            "flagged_count": len(flagged),
            "flagged_items": flagged,
        }

    def extract_roof_contours(
        self,
        image_path: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
    ) -> Dict[str, Any]:
        image_bgr = maybe_load_image_bgr(image_path=image_path, image_bytes=image_bytes)
        roofs = self.segmenter.segment_image(image_bgr)
        preview = image_bgr.copy()
        style = self._get_preview_style(preview)
        for idx, roof in enumerate(roofs, start=1):
            contour_np = np.asarray(roof.contour, dtype=np.int32)
            cv2.polylines(preview, [contour_np], True, (0, 255, 255), style["line_thickness"])
            x1, y1, x2, y2 = roof.bbox
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 128, 255), style["box_thickness"])
            cv2.putText(
                preview,
                f"{idx}:{roof.confidence:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                style["font_scale"],
                (0, 255, 255),
                style["font_thickness"],
                cv2.LINE_AA,
            )
        return {
            "count": len(roofs),
            "items": [
                {
                    "contour": roof.contour,
                    "bbox": roof.bbox,
                    "confidence": roof.confidence,
                }
                for roof in roofs
            ],
            "preview_base64": image_to_base64(preview, ".jpg"),
        }

    def get_sample_image_path(self, model_id: str, sample_id: str, kind: str = "processed") -> str:
        model_meta = self.get_model(model_id)
        version_id = model_meta["current_version_id"]
        samples = self._load_samples(model_id, version_id)
        for sample in samples:
            if sample["sample_id"] != sample_id:
                continue
            if kind == "raw":
                return sample["raw_image_path"]
            return sample["processed_image_path"]
        raise ValueError(f"Unknown sample_id: {sample_id}")

    def get_sample_tile_image_path(self, model_id: str, sample_id: str, tile_id: str) -> str:
        model_meta = self.get_model(model_id)
        version_id = model_meta["current_version_id"]
        tiles_meta = self._load_sample_tiles(model_id, version_id, sample_id)
        for tile in tiles_meta.get("tiles", []):
            if tile["tile_id"] == tile_id:
                return tile["image_path"]
        raise ValueError(f"Unknown tile_id: {tile_id}")

    def _rebuild_model_from_samples(self, model_id: str, samples: List[Dict[str, Any]], options: RuntimeOptions, recalibrate: bool = False) -> Dict[str, Any]:
        model_meta = self.get_model(model_id)
        version_id = model_meta["current_version_id"]
        self._save_samples(model_id, version_id, samples)
        rebuild = self._rebuild_engine_from_tiles(model_id, version_id, options, preserve_threshold=not recalibrate, calibrate=recalibrate)
        threshold = rebuild["threshold"]
        version_meta = self._load_version_meta(model_id, version_id)
        version_meta["threshold"] = threshold
        version_meta["sample_count"] = len(samples)
        self._write_json(self._version_meta_path(model_id, version_id), version_meta)
        model_meta["updated_at"] = utc_now()
        self._write_json(self._model_meta_path(model_id), model_meta)
        return version_meta

    def update_model_threshold(self, model_id: str, threshold: float) -> Dict[str, Any]:
        model_meta = self.get_model(model_id)
        version_id = model_meta["current_version_id"]
        version_meta = self._load_version_meta(model_id, version_id)
        options = RuntimeOptions(**version_meta["runtime_options"])
        engine = self._create_engine(options)
        engine.load(str(self._engine_path(model_id, version_id)))

        next_threshold = float(threshold)
        version_meta["threshold"] = next_threshold
        self._write_json(self._version_meta_path(model_id, version_id), version_meta)

        for version in model_meta.get("versions", []):
            if version.get("version_id") == version_id:
                version["threshold"] = next_threshold
                break
        model_meta["updated_at"] = utc_now()
        self._write_json(self._model_meta_path(model_id), model_meta)

        engine.recommended_threshold = next_threshold
        engine.save(str(self._engine_path(model_id, version_id)))

        return {
            "model_id": model_id,
            "version_id": version_id,
            "threshold": next_threshold,
            "model": model_meta,
        }

    def delete_model(self, model_id: str) -> Dict[str, Any]:
        model_meta = self.get_model(model_id)
        model_dir = Path(model_meta["model_dir"])
        if model_dir.exists():
            shutil.rmtree(model_dir)

        registry = self._read_registry()
        registry["models"] = [item for item in registry["models"] if item.get("model_id") != model_id]
        self._save_registry(registry)

        return {
            "deleted_model_id": model_id,
            "deleted_model_name": model_meta.get("model_name", ""),
        }

    def prune_model_assets(self, model_id: str) -> Dict[str, Any]:
        model_meta = self.get_model(model_id)
        version_id = model_meta["current_version_id"]
        version_dir = self._version_dir(model_id, version_id)
        target_dirs = [
            version_dir / "raw",
            version_dir / "processed",
            version_dir / "tiles",
            version_dir / "calibrate_processed",
        ]

        deleted_paths: List[str] = []
        deleted_file_count = 0
        released_bytes = 0
        for target_dir in target_dirs:
            if not target_dir.exists():
                continue
            for path in target_dir.rglob("*"):
                if path.is_file():
                    deleted_file_count += 1
                    released_bytes += path.stat().st_size
            shutil.rmtree(target_dir)
            deleted_paths.append(str(target_dir))

        return {
            "model_id": model_id,
            "version_id": version_id,
            "deleted_dir_count": len(deleted_paths),
            "deleted_file_count": deleted_file_count,
            "released_bytes": int(released_bytes),
            "deleted_paths": deleted_paths,
            "message": "已精简样本派生文件。异物检测和后续追加正样本不受影响，但无法再查看或维护历史向量库样本。",
        }

    def delete_sample(self, model_id: str, sample_id: str) -> Dict[str, Any]:
        model_meta = self.get_model(model_id)
        version_id = model_meta["current_version_id"]
        version_meta = self._load_version_meta(model_id, version_id)
        options = RuntimeOptions(**version_meta["runtime_options"])
        samples = self._load_samples(model_id, version_id)
        kept = []
        removed = None
        for sample in samples:
            if sample["sample_id"] == sample_id:
                removed = sample
                continue
            kept.append(sample)
        if removed is None:
            raise ValueError(f"Unknown sample_id: {sample_id}")
        for key in ("raw_image_path", "processed_image_path"):
            path = Path(removed[key])
            if path.exists():
                path.unlink()
        tile_dir = self._sample_tile_dir(model_id, version_id, sample_id)
        if tile_dir.exists():
            for child in sorted(tile_dir.rglob("*"), reverse=True):
                if child.is_file():
                    child.unlink()
                elif child.is_dir():
                    child.rmdir()
            if tile_dir.exists():
                tile_dir.rmdir()
        version_meta = self._rebuild_model_from_samples(model_id, kept, options)
        return {"deleted_sample_id": sample_id, "version": version_meta}

    def update_sample_contour(
        self,
        model_id: str,
        sample_id: str,
        contour: Sequence[Sequence[int]],
        note: str = "",
        enabled_tile_ids: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        model_meta = self.get_model(model_id)
        version_id = model_meta["current_version_id"]
        version_meta = self._load_version_meta(model_id, version_id)
        options = RuntimeOptions(**version_meta["runtime_options"])
        samples = self._load_samples(model_id, version_id)

        updated = None
        for sample in samples:
            if sample["sample_id"] != sample_id:
                continue
            raw_image = read_image_bgr(sample["raw_image_path"])
            rebuilt = self._build_processed_sample(
                source_image_name=sample["source_image_name"],
                source_image_path=sample["source_image_path"],
                image_bgr=raw_image,
                contour=contour,
                target_raw_path=Path(sample["raw_image_path"]),
                target_processed_path=Path(sample["processed_image_path"]),
                source_type=sample["source_type"],
                note=note or "manual contour update",
            )
            rebuilt["sample_id"] = sample_id
            rebuilt["created_at"] = sample["created_at"]
            rebuilt["updated_at"] = utc_now()
            sample.update(rebuilt)
            self._extract_sample_tiles(self._create_engine(options), model_id, version_id, sample, options)
            tiles_meta = self._load_sample_tiles(model_id, version_id, sample_id)
            enabled_set = set(enabled_tile_ids or [tile["tile_id"] for tile in tiles_meta.get("tiles", [])])
            for tile in tiles_meta.get("tiles", []):
                tile["enabled"] = tile["tile_id"] in enabled_set
            self._save_sample_tiles(model_id, version_id, sample_id, tiles_meta)
            updated = sample
            break
        if updated is None:
            raise ValueError(f"Unknown sample_id: {sample_id}")

        version_meta = self._rebuild_model_from_samples(model_id, samples, options)
        return {"sample": updated, "version": version_meta}

    def add_positive_sample(
        self,
        model_id: str,
        image_path: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        contour: Optional[Sequence[Any]] = None,
        note: str = "",
        append_max_vectors: int = DEFAULT_APPEND_MAX_VECTORS,
    ) -> Dict[str, Any]:
        engine, model_meta, version_meta = self._load_engine_for_model(model_id)
        version_id = model_meta["current_version_id"]
        version_dir = self._version_dir(model_id, version_id)
        samples = self._load_samples(model_id, version_id)
        options = RuntimeOptions(**version_meta["runtime_options"])
        image_bgr = maybe_load_image_bgr(image_path=image_path, image_bytes=image_bytes)

        contours = self._normalize_contours_payload(contour) if contour is not None else [roof.contour for roof in self.segmenter.segment_image(image_bgr)]
        if not contours:
            raise ValueError("No contour available for append_positive.")

        new_records = []
        raw_dir = version_dir / "raw"
        processed_dir = version_dir / "processed"
        for idx, current_contour in enumerate(contours):
            sample_id = new_id("sample")
            record = self._build_processed_sample(
                source_image_name=Path(image_path).name if image_path else f"upload_{idx + 1}.png",
                source_image_path=image_path or "",
                image_bgr=image_bgr,
                contour=current_contour,
                target_raw_path=raw_dir / f"{sample_id}.png",
                target_processed_path=processed_dir / f"{sample_id}.png",
                source_type="append_positive",
                note=note or "append_positive",
            )
            new_records.append(record)
        samples.extend(new_records)
        tile_engine = self._create_engine(options)
        new_embedding_chunks: List[torch.Tensor] = []
        for record in new_records:
            self._extract_sample_tiles(tile_engine, model_id, version_id, record, options)
            new_embedding_chunks.append(self._collect_enabled_embeddings_for_sample(model_id, version_id, record["sample_id"]))
        self._save_samples(model_id, version_id, samples)
        if not new_embedding_chunks:
            raise ValueError("No embeddings generated for appended samples.")
        new_embeddings = torch.cat(new_embedding_chunks, dim=0).float()
        append_limit = max(0, int(append_max_vectors))
        if append_limit > 0 and new_embeddings.shape[0] > append_limit:
            new_embeddings = tile_engine._compress_to_size(new_embeddings, append_limit)
        existing_memory = engine.memory_bank.cpu().float() if engine.memory_bank is not None else None
        engine.memory_bank = new_embeddings if existing_memory is None or existing_memory.numel() == 0 else torch.cat([existing_memory, new_embeddings], dim=0).float()
        engine._build_index()
        threshold = version_meta.get("threshold")
        engine.recommended_threshold = threshold
        engine.save(str(self._engine_path(model_id, version_id)))
        version_meta["threshold"] = threshold
        version_meta["sample_count"] = len(samples)
        self._write_json(self._version_meta_path(model_id, version_id), version_meta)
        model_meta["updated_at"] = utc_now()
        self._write_json(self._model_meta_path(model_id), model_meta)
        return {
            "model_id": model_id,
            "version_id": version_id,
            "added_count": len(new_records),
            "items": new_records,
            "threshold": threshold,
            "added_vector_count": int(new_embeddings.shape[0]),
            "append_max_vectors": append_limit,
        }

    def update_sample_tiles_enabled(self, model_id: str, sample_id: str, enabled_tile_ids: Sequence[str]) -> Dict[str, Any]:
        model_meta = self.get_model(model_id)
        version_id = model_meta["current_version_id"]
        version_meta = self._load_version_meta(model_id, version_id)
        options = RuntimeOptions(**version_meta["runtime_options"])
        tiles_meta = self._load_sample_tiles(model_id, version_id, sample_id)
        enabled_set = set(enabled_tile_ids)
        for tile in tiles_meta.get("tiles", []):
            tile["enabled"] = tile["tile_id"] in enabled_set
        self._save_sample_tiles(model_id, version_id, sample_id, tiles_meta)
        rebuild = self._rebuild_model_from_samples(model_id, self._load_samples(model_id, version_id), options, recalibrate=False)
        return {
            "sample_id": sample_id,
            "enabled_tile_ids": list(enabled_set),
            "version": rebuild,
        }
