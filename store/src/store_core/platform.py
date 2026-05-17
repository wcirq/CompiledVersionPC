from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np

from .model_store import ModelStoreManager
from .package_data import get_default_bm_yolo_weight_path, get_default_yolo_weight_path
from .schemas import RuntimeOptions


class TrainRoofAnomalyStore:
    def __init__(
        self,
        root_dir: str = "./store_data",
        autostart_service: bool = True,
        service_host: str = "127.0.0.1",
        service_port: int = 55555,
        yolo_weight_path: Optional[str] = None,
        yolo_bm_weight_path: Optional[str] = None,
        yolo_conf_threshold: float = 0.5,
    ):
        self.root_dir = str(Path(root_dir).resolve())
        self.manager = ModelStoreManager(
            root_dir=self.root_dir,
            yolo_weight_path=yolo_weight_path or get_default_yolo_weight_path(),
            yolo_bm_weight_path=yolo_bm_weight_path or get_default_bm_yolo_weight_path(),
            yolo_conf_threshold=yolo_conf_threshold,
        )
        self.service_info: Optional[Dict[str, Any]] = None
        if autostart_service:
            from store_service.server import ensure_background_server

            self.service_info = ensure_background_server(
                manager=self.manager,
                host=service_host,
                start_port=service_port,
            )

    def serve_forever(self) -> Dict[str, Any]:
        from store_service.server import wait_for_background_server

        result = wait_for_background_server()
        if result.get("host") and result.get("port"):
            self.service_info = result
        return result

    def train_model(
        self,
        model_name: str,
        image_dir: str,
        runtime_options: Optional[RuntimeOptions] = None,
        save_root_dir: Optional[str] = None,
        calibrate_dir: Optional[str] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        if save_root_dir and str(Path(save_root_dir).resolve()) != self.root_dir:
            temp_manager = ModelStoreManager(
                root_dir=save_root_dir,
                yolo_weight_path=self.manager.yolo_weight_path,
                yolo_bm_weight_path=self.manager.yolo_bm_weight_path,
                yolo_conf_threshold=self.manager.yolo_conf_threshold,
            )
            return temp_manager.train_model(
                model_name=model_name,
                image_dir=image_dir,
                runtime_options=runtime_options,
                save_root_dir=save_root_dir,
                calibrate_dir=calibrate_dir,
                progress_callback=progress_callback,
            )
        return self.manager.train_model(
            model_name=model_name,
            image_dir=image_dir,
            runtime_options=runtime_options,
            save_root_dir=save_root_dir,
            calibrate_dir=calibrate_dir,
            progress_callback=progress_callback,
        )

    def detect_image(
        self,
        model_id: str,
        image_path: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        image_bgr: Optional[np.ndarray] = None,
        include_heatmap_base64: bool = False,
        threshold: Optional[float] = None,
        threshold_percent: Optional[float] = None,
        heatmap_include_background: bool = True,
        heatmap_zero_below_threshold: Optional[bool] = None,
        enable_tiling: Optional[bool] = None,
        postprocess_mode: Optional[str] = None,
        score_aggregation: Optional[str] = None,
        min_anomaly_area: Optional[int] = None,
        merge_distance_pixels: Optional[int] = None,
        use_segmentation: Optional[bool] = None,
        segment_conf_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        return self.manager.detect_image(
            model_id=model_id,
            image_path=image_path,
            image_bytes=image_bytes,
            image_bgr=image_bgr,
            include_heatmap_base64=include_heatmap_base64,
            threshold=threshold,
            threshold_percent=threshold_percent,
            heatmap_include_background=heatmap_include_background,
            heatmap_zero_below_threshold=heatmap_zero_below_threshold,
            enable_tiling=enable_tiling,
            postprocess_mode=postprocess_mode,
            score_aggregation=score_aggregation,
            min_anomaly_area=min_anomaly_area,
            merge_distance_pixels=merge_distance_pixels,
            use_segmentation=use_segmentation,
            segment_conf_threshold=segment_conf_threshold,
        )

    def list_models(self) -> Dict[str, Any]:
        return {"items": self.manager.list_models()}

    def detect_and_save_results(
        self,
        model_id: str,
        output_dir: str,
        image_path: Optional[str] = None,
        image_paths: Optional[Sequence[str]] = None,
        image_dir: Optional[str] = None,
        threshold: Optional[float] = None,
        threshold_percent: Optional[float] = None,
        use_segmentation: Optional[bool] = None,
        segment_conf_threshold: Optional[float] = None,
        enable_tiling: Optional[bool] = None,
        postprocess_mode: Optional[str] = None,
        score_aggregation: Optional[str] = None,
        min_anomaly_area: Optional[int] = None,
        merge_distance_pixels: Optional[int] = None,
        heatmap_zero_below_threshold: Optional[bool] = None,
        crop_expand_ratio: float = 0.6,
        save_process_files: bool = True,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        return self.manager.detect_and_save_results(
            model_id=model_id,
            output_dir=output_dir,
            image_path=image_path,
            image_paths=image_paths,
            image_dir=image_dir,
            threshold=threshold,
            threshold_percent=threshold_percent,
            use_segmentation=use_segmentation,
            segment_conf_threshold=segment_conf_threshold,
            enable_tiling=enable_tiling,
            postprocess_mode=postprocess_mode,
            score_aggregation=score_aggregation,
            min_anomaly_area=min_anomaly_area,
            merge_distance_pixels=merge_distance_pixels,
            heatmap_zero_below_threshold=heatmap_zero_below_threshold,
            crop_expand_ratio=crop_expand_ratio,
            save_process_files=save_process_files,
            progress_callback=progress_callback,
        )

    def get_model(self, model_id: str) -> Dict[str, Any]:
        return self.manager.get_model(model_id)

    def export_model_archive(self, model_id: str, deployment_only: bool = False) -> str:
        return self.manager.export_model_archive(model_id=model_id, deployment_only=deployment_only)

    def import_model_archive(self, archive_path: str) -> Dict[str, Any]:
        return self.manager.import_model_archive(archive_path=archive_path)

    def update_model_threshold(self, model_id: str, threshold: float) -> Dict[str, Any]:
        return self.manager.update_model_threshold(model_id=model_id, threshold=threshold)

    def delete_model(self, model_id: str) -> Dict[str, Any]:
        return self.manager.delete_model(model_id=model_id)

    def prune_model_assets(self, model_id: str) -> Dict[str, Any]:
        return self.manager.prune_model_assets(model_id=model_id)

    def list_samples(self, model_id: str, page: int = 1, page_size: int = 20) -> Dict[str, Any]:
        return self.manager.list_samples(model_id=model_id, page=page, page_size=page_size)

    def get_sample_detail(self, model_id: str, sample_id: str) -> Dict[str, Any]:
        return self.manager.get_sample_detail(model_id=model_id, sample_id=sample_id)

    def scan_vector_bank(self, model_id: str, threshold: Optional[float] = None) -> Dict[str, Any]:
        return self.manager.scan_samples_for_anomalies(model_id=model_id, threshold=threshold)

    def delete_sample(self, model_id: str, sample_id: str) -> Dict[str, Any]:
        return self.manager.delete_sample(model_id=model_id, sample_id=sample_id)

    def update_sample_contour(
        self,
        model_id: str,
        sample_id: str,
        contour: Sequence[Sequence[int]],
        note: str = "",
        enabled_tile_ids: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        return self.manager.update_sample_contour(
            model_id=model_id,
            sample_id=sample_id,
            contour=contour,
            note=note,
            enabled_tile_ids=enabled_tile_ids,
        )

    def add_positive_sample(
        self,
        model_id: str,
        image_path: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        contour: Optional[Sequence[Sequence[int]]] = None,
        note: str = "",
        append_max_vectors: int = 20,
    ) -> Dict[str, Any]:
        return self.manager.add_positive_sample(
            model_id=model_id,
            image_path=image_path,
            image_bytes=image_bytes,
            contour=contour,
            note=note,
            append_max_vectors=append_max_vectors,
        )

    def update_sample_tiles_enabled(self, model_id: str, sample_id: str, enabled_tile_ids: Sequence[str]) -> Dict[str, Any]:
        return self.manager.update_sample_tiles_enabled(model_id=model_id, sample_id=sample_id, enabled_tile_ids=enabled_tile_ids)

    def extract_roof_contours(
        self,
        image_path: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
    ) -> Dict[str, Any]:
        return self.manager.extract_roof_contours(image_path=image_path, image_bytes=image_bytes)
