from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import uuid


def utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


@dataclass
class RuntimeOptions:
    device: str = "cuda"
    backbone_backend: str = "auto"
    knn_backend: str = "auto"
    input_size: Tuple[int, int] = (640, 640)
    enable_tiling: bool = False
    crop_size: Tuple[int, int] = (640, 640)
    stride: Tuple[int, int] = (512, 512)
    batch_size: int = 32
    detect_batch_size: int = 8
    local_kernel: int = 3
    memory_ratio: float = 0.002
    target_embed_dimension: int = 1024
    knn_neighbors: int = 1
    knn_query_chunk_size: int = 8192
    infer_long_side: int = 0
    use_amp: bool = False
    threshold_quantile: float = 0.999
    heatmap_std_scale: float = 3.0
    heatmap_quantile: float = 0.999
    max_heatmap_samples: int = 2_000_000
    fast_calibrate: bool = False
    postprocess_mode: str = "adaptive"
    score_aggregation: str = "topk_mean"
    score_topk_ratio: float = 0.01
    adaptive_region_min_factor: float = 0.75
    adaptive_bbox_expand_ratio: float = 0.12
    max_embeddings: int = 1_200_000
    train_crop_scale_range: Tuple[float, float] = (0.7, 1.3)
    train_crop_round_multiple: int = 8
    train_min_crop_size: int = 240
    random_seed: int = 42
    stream_to_disk: bool = True
    stream_max_embeddings: int = 0
    online_compress_ratio: float = 0.5
    online_novelty_threshold: float = 0.0
    cleanup_stream_dir: bool = True
    bm_bmodel_path: Optional[str] = None
    bm_device_id: int = 0
    bm_db_chunk_size: int = 4096
    bm_graph_name: Optional[str] = None
    bm_query_input_name: Optional[str] = None
    bm_database_input_name: Optional[str] = None
    bm_output_name: Optional[str] = None
    backbone_bmodel_path: Optional[str] = None
    backbone_device_id: int = 0
    backbone_graph_name: Optional[str] = None
    backbone_input_name: Optional[str] = None
    backbone_feat2_output_name: Optional[str] = None
    backbone_feat3_output_name: Optional[str] = None
    heatmap_zero_below_threshold: bool = True

    def to_engine_kwargs(self) -> Dict[str, Any]:
        return {
            "device": self.device,
            "backbone": "resnet50",
            "backbone_backend": self.backbone_backend,
            "backbone_bmodel_path": self.backbone_bmodel_path,
            "backbone_device_id": self.backbone_device_id,
            "backbone_graph_name": self.backbone_graph_name,
            "backbone_input_name": self.backbone_input_name,
            "backbone_feat2_output_name": self.backbone_feat2_output_name,
            "backbone_feat3_output_name": self.backbone_feat3_output_name,
            "input_size": self.input_size,
            "memory_ratio": self.memory_ratio,
            "target_embed_dimension": self.target_embed_dimension,
            "local_kernel": self.local_kernel,
            "knn_neighbors": self.knn_neighbors,
            "knn_backend": self.knn_backend,
            "knn_query_chunk_size": self.knn_query_chunk_size,
            "bm_bmodel_path": self.bm_bmodel_path,
            "bm_device_id": self.bm_device_id,
            "bm_db_chunk_size": self.bm_db_chunk_size,
            "bm_graph_name": self.bm_graph_name,
            "bm_query_input_name": self.bm_query_input_name,
            "bm_database_input_name": self.bm_database_input_name,
            "bm_output_name": self.bm_output_name,
            "use_amp": self.use_amp,
        }


@dataclass
class SampleRecord:
    sample_id: str
    source_image_name: str
    source_image_path: str
    raw_image_path: str
    processed_image_path: str
    contour: List[List[int]]
    bbox: List[int]
    source_type: str
    status: str = "active"
    note: str = ""
    last_scan_score: Optional[float] = None
    last_scan_is_anomaly: Optional[bool] = None
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class ModelVersionInfo:
    version_id: str
    version_dir: str
    created_at: str
    status: str
    threshold: Optional[float]
    sample_count: int
    failed_image_count: int
    failed_images: List[str]
    runtime_options: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class ModelInfo:
    model_id: str
    model_name: str
    model_dir: str
    current_version_id: str
    created_at: str
    updated_at: str
    versions: List[ModelVersionInfo]

    def to_dict(self) -> Dict[str, Any]:
        payload = self.__dict__.copy()
        payload["versions"] = [item.to_dict() for item in self.versions]
        return payload
