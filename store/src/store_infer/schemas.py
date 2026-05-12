from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class InferenceModelConfig:
    name: str
    backend: str
    task_type: str
    weight_path: str
    class_names: List[str]
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_det: int = 100
    enabled: bool = True
    description: str = ""
    device: Optional[str] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any], package_root: Path) -> "InferenceModelConfig":
        weight_path = str((package_root / payload["weight_path"]).resolve())
        return cls(
            name=str(payload["name"]),
            backend=str(payload["backend"]),
            task_type=str(payload["task_type"]),
            weight_path=weight_path,
            class_names=[str(item) for item in payload.get("class_names", [])],
            conf_threshold=float(payload.get("conf_threshold", 0.25)),
            iou_threshold=float(payload.get("iou_threshold", 0.45)),
            max_det=int(payload.get("max_det", 100)),
            enabled=bool(payload.get("enabled", True)),
            description=str(payload.get("description", "")),
            device=payload.get("device"),
        )

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "backend": self.backend,
            "task_type": self.task_type,
            "class_names": list(self.class_names),
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "max_det": self.max_det,
            "enabled": self.enabled,
            "description": self.description,
        }
