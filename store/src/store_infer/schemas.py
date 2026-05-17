from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class InferenceModelConfig:
    name: str
    backend: str
    task_type: str
    weight_path: Optional[str]
    class_names: List[str]
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    imgsz: int = 640
    max_det: int = 100
    enabled: bool = True
    description: str = ""
    device: Optional[str] = None
    backends: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @staticmethod
    def _resolve_path(raw_path: Optional[str], package_root: Path) -> Optional[str]:
        if not raw_path:
            return None
        path = Path(raw_path)
        if not path.is_absolute():
            path = (package_root / path).resolve()
        return str(path)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any], package_root: Path) -> "InferenceModelConfig":
        weight_path = cls._resolve_path(payload.get("weight_path"), package_root)
        backends: Dict[str, Dict[str, Any]] = {}
        for backend_name, backend_payload in (payload.get("backends") or {}).items():
            item = dict(backend_payload or {})
            item["weight_path"] = cls._resolve_path(item.get("weight_path"), package_root)
            item["python_path"] = cls._resolve_path(item.get("python_path"), package_root)
            backends[str(backend_name)] = item
        return cls(
            name=str(payload["name"]),
            backend=str(payload["backend"]),
            task_type=str(payload["task_type"]),
            weight_path=weight_path,
            class_names=[str(item) for item in payload.get("class_names", [])],
            conf_threshold=float(payload.get("conf_threshold", 0.25)),
            iou_threshold=float(payload.get("iou_threshold", 0.45)),
            imgsz=int(payload.get("imgsz", 640)),
            max_det=int(payload.get("max_det", 100)),
            enabled=bool(payload.get("enabled", True)),
            description=str(payload.get("description", "")),
            device=payload.get("device"),
            backends=backends,
        )

    def get_backend_config(self, backend_name: str) -> Dict[str, Any]:
        if backend_name in self.backends:
            merged = dict(self.backends[backend_name] or {})
        else:
            merged = {}
            if backend_name == self.backend and self.weight_path:
                merged["weight_path"] = self.weight_path
                if self.device is not None:
                    merged["device"] = self.device
        if "weight_path" not in merged and self.weight_path and backend_name == self.backend:
            merged["weight_path"] = self.weight_path
        if "device" not in merged and self.device is not None:
            merged["device"] = self.device
        return merged

    def backend_names(self) -> List[str]:
        if self.backends:
            return list(self.backends.keys())
        return [self.backend]

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "backend": self.backend,
            "available_backends": self.backend_names(),
            "task_type": self.task_type,
            "class_names": list(self.class_names),
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "imgsz": self.imgsz,
            "max_det": self.max_det,
            "enabled": self.enabled,
            "description": self.description,
        }
