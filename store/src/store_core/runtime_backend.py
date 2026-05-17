from __future__ import annotations

from dataclasses import dataclass

import torch


def sophon_sail_available() -> bool:
    try:
        import sophon.sail  # noqa: F401
    except Exception:
        return False
    return True


@dataclass(frozen=True)
class RuntimeBackendSelection:
    backend: str
    torch_device: str
    feature_backend: str
    knn_backend: str


def resolve_runtime_backend() -> RuntimeBackendSelection:
    if sophon_sail_available():
        return RuntimeBackendSelection(
            backend="bm",
            torch_device="cpu",
            feature_backend="bm",
            knn_backend="bm",
        )
    if torch.cuda.is_available():
        return RuntimeBackendSelection(
            backend="cuda",
            torch_device="cuda",
            feature_backend="torch",
            knn_backend="torch",
        )
    return RuntimeBackendSelection(
        backend="cpu",
        torch_device="cpu",
        feature_backend="torch",
        knn_backend="sklearn",
    )
