from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any, Dict, List

from .builtin_models import BUILTIN_MODELS
from .schemas import InferenceModelConfig


class InferenceRegistry:
    def __init__(self):
        self._package_root = Path(files("store_infer"))
        self._configs: Dict[str, InferenceModelConfig] = {}
        self._runners: Dict[str, Any] = {}
        self._load_builtin_models()

    def _load_builtin_models(self) -> None:
        for item in BUILTIN_MODELS:
            config = InferenceModelConfig.from_dict(item, self._package_root)
            self._configs[config.name] = config

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            config.to_public_dict()
            for config in self._configs.values()
            if config.enabled
        ]

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        return self.get_model_config(model_name).to_public_dict()

    def get_model_config(self, model_name: str) -> InferenceModelConfig:
        config = self._configs.get(model_name)
        if config is None or not config.enabled:
            raise ValueError(f"Unknown inference model: {model_name}")
        if not Path(config.weight_path).exists():
            raise ValueError(f"Inference model weight not found: {config.weight_path}")
        return config

    def get_runner(self, model_name: str):
        config = self.get_model_config(model_name)
        runner = self._runners.get(model_name)
        if runner is None:
            if config.backend != "ultralytics":
                raise ValueError(f"Unsupported inference backend: {config.backend}")
            from .yolo_runner import YoloInferenceRunner

            runner = YoloInferenceRunner(config)
            self._runners[model_name] = runner
        return runner

    def run(
        self,
        model_name: str,
        image_path: str | None = None,
        image_bytes: bytes | None = None,
        image_bgr: Any = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        runner = self.get_runner(model_name)
        return runner.predict(
            image_path=image_path,
            image_bytes=image_bytes,
            image_bgr=image_bgr,
            **kwargs,
        )


_DEFAULT_REGISTRY: InferenceRegistry | None = None


def get_inference_registry() -> InferenceRegistry:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = InferenceRegistry()
    return _DEFAULT_REGISTRY


def list_models() -> List[Dict[str, Any]]:
    return get_inference_registry().list_models()


def run_inference(
    model_name: str,
    image_path: str | None = None,
    image_bytes: bytes | None = None,
    image_bgr: Any = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    return get_inference_registry().run(
        model_name=model_name,
        image_path=image_path,
        image_bytes=image_bytes,
        image_bgr=image_bgr,
        **kwargs,
    )
