from __future__ import annotations

import logging
from importlib.resources import files
from pathlib import Path
from typing import Any, Dict, List

from .builtin_models import BUILTIN_MODELS
from .schemas import InferenceModelConfig

LOGGER = logging.getLogger(__name__)


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

    @staticmethod
    def _runner_cache_key(model_name: str, backend_name: str) -> str:
        return f"{model_name}:{backend_name}"

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            config.to_public_dict()
            for config in self._configs.values()
            if config.enabled
        ]

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        config = self.get_model_config(model_name)
        payload = config.to_public_dict()
        try:
            payload["resolved_backend"] = self._resolve_backend_name(config)
        except Exception:
            payload["resolved_backend"] = None
        return payload

    def get_model_config(self, model_name: str) -> InferenceModelConfig:
        config = self._configs.get(model_name)
        if config is None or not config.enabled:
            raise ValueError(f"Unknown inference model: {model_name}")
        return config

    def _create_runner(self, config: InferenceModelConfig, backend_name: str, backend_config: Dict[str, Any]):
        print(f"------------{backend_name}------------------")
        if backend_name == "ultralytics":
            weight_path = backend_config.get("weight_path")
            if not weight_path or not Path(weight_path).exists():
                raise ValueError(f"Ultralytics model weight not found: {weight_path}")
            from .yolo_runner import YoloInferenceRunner

            LOGGER.info(
                "Preparing inference runner: model=%s backend=%s weight=%s",
                config.name,
                backend_name,
                weight_path,
            )
            return YoloInferenceRunner(config, backend_config=backend_config)
        if backend_name == "sophon_bmcv":
            weight_path = backend_config.get("weight_path")
            if not weight_path or not Path(weight_path).exists():
                raise ValueError(f"Sophon bmodel not found: {weight_path}")
            from .sophon_yolo_runner import SophonYoloInferenceRunner

            runner = SophonYoloInferenceRunner(config, backend_config=backend_config)
            runner.validate_environment()
            LOGGER.info(
                "Preparing inference runner: model=%s backend=%s weight=%s python_path=%s",
                config.name,
                backend_name,
                weight_path,
                backend_config.get("python_path"),
            )
            return runner
        raise ValueError(f"Unsupported inference backend: {backend_name}")

    def _resolve_backend_name(self, config: InferenceModelConfig) -> str:
        if config.backend != "auto":
            backend_config = config.get_backend_config(config.backend)
            self._create_runner(config, config.backend, backend_config)
            return config.backend
        errors: List[str] = []
        for backend_name in config.backend_names():
            backend_config = config.get_backend_config(backend_name)
            try:
                self._create_runner(config, backend_name, backend_config)
                return backend_name
            except Exception as exc:
                errors.append(f"{backend_name}: {exc}")
        raise ValueError(f"No available inference backend for {config.name}. " + "; ".join(errors))

    def get_runner(self, model_name: str, preferred_backend: str | None = None):
        config = self.get_model_config(model_name)
        use_auto_selection = config.backend == "auto" and preferred_backend is None
        backend_hint = "auto" if use_auto_selection else (preferred_backend or config.backend)
        cache_key = self._runner_cache_key(model_name, backend_hint)
        runner = self._runners.get(cache_key)
        if runner is None:
            if use_auto_selection:
                errors: List[str] = []
                for backend_name in config.backend_names():
                    backend_config = config.get_backend_config(backend_name)
                    try:
                        runner = self._create_runner(config, backend_name, backend_config)
                        resolved_cache_key = self._runner_cache_key(model_name, backend_name)
                        self._runners[resolved_cache_key] = runner
                        LOGGER.info(
                            "Selected inference backend: model=%s backend=%s",
                            model_name,
                            backend_name,
                        )
                        break
                    except Exception as exc:
                        LOGGER.warning(
                            "Inference backend unavailable: model=%s backend=%s reason=%s",
                            model_name,
                            backend_name,
                            exc,
                        )
                        errors.append(f"{backend_name}: {exc}")
                if runner is None:
                    raise ValueError(f"No available inference backend for {config.name}. " + "; ".join(errors))
            else:
                backend_name = preferred_backend or config.backend
                backend_config = config.get_backend_config(backend_name)
                runner = self._create_runner(config, backend_name, backend_config)
                resolved_cache_key = self._runner_cache_key(model_name, backend_name)
                self._runners[resolved_cache_key] = runner
                LOGGER.info(
                    "Selected inference backend: model=%s backend=%s",
                    model_name,
                    backend_name,
                )
            self._runners[cache_key] = runner
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
