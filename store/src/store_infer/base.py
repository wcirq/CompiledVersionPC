from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

from .schemas import InferenceModelConfig


class BaseInferenceRunner(ABC):
    def __init__(self, config: InferenceModelConfig):
        self.config = config

    @abstractmethod
    def predict(
        self,
        image_path: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        image_bgr: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        raise NotImplementedError
