from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn


DEFAULT_RESNET50_WEIGHTS = Path(__file__).resolve().parent / "weight" / "resnet50-11ad3fa6.pth"


class FeatureBackbone(nn.Module):
    def __init__(self, weights_path: Optional[str] = None):
        super().__init__()
        try:
            from torchvision.models import ResNet50_Weights, resnet50
        except ImportError as exc:
            raise ImportError(
                "torchvision is required for the torch backbone. "
                "Install torchvision or switch to --backbone_backend bm."
            ) from exc

        resolved_weights_path = Path(weights_path).resolve() if weights_path else DEFAULT_RESNET50_WEIGHTS
        if resolved_weights_path.exists():
            model = resnet50(weights=None)
            state_dict = torch.load(str(resolved_weights_path), map_location="cpu")
            model.load_state_dict(state_dict)
        else:
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        self.backend = "torch"
        self.weights_path = str(resolved_weights_path) if resolved_weights_path.exists() else None
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        feat2 = self.layer2(x)
        feat3 = self.layer3(feat2)
        return feat2, feat3
