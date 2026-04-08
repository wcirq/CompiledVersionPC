from typing import Tuple

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class FeatureBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
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
