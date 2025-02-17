import timm
import torch.nn as nn
from torch import Tensor
from enum import Enum


class LandmarkEstimationResNet(nn.Module):
    class ResNetBackbone(Enum):
        Resnet18 = "resnet18"
        Resnet34 = "resnet34"
        Resnet50 = "resnet50"
        Resnet101 = "resnet101"

    def __init__(self, backbone=ResNetBackbone.Resnet18, num_out: int = 2, num_landmarks: int = 68):
        super().__init__()
        backbone = timm.create_model(backbone.value, pretrained=True, num_classes=0)

        self.model = nn.Sequential(backbone,
                                   nn.GroupNorm(8, backbone.num_features),
                                   nn.SiLU(),
                                   nn.Linear(backbone.num_features, backbone.num_features),
                                   nn.GroupNorm(8, backbone.num_features),
                                   nn.Tanhshrink(),
                                   nn.Linear(backbone.num_features, num_landmarks * num_out),
                                   )

    def forward(self, imgs: Tensor) -> Tensor:
        x = self.model(imgs)
        return x