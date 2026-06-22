"""ResNet backbones shared by DETR and RT-DETR."""

from typing import List

import torch
import torch.nn as nn


class ResNetBackbone(nn.Module):
    """ResNet backbone producing a single feature map for DETR.

    Uses torchvision ResNet and returns the output of layer4 (stride 32).
    A 1x1 conv projects the channels to the transformer hidden dimension.
    """

    def __init__(self, variant: str = "resnet50", d_model: int = 256, pretrained: bool = True):
        super().__init__()
        import torchvision.models as tv_models

        factory = {
            "resnet18": (tv_models.resnet18, 512),
            "resnet34": (tv_models.resnet34, 512),
            "resnet50": (tv_models.resnet50, 2048),
            "resnet101": (tv_models.resnet101, 2048),
        }
        if variant not in factory:
            raise ValueError(f"Unknown ResNet variant '{variant}'. Choose from {list(factory.keys())}")

        builder, out_ch = factory[variant]
        weights = "DEFAULT" if pretrained else None
        resnet = builder(weights=weights)

        self.body = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )
        self.proj = nn.Conv2d(out_ch, d_model, 1)
        self.out_channels = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.body(x))


class ResNetMultiScaleBackbone(nn.Module):
    """ResNet backbone producing multi-scale features for RT-DETR.

    Returns features from layer2, layer3, layer4 (strides 8, 16, 32).

    Args:
        variant: ResNet variant ('resnet18', 'resnet34', 'resnet50', 'resnet101').
        pretrained: Whether to load ImageNet-pretrained weights.
    """

    CHANNEL_MAP = {
        "resnet18": [128, 256, 512],
        "resnet34": [128, 256, 512],
        "resnet50": [512, 1024, 2048],
        "resnet101": [512, 1024, 2048],
    }

    def __init__(self, variant: str = "resnet50", pretrained: bool = True):
        super().__init__()
        import torchvision.models as tv

        backbone_factory = {
            "resnet18": tv.resnet18,
            "resnet34": tv.resnet34,
            "resnet50": tv.resnet50,
            "resnet101": tv.resnet101,
        }
        builder = backbone_factory[variant]
        weights = "DEFAULT" if pretrained else None
        resnet = builder(weights=weights)

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.out_channels = self.CHANNEL_MAP[variant]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return [c3, c4, c5]
