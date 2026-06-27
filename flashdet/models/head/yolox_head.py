"""YOLOX decoupled detection head.

Independent implementation based on:
    Ge et al., "YOLOX: Exceeding YOLO Series in 2021",
    arXiv:2107.08430, 2021.

License: MIT (same as FlashDet)
"""

from typing import Dict, List

import torch
import torch.nn as nn

from flashdet.models.layers.yolo_blocks import ConvBNSiLU


class YOLOXHead(nn.Module):
    """YOLOX decoupled detection head.

    Separates classification and regression into independent branches,
    which improves convergence compared to coupled heads.

    Args:
        num_classes: Number of object classes.
        in_channels: Input channel sizes per scale.
        stacked_convs: Number of stacked 3x3 convs per branch.
    """

    def __init__(self, num_classes: int, in_channels: List[int], stacked_convs: int = 2):
        super().__init__()
        self.num_classes = num_classes
        self.n_scales = len(in_channels)

        self.stems = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()

        for ch in in_channels:
            mid_ch = max(ch // 2, 64)
            self.stems.append(ConvBNSiLU(ch, mid_ch, 1))

            cls_layers = [ConvBNSiLU(mid_ch, mid_ch, 3) for _ in range(stacked_convs)]
            self.cls_convs.append(nn.Sequential(*cls_layers))
            self.cls_preds.append(nn.Conv2d(mid_ch, num_classes, 1))

            reg_layers = [ConvBNSiLU(mid_ch, mid_ch, 3) for _ in range(stacked_convs)]
            self.reg_convs.append(nn.Sequential(*reg_layers))
            self.reg_preds.append(nn.Conv2d(mid_ch, 4, 1))

            self.obj_preds.append(nn.Conv2d(mid_ch, 1, 1))

    def forward(self, features: List[torch.Tensor]) -> Dict[str, List[torch.Tensor]]:
        cls_outputs = []
        reg_outputs = []
        obj_outputs = []

        for i, feat in enumerate(features):
            x = self.stems[i](feat)

            cls_feat = self.cls_convs[i](x)
            cls_outputs.append(self.cls_preds[i](cls_feat))

            reg_feat = self.reg_convs[i](x)
            reg_outputs.append(self.reg_preds[i](reg_feat))
            obj_outputs.append(self.obj_preds[i](reg_feat))

        return {
            "cls_preds": cls_outputs,
            "reg_preds": reg_outputs,
            "obj_preds": obj_outputs,
        }
