"""
DFL-Free End-to-End Dual Detection Head for FlashDet.

Key design choices:
  - No Distribution Focal Loss (DFL): regression outputs 4 values
    (LTRB) directly instead of 4*(reg_max+1) distribution logits.
  - Dual-head: One-to-One (NMS-free inference) + One-to-Many (dense training).
  - Depthwise-separable convolutions with gated residual for efficiency.
  - Both heads share the same architecture but are independent modules.
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class E2EDetHead(nn.Module):
    """Depthwise-separable detection head with gated residual.

    Uses DW 3x3 -> PW 1x1 instead of full 3x3 convolutions,
    with learnable gated residual connections for stable training.

    Outputs:
        cls: [B, num_classes, H, W]
        reg: [B, 4, H, W] — raw LTRB logits (decode with softplus * stride)
    """

    def __init__(self, num_classes: int, in_channels: int):
        super().__init__()
        self.num_classes = num_classes

        self.cls_convs = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )
        self.reg_convs = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )
        self.cls_pred = nn.Conv2d(in_channels, num_classes, 1)
        self.reg_pred = nn.Conv2d(in_channels, 4, 1)
        self.cls_gate = nn.Parameter(torch.zeros(1))
        self.reg_gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (cls_logits, reg_preds) each [B, C, H, W]."""
        cls_feat = self.cls_convs(x) + self.cls_gate * x
        reg_feat = self.reg_convs(x) + self.reg_gate * x
        return self.cls_pred(cls_feat), self.reg_pred(reg_feat)


class E2EDualHead(nn.Module):
    """Dual detection head: One-to-One + One-to-Many.

    During training, both heads produce predictions. At inference,
    only the one-to-one head is used (NMS-free).

    Args:
        num_classes: Number of detection classes.
        in_channels: Input channels from each FPN level.
        num_levels: Number of FPN levels (default 3).
    """

    def __init__(self, num_classes: int, in_channels: int, num_levels: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.num_levels = num_levels

        self.o2o_heads = nn.ModuleList([
            E2EDetHead(num_classes, in_channels) for _ in range(num_levels)
        ])
        self.o2m_heads = nn.ModuleList([
            E2EDetHead(num_classes, in_channels) for _ in range(num_levels)
        ])

    def forward(
        self,
        features: List[torch.Tensor],
        training: bool = True,
    ) -> dict:
        """Forward pass through both heads.

        Args:
            features: List of FPN features, each [B, C, H_l, W_l].
            training: If True, return both heads; otherwise only o2o.

        Returns:
            Dict with:
                "o2o_cls": [B, N_total, num_classes]
                "o2o_reg": [B, N_total, 4]
                "o2m_cls": [B, N_total, num_classes]  (training only)
                "o2m_reg": [B, N_total, 4]             (training only)
                "feat_sizes": [(H, W)] per level
        """
        o2o_cls_list = []
        o2o_reg_list = []
        feat_sizes = []

        for head, feat in zip(self.o2o_heads, features):
            cls, reg = head(feat)
            B, _, H, W = cls.shape
            feat_sizes.append((H, W))
            o2o_cls_list.append(cls.permute(0, 2, 3, 1).reshape(B, H * W, -1))
            o2o_reg_list.append(reg.permute(0, 2, 3, 1).reshape(B, H * W, 4))

        result = {
            "o2o_cls": torch.cat(o2o_cls_list, dim=1),
            "o2o_reg": torch.cat(o2o_reg_list, dim=1),
            "feat_sizes": feat_sizes,
        }

        if training:
            o2m_cls_list = []
            o2m_reg_list = []
            for head, feat in zip(self.o2m_heads, features):
                cls, reg = head(feat)
                B, _, H, W = cls.shape
                o2m_cls_list.append(cls.permute(0, 2, 3, 1).reshape(B, H * W, -1))
                o2m_reg_list.append(reg.permute(0, 2, 3, 1).reshape(B, H * W, 4))
            result["o2m_cls"] = torch.cat(o2m_cls_list, dim=1)
            result["o2m_reg"] = torch.cat(o2m_reg_list, dim=1)

        return result
