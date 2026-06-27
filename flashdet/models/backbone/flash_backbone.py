"""FlashDet backbone — scaled RepNeXt-style architecture.

Uses the same reparameterizable building blocks as PicoBackbone
(PicoBlock + StrideDown + MultiScaleConv), with explicit channel
and depth configs for N/S/M/L/X model variants.

Independent implementation based on:
    Mao et al., "RepNeXt: A Fast Multi-Scale CNN using Structural
    Reparameterization", arXiv 2024.

Stage layout::

    Stem   (3 → stem_ch, stride 2) + MaxPool (stride 2)   → /4
    Stage0  StrideDown → stem_ch×2, + PicoBlock×d          → /8   ← P3
    Stage1  StrideDown → stem_ch×4, + PicoBlock×d          → /16  ← P4
    Stage2  StrideDown → stem_ch×8, + PicoBlock×d + SPPF   → /32  ← P5
"""

import logging
from typing import List, Tuple

import torch
import torch.nn as nn

from flashdet.models.layers.reparam import PicoBlock, StrideDown
from flashdet.models.layers import SpatialPool

logger = logging.getLogger(__name__)


class FlashBackbone(nn.Module):
    """Scaled RepNeXt backbone for FlashDet-N/S/M/L/X.

    Same multi-scale depthwise + reparameterization design as PicoBackbone,
    scaled to larger channel widths and deeper stages. StrideDown doubles
    channels at each spatial reduction (stem_ch → ×2 → ×4 → ×8).

    Args:
        stem_channels: Number of channels in the stem convolution. Must be
            divisible by 4. StrideDown doubles at each stage, so backbone
            stages produce (stem×2, stem×4, stem×8) channels.
        stage_depths: Number of PicoBlock repeats per stage (P3, P4, P5).
        use_sppf: Add SpatialPool at P5 for global spatial context.
    """

    def __init__(
        self,
        stem_channels: int = 64,
        stage_depths: Tuple[int, ...] = (3, 6, 3),
        use_sppf: bool = True,
    ):
        super().__init__()
        assert stem_channels % 4 == 0, f"stem_channels ({stem_channels}) must be divisible by 4"

        stage_channels = (stem_channels * 2, stem_channels * 4, stem_channels * 8)

        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.pool = nn.MaxPool2d(3, 2, 1)

        in_ch = stem_channels
        self.stages = nn.ModuleList()
        for i, (ch, depth) in enumerate(zip(stage_channels, stage_depths)):
            layers: List[nn.Module] = [StrideDown(in_ch)]
            for _ in range(depth):
                layers.append(PicoBlock(ch))
            if i == len(stage_channels) - 1 and use_sppf:
                layers.append(SpatialPool(ch, ch))
            self.stages.append(nn.Sequential(*layers))
            in_ch = ch

        self.out_channels = list(stage_channels)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.pool(self.stem(x))
        outputs: List[torch.Tensor] = []
        for stage in self.stages:
            x = stage(x)
            outputs.append(x)
        return outputs

    @torch.no_grad()
    def fuse(self):
        """Fuse all reparameterizable branches for deployment."""
        def _fuse_recursive(module):
            for name, child in module.named_children():
                if hasattr(child, "fuse") and child is not self:
                    fused = child.fuse()
                    if fused is not None:
                        setattr(module, name, fused)
                        _fuse_recursive(fused)
                    else:
                        _fuse_recursive(child)
                else:
                    _fuse_recursive(child)
        _fuse_recursive(self)
        logger.info("FlashBackbone: fused all reparameterizable branches")
        return self
