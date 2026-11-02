"""Kernel-level fusion: op groups collapsed into single modules/functions."""

from flashdet.fusing.kernel_fusing.fused_ops import (
    ConvBnActFusionPass,
    FusedConvBnAct,
    FusedDWPointwise,
    fused_residual_act,
)

__all__ = [
    "ConvBnActFusionPass",
    "FusedConvBnAct",
    "FusedDWPointwise",
    "fused_residual_act",
]
