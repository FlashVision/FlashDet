"""Layer fusion passes: weight folding and structural merging."""

from flashdet.fusing.layer_fusing.conv_bn import (
    ConvBnFusionPass,
    fold_bn_into_conv,
    fold_bn_into_linear,
    LinearBnFusionPass,
)
from flashdet.fusing.layer_fusing.linear_pair import (
    LinearPairFusionPass,
    merge_linears,
)
from flashdet.fusing.layer_fusing.rep_fusing import RepFusionPass

__all__ = [
    "ConvBnFusionPass",
    "LinearBnFusionPass",
    "LinearPairFusionPass",
    "RepFusionPass",
    "fold_bn_into_conv",
    "fold_bn_into_linear",
    "merge_linears",
]
