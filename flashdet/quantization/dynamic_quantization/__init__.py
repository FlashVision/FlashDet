"""Dynamic and weight-only quantization algorithms."""

from .dynamic import quantize_dynamic, quantize_dynamic_int8
from .weight_only import (
    Int8WeightOnlyConv2d,
    Int8WeightOnlyLinear,
    quantize_weight_only,
)

__all__ = [
    "quantize_dynamic",
    "quantize_dynamic_int8",
    "quantize_weight_only",
    "Int8WeightOnlyConv2d",
    "Int8WeightOnlyLinear",
]
