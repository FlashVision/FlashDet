"""Weight-only INT8 modules for layers without native dynamic kernels.

PyTorch ships dynamic INT8 kernels for Linear/RNN. Convolutions usually need
static quantization, so this module provides a deployment-safe weight-only
fallback: INT8 weights + FP32 activations.
"""

from __future__ import annotations

import copy
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class Int8WeightOnlyConv2d(nn.Module):
    """Conv2d with per-output-channel INT8 weights and FP32 activations."""

    def __init__(self, module: nn.Conv2d):
        super().__init__()
        weight = module.weight.detach()
        scale = weight.abs().amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-8) / 127.0
        qweight = torch.round(weight / scale).clamp(-127, 127).to(torch.int8)
        self.register_buffer("weight_int8", qweight)
        self.register_buffer("weight_scale", scale)
        self.register_buffer(
            "bias",
            module.bias.detach().clone() if module.bias is not None else None,
        )
        self.stride = module.stride
        self.padding = module.padding
        self.dilation = module.dilation
        self.groups = module.groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight_int8.float() * self.weight_scale
        return F.conv2d(
            x, weight, self.bias, self.stride, self.padding,
            self.dilation, self.groups,
        )


class Int8WeightOnlyLinear(nn.Module):
    """Linear with per-output-channel INT8 weights and FP32 activations."""

    def __init__(self, module: nn.Linear):
        super().__init__()
        weight = module.weight.detach()
        scale = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
        qweight = torch.round(weight / scale).clamp(-127, 127).to(torch.int8)
        self.register_buffer("weight_int8", qweight)
        self.register_buffer("weight_scale", scale)
        self.register_buffer(
            "bias",
            module.bias.detach().clone() if module.bias is not None else None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight_int8.float() * self.weight_scale, self.bias)


def _matches(name: str, prefixes: Optional[Sequence[str]]) -> bool:
    if not prefixes:
        return True
    return any(name == p or name.startswith(p + ".") for p in prefixes)


def quantize_weight_only(
    model: nn.Module,
    layer_prefixes: Optional[Sequence[str]] = None,
    inplace: bool = False,
) -> nn.Module:
    """Replace selected Conv2d/Linear layers with weight-only INT8 modules."""
    work = model if inplace else copy.deepcopy(model)
    replacements = []
    for name, module in work.named_modules():
        if not name or not _matches(name, layer_prefixes):
            continue
        if isinstance(module, nn.Conv2d):
            replacements.append((name, module, Int8WeightOnlyConv2d(module)))
        elif isinstance(module, nn.Linear):
            replacements.append((name, module, Int8WeightOnlyLinear(module)))

    for name, old, new in replacements:
        parent_name, _, child_name = name.rpartition(".")
        parent = work.get_submodule(parent_name) if parent_name else work
        setattr(parent, child_name, new)
    return work
