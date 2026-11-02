"""Dynamic INT8 quantization.

Linear layers use PyTorch's native dynamic quantized kernel. Conv2d layers use
a weight-only INT8 fallback because PyTorch has no portable dynamic Conv2d op.
"""

from __future__ import annotations

import copy
from typing import Optional, Sequence

import torch
import torch.nn as nn

from .weight_only import quantize_weight_only


def _matches(name: str, prefixes: Optional[Sequence[str]]) -> bool:
    if not prefixes:
        return True
    return any(name == p or name.startswith(p + ".") for p in prefixes)


def quantize_dynamic(
    model: nn.Module,
    layer_prefixes: Optional[Sequence[str]] = None,
    dtype: torch.dtype = torch.qint8,
    inplace: bool = False,
    linear_only: bool = False,
) -> nn.Module:
    """Apply dynamic INT8 quantization to Linear layers, INT8 weight-only to Conv2d."""
    work = model if inplace else copy.deepcopy(model)

    if linear_only:
        return torch.ao.quantization.quantize_dynamic(
            work, {nn.Linear}, dtype=dtype, inplace=True,
        )

    # First replace Conv2d with the weight-only fallback.
    work = quantize_weight_only(work, layer_prefixes=layer_prefixes, inplace=True)

    # Then use PyTorch's native dynamic kernel for Linear layers.
    from torch.ao.nn.quantized.dynamic import Linear as DynamicLinear
    from torch.ao.quantization import default_dynamic_qconfig

    work.eval()
    for name, module in list(work.named_modules()):
        if not name or not _matches(name, layer_prefixes):
            continue
        if isinstance(module, nn.Linear):
            module.qconfig = default_dynamic_qconfig
            parent_name, _, child_name = name.rpartition(".")
            parent = work.get_submodule(parent_name) if parent_name else work
            setattr(parent, child_name, DynamicLinear.from_float(module))
    return work


def quantize_dynamic_int8(model: nn.Module, **kwargs) -> nn.Module:
    return quantize_dynamic(model, dtype=torch.qint8, **kwargs)
