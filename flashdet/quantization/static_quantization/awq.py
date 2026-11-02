"""AWQ — Activation-aware Weight Quantization.

AWQ protects salient weights by scaling input channels using activation
magnitudes before group-wise weight-only quantization. This implementation is
self-contained and works with Linear and Conv2d layers.
"""

from __future__ import annotations

import copy
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .weight_utils import (
    dequantize_weight_per_group,
    quantize_weight_per_group,
)

logger = logging.getLogger(__name__)


@dataclass
class AWQConfig:
    bits: int = 4
    group_size: int = 128
    alpha: float = 0.5
    symmetric: bool = True


def _iter_calibration_batches(loader: Iterable):
    for batch in loader:
        if isinstance(batch, (tuple, list)):
            yield batch[0]
        elif isinstance(batch, dict):
            yield batch.get("image", batch.get("images", batch))
        else:
            yield batch


def _matches(name: str, prefixes: Optional[Sequence[str]]) -> bool:
    if not prefixes:
        return True
    return any(name == p or name.startswith(p + ".") for p in prefixes)


class AWQLinear(nn.Module):
    """Weight-only AWQ Linear layer."""

    def __init__(self, module: nn.Linear, input_scale: torch.Tensor, cfg: AWQConfig):
        super().__init__()
        weight = module.weight.detach()
        scale = input_scale.to(weight.device).clamp(min=1e-8)
        scaled = weight * scale.view(1, -1)
        q, qscale, zero, gs = quantize_weight_per_group(
            scaled, bits=cfg.bits, group_size=cfg.group_size, symmetric=cfg.symmetric,
        )
        self.register_buffer("qweight", q)
        self.register_buffer("scales", qscale)
        self.register_buffer("zeros", zero)
        self.register_buffer("input_scale", scale)
        self.register_buffer(
            "bias", module.bias.detach().clone() if module.bias is not None else None
        )
        self.bits = cfg.bits
        self.group_size = gs
        self.weight_shape = tuple(weight.shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = dequantize_weight_per_group(
            self.qweight, self.scales, self.zeros, self.weight_shape, self.group_size,
        )
        return F.linear(x / self.input_scale.view(1, -1), weight, self.bias)


class AWQConv2d(nn.Module):
    """Weight-only AWQ Conv2d layer."""

    def __init__(self, module: nn.Conv2d, input_scale: torch.Tensor, cfg: AWQConfig):
        super().__init__()
        weight = module.weight.detach()
        scale = input_scale.to(weight.device).clamp(min=1e-8)
        scaled = weight * scale.view(1, -1, 1, 1)
        q, qscale, zero, gs = quantize_weight_per_group(
            scaled, bits=cfg.bits, group_size=cfg.group_size, symmetric=cfg.symmetric,
        )
        self.register_buffer("qweight", q)
        self.register_buffer("scales", qscale)
        self.register_buffer("zeros", zero)
        self.register_buffer("input_scale", scale)
        self.register_buffer(
            "bias", module.bias.detach().clone() if module.bias is not None else None
        )
        self.stride = module.stride
        self.padding = module.padding
        self.dilation = module.dilation
        self.groups = module.groups
        self.bits = cfg.bits
        self.group_size = gs
        self.weight_shape = tuple(weight.shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = dequantize_weight_per_group(
            self.qweight, self.scales, self.zeros, self.weight_shape, self.group_size,
        )
        return F.conv2d(
            x / self.input_scale.view(1, -1, 1, 1),
            weight, self.bias, self.stride, self.padding,
            self.dilation, self.groups,
        )


@torch.no_grad()
def _collect_activation_scales(
    model: nn.Module,
    calib_loader: Iterable,
    selected: list,
    num_batches: Optional[int],
    device: str,
) -> dict:
    scales = defaultdict(lambda: None)
    hooks = []

    def _hook(name):
        def fn(_module, inputs):
            x = inputs[0].detach()
            if x.ndim == 2:
                cur = x.abs().amax(dim=0)
            else:
                cur = x.abs().amax(dim=(0, 2, 3))
            scales[name] = cur if scales[name] is None else torch.maximum(scales[name], cur)
        return fn

    for name, module in selected:
        hooks.append(module.register_forward_pre_hook(_hook(name)))

    seen = 0
    model.eval().to(device)
    with torch.no_grad():
        for images in _iter_calibration_batches(calib_loader):
            if num_batches is not None and seen >= num_batches:
                break
            model(images.to(device))
            seen += 1
    for hook in hooks:
        hook.remove()
    return dict(scales)


def quantize_awq(
    model: nn.Module,
    calib_loader: Iterable,
    bits: int = 4,
    group_size: int = 128,
    alpha: float = 0.5,
    layer_prefixes: Optional[Sequence[str]] = None,
    num_batches: Optional[int] = None,
    inplace: bool = False,
    device: str = "cpu",
) -> nn.Module:
    """Apply AWQ weight-only quantization to Linear/Conv2d layers."""
    work = model if inplace else copy.deepcopy(model)
    selected = [
        (name, module)
        for name, module in work.named_modules()
        if name and _matches(name, layer_prefixes)
        and (
            isinstance(module, nn.Linear)
            or (isinstance(module, nn.Conv2d) and module.groups == 1)
        )
    ]
    if not selected:
        logger.warning("AWQ: no Linear/Conv2d layers matched")
        return work

    act_scales = _collect_activation_scales(
        work, calib_loader, selected, num_batches, device
    )
    cfg = AWQConfig(bits=bits, group_size=group_size, alpha=alpha)

    replacements = []
    for name, module in selected:
        act = act_scales.get(name)
        if act is None:
            continue
        weight = module.weight.detach()
        if isinstance(module, nn.Linear):
            w_scale = weight.abs().amax(dim=0).clamp(min=1e-8)
        else:
            w_scale = weight.abs().amax(dim=(0, 2, 3)).clamp(min=1e-8)
        act_scale = act.to(weight.device).clamp(min=1e-8)
        input_scale = (act_scale.pow(alpha) / w_scale.pow(1 - alpha)).clamp(min=1e-8)
        if isinstance(module, nn.Linear):
            replacements.append((name, module, AWQLinear(module, input_scale, cfg)))
        else:
            replacements.append((name, module, AWQConv2d(module, input_scale, cfg)))

    for name, _old, new in replacements:
        parent_name, _, child_name = name.rpartition(".")
        parent = work.get_submodule(parent_name) if parent_name else work
        setattr(parent, child_name, new)

    logger.info("AWQ: quantized %d layers (%d-bit, group=%s)",
                len(replacements), bits, group_size)
    return work


def awq_quantize_weight(
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    bits: int = 4,
    group_size: int = 128,
    alpha: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Standalone helper returning ``(qweight, scales, zeros, input_scale, group)``."""
    weight = weight.detach()
    if weight.ndim == 2:
        w_scale = weight.abs().amax(dim=0).clamp(min=1e-8)
    else:
        w_scale = weight.abs().amax(dim=(0, 2, 3)).clamp(min=1e-8)
    act = activation_scale.to(weight.device).clamp(min=1e-8)
    input_scale = (act.pow(alpha) / w_scale.pow(1 - alpha)).clamp(min=1e-8)
    if weight.ndim == 2:
        scaled = weight * input_scale.view(1, -1)
    else:
        scaled = weight * input_scale.view(1, -1, 1, 1)
    q, scales, zeros, gs = quantize_weight_per_group(
        scaled, bits=bits, group_size=group_size, symmetric=True,
    )
    return q, scales, zeros, input_scale, gs
