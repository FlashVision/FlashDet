"""GPTQ — Hessian-aware post-training weight quantization.

This is a compact, self-contained GPTQ implementation for Linear and regular
Conv2d layers. It performs column-by-column quantization with error
compensation using an input Hessian collected from calibration data.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .weight_utils import dequantize_weight_per_group, group_qparams

logger = logging.getLogger(__name__)


@dataclass
class GPTQConfig:
    bits: int = 8
    group_size: int = -1
    damp: float = 0.01
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


class GPTQLinear(nn.Module):
    """GPTQ-quantized Linear layer."""

    def __init__(self, module: nn.Linear, qweight, scales, zeros, group_size):
        super().__init__()
        self.register_buffer("qweight", qweight)
        self.register_buffer("scales", scales)
        self.register_buffer("zeros", zeros)
        self.register_buffer(
            "bias", module.bias.detach().clone() if module.bias is not None else None
        )
        self.group_size = group_size
        self.weight_shape = tuple(module.weight.shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = dequantize_weight_per_group(
            self.qweight, self.scales, self.zeros, self.weight_shape, self.group_size,
        )
        return F.linear(x, weight, self.bias)


class GPTQConv2d(nn.Module):
    """GPTQ-quantized Conv2d layer (groups=1)."""

    def __init__(self, module: nn.Conv2d, qweight, scales, zeros, group_size):
        super().__init__()
        self.register_buffer("qweight", qweight)
        self.register_buffer("scales", scales)
        self.register_buffer("zeros", zeros)
        self.register_buffer(
            "bias", module.bias.detach().clone() if module.bias is not None else None
        )
        self.stride = module.stride
        self.padding = module.padding
        self.dilation = module.dilation
        self.groups = module.groups
        self.group_size = group_size
        self.weight_shape = tuple(module.weight.shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = dequantize_weight_per_group(
            self.qweight, self.scales, self.zeros, self.weight_shape, self.group_size,
        )
        return F.conv2d(
            x, weight, self.bias, self.stride, self.padding,
            self.dilation, self.groups,
        )


def gptq_quantize_weight(
    weight: torch.Tensor,
    hessian: Optional[torch.Tensor] = None,
    bits: int = 8,
    group_size: int = -1,
    damp: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Quantize one weight matrix with GPTQ error compensation."""
    out_features = weight.shape[0]
    W = weight.reshape(out_features, -1).float()
    in_features = W.shape[1]
    groups, gs = group_qparams(weight, bits, group_size)
    padded = groups * gs
    if padded != in_features:
        W = F.pad(W, (0, padded - in_features))
    else:
        padded = in_features

    if hessian is None or hessian.numel() == 0:
        H = torch.eye(padded, device=W.device, dtype=torch.float32)
    else:
        H = hessian.float()
        if H.shape[0] != in_features:
            raise ValueError(
                f"Hessian is {tuple(H.shape)} but weight has {in_features} inputs"
            )
        if padded != in_features:
            Hpad = torch.eye(padded, device=W.device, dtype=torch.float32)
            Hpad[:in_features, :in_features] = H
            H = Hpad
        H = H + damp * torch.eye(padded, device=W.device, dtype=torch.float32)

    Hinv = torch.linalg.pinv(H)
    diag = torch.diagonal(Hinv).clamp(min=1e-8)

    qmax = 2 ** (bits - 1) - 1 if bits > 1 else 1
    grouped = W.reshape(out_features, groups, gs)
    scales = grouped.abs().amax(dim=-1).clamp(min=1e-8) / qmax
    zeros = torch.zeros_like(scales)
    Q = torch.zeros_like(W, dtype=torch.int8)

    for i in range(padded):
        g = i // gs
        w_col = W[:, i]
        q = torch.round(w_col / scales[:, g]).clamp(-qmax, qmax)
        Q[:, i] = q.to(torch.int8)
        err = w_col - q * scales[:, g]
        if i + 1 < padded:
            update = Hinv[i, i + 1:] / diag[i]
            W[:, i + 1:] -= err.unsqueeze(1) * update.unsqueeze(0)

    return Q, scales, zeros, gs


@torch.no_grad()
def _collect_hessians(
    model: nn.Module,
    calib_loader: Iterable,
    selected: list,
    num_batches: Optional[int],
    device: str,
) -> dict:
    hessians = {}
    hooks = []

    def _hook(name, module):
        def fn(_module, inputs):
            x = inputs[0].detach().float().to(device)
            if isinstance(module, nn.Linear):
                x = x.reshape(-1, x.shape[-1])
            else:
                patches = F.unfold(
                    x, module.kernel_size, module.dilation,
                    module.padding, module.stride,
                )
                x = patches.transpose(1, 2).reshape(-1, patches.shape[1])
            h = x.t() @ x
            hessians[name] = h if name not in hessians else hessians[name] + h
        return fn

    for name, module in selected:
        hooks.append(module.register_forward_pre_hook(_hook(name, module)))

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
    return hessians


def quantize_gptq(
    model: nn.Module,
    calib_loader: Iterable,
    bits: int = 8,
    group_size: int = -1,
    damp: float = 0.01,
    layer_prefixes: Optional[Sequence[str]] = None,
    num_batches: Optional[int] = None,
    inplace: bool = False,
    device: str = "cpu",
) -> nn.Module:
    """Apply GPTQ to selected Linear and regular Conv2d layers."""
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
        logger.warning("GPTQ: no Linear/Conv2d layers matched")
        return work

    hessians = _collect_hessians(
        work, calib_loader, selected, num_batches, device
    )
    replacements = []
    for name, module in selected:
        q, scales, zeros, gs = gptq_quantize_weight(
            module.weight.detach(), hessians.get(name),
            bits=bits, group_size=group_size, damp=damp,
        )
        if isinstance(module, nn.Linear):
            replacements.append((name, GPTQLinear(module, q, scales, zeros, gs)))
        else:
            replacements.append((name, GPTQConv2d(module, q, scales, zeros, gs)))

    for name, new in replacements:
        parent_name, _, child_name = name.rpartition(".")
        parent = work.get_submodule(parent_name) if parent_name else work
        setattr(parent, child_name, new)

    logger.info("GPTQ: quantized %d layers (%d-bit, group=%s)",
                len(replacements), bits, group_size)
    return work
