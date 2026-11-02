"""SmoothQuant-style activation/weight channel balancing.

SmoothQuant moves activation outliers into weights before static quantization.
This module implements the scaling math; calibration is left to the caller.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


@torch.no_grad()
def compute_smooth_scale(
    weight: torch.Tensor,
    activation_max: torch.Tensor,
    alpha: float = 0.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return one scale per Conv2d input channel.

    Args:
        weight: ``[out, in, kH, kW]`` conv weight.
        activation_max: ``[in]`` per-input-channel absolute activation max.
        alpha: 0.5 balances activation and weight difficulty.
    """
    if weight.ndim != 4:
        raise ValueError("weight must be a 4-D conv tensor")
    if activation_max.numel() != weight.shape[1]:
        raise ValueError(
            f"activation_max has {activation_max.numel()} values, "
            f"expected {weight.shape[1]}"
        )
    weight_scale = weight.abs().amax(dim=(0, 2, 3)).clamp(min=eps)
    act_scale = activation_max.detach().abs().clamp(min=eps)
    scale = (act_scale.pow(alpha) / weight_scale.pow(1 - alpha)).clamp(min=eps)
    return scale


@torch.no_grad()
def apply_smooth_scale(conv: nn.Conv2d, scale: torch.Tensor) -> nn.Conv2d:
    """Fold a per-input-channel scale into a Conv2d weight.

    The caller must scale the corresponding activation by ``1 / scale``.
    """
    if conv.groups != 1:
        raise ValueError(
            "apply_smooth_scale currently supports groups=1; depthwise layers "
            "should use a per-channel scale instead."
        )
    if scale.numel() != conv.in_channels:
        raise ValueError("scale length must equal conv.in_channels")
    conv.weight.data.mul_(scale.view(1, -1, 1, 1))
    return conv


def smoothquant_layer(
    conv: nn.Conv2d,
    activation_max: torch.Tensor,
    alpha: float = 0.5,
) -> Optional[torch.Tensor]:
    """Convenience helper: compute and apply SmoothQuant to one layer."""
    scale = compute_smooth_scale(conv.weight, activation_max, alpha=alpha)
    apply_smooth_scale(conv, scale)
    return scale
