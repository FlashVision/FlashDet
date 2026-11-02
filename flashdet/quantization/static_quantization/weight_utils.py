"""Group-wise weight quantization helpers for AWQ/GPTQ and weight-only paths."""

from __future__ import annotations

from typing import Tuple

import torch


def group_qparams(
    weight: torch.Tensor,
    bits: int = 8,
    group_size: int = -1,
    symmetric: bool = True,
) -> Tuple[int, int]:
    """Return ``(num_groups, group_size)`` for a flattened weight row."""
    if bits not in (2, 4, 8):
        raise ValueError("bits must be 2, 4 or 8")
    in_features = weight.reshape(weight.shape[0], -1).shape[1]
    if group_size is None or group_size <= 0:
        group_size = in_features
    groups = (in_features + group_size - 1) // group_size
    return groups, group_size


def quantize_weight_per_group(
    weight: torch.Tensor,
    bits: int = 8,
    group_size: int = -1,
    symmetric: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Quantize ``[out, ...]`` weights per output channel/group.

    Returns:
        qweight: int8 tensor with the same flattened length as the weight.
        scales: ``[out, groups]`` float scales.
        zeros: ``[out, groups]`` zero points (0 for symmetric).
        group_size: effective group size.
    """
    flat = weight.reshape(weight.shape[0], -1)
    out_features, in_features = flat.shape
    groups, group_size = group_qparams(weight, bits, group_size, symmetric)
    padded = groups * group_size
    if padded != in_features:
        flat = torch.nn.functional.pad(flat, (0, padded - in_features))

    grouped = flat.reshape(out_features, groups, group_size)
    if symmetric:
        qmax = 2 ** (bits - 1) - 1
        scale = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
        zero = torch.zeros_like(scale)
    else:
        qmax = 2 ** bits - 1
        lo = grouped.amin(dim=-1, keepdim=True)
        hi = grouped.amax(dim=-1, keepdim=True)
        scale = ((hi - lo) / qmax).clamp(min=1e-8)
        zero = torch.round(-lo / scale).clamp(0, qmax)

    q = torch.round(grouped / scale + zero).clamp(
        -(qmax + 1) if symmetric else 0, qmax
    ).to(torch.int8)
    return q.reshape(out_features, padded), scale.squeeze(-1), zero.squeeze(-1), group_size


def dequantize_weight_per_group(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    shape: Tuple[int, ...],
    group_size: int,
) -> torch.Tensor:
    """Reconstruct a float weight tensor from group-quantized values."""
    out_features = shape[0]
    in_features = int(torch.tensor(shape[1:]).prod().item())
    groups = scales.shape[1]
    q = qweight.reshape(out_features, groups, group_size)
    dq = (q.float() - zeros.unsqueeze(-1)) * scales.unsqueeze(-1)
    return dq.reshape(out_features, -1)[:, :in_features].reshape(shape)


def fake_quant_weight(
    weight: torch.Tensor,
    bits: int = 8,
    group_size: int = -1,
    symmetric: bool = True,
) -> torch.Tensor:
    q, scales, zeros, gs = quantize_weight_per_group(
        weight, bits=bits, group_size=group_size, symmetric=symmetric
    )
    return dequantize_weight_per_group(q, scales, zeros, tuple(weight.shape), gs)
