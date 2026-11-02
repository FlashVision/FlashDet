"""Training performance and stability helpers.

These settings are optional and conservative by default: channels-last is
opt-in, TF32 is opt-in, and cuDNN autotuning is opt-in.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def configure_performance(
    device: torch.device,
    tf32: bool = False,
    cudnn_benchmark: bool = False,
    matmul_precision: str = "high",
) -> None:
    """Configure backend flags that affect training speed and determinism."""
    if matmul_precision in ("highest", "high", "medium"):
        torch.set_float32_matmul_precision(matmul_precision)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
        if tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("TF32 enabled")


def to_channels_last(model: nn.Module) -> nn.Module:
    """Convert a CNN model to channels-last memory format."""
    return model.to(memory_format=torch.channels_last)


def prepare_images(images: torch.Tensor, device: torch.device,
                   channels_last: bool = False) -> torch.Tensor:
    """Move images to the target device with optional channels-last layout."""
    images = images.to(device, non_blocking=True)
    if channels_last:
        images = images.to(memory_format=torch.channels_last)
    return images


def make_adamw(params, lr: float, weight_decay: float, **kwargs):
    """AdamW with fused CUDA kernels when available, else foreach."""
    if kwargs.pop("fused", True) and torch.cuda.is_available():
        try:
            return torch.optim.AdamW(
                params, lr=lr, weight_decay=weight_decay, fused=True, **kwargs
            )
        except (RuntimeError, TypeError):
            pass
    try:
        return torch.optim.AdamW(
            params, lr=lr, weight_decay=weight_decay, foreach=True, **kwargs
        )
    except (RuntimeError, TypeError):
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, **kwargs)


def make_sgd(params, lr: float, momentum: float = 0.9,
             weight_decay: float = 0.0, **kwargs):
    """SGD with foreach when supported."""
    try:
        return torch.optim.SGD(
            params, lr=lr, momentum=momentum, weight_decay=weight_decay,
            foreach=True, **kwargs,
        )
    except (RuntimeError, TypeError):
        return torch.optim.SGD(
            params, lr=lr, momentum=momentum, weight_decay=weight_decay, **kwargs
        )
