"""Uniform class-based interfaces for static PTQ algorithms."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import torch.nn as nn

from .awq import AWQConfig, quantize_awq
from .gptq import GPTQConfig, quantize_gptq
from .ptq import quantize_static


class PTQQuantizer:
    """Calibration-based static INT8 quantizer."""

    def __init__(self, model: nn.Module, observer: str = "minmax"):
        self.model = model
        self.observer = observer

    def quantize(
        self,
        calib_loader: Iterable,
        layer_prefixes: Optional[Sequence[str]] = None,
        keep_float_prefixes: Sequence[str] = ("head",),
        **kwargs,
    ) -> nn.Module:
        return quantize_static(
            self.model,
            calib_loader,
            layer_prefixes=layer_prefixes,
            keep_float_prefixes=keep_float_prefixes,
            observer=self.observer,
            **kwargs,
        )


class AWQQuantizer:
    """AWQ weight-only quantizer."""

    def __init__(
        self,
        model: nn.Module,
        bits: int = 4,
        group_size: int = 128,
        alpha: float = 0.5,
    ):
        self.model = model
        self.config = AWQConfig(bits=bits, group_size=group_size, alpha=alpha)

    def quantize(
        self,
        calib_loader: Iterable,
        layer_prefixes: Optional[Sequence[str]] = None,
        **kwargs,
    ) -> nn.Module:
        return quantize_awq(
            self.model,
            calib_loader,
            bits=self.config.bits,
            group_size=self.config.group_size,
            alpha=self.config.alpha,
            layer_prefixes=layer_prefixes,
            **kwargs,
        )


class GPTQQuantizer:
    """GPTQ Hessian-aware weight quantizer."""

    def __init__(
        self,
        model: nn.Module,
        bits: int = 8,
        group_size: int = -1,
        damp: float = 0.01,
    ):
        self.model = model
        self.config = GPTQConfig(bits=bits, group_size=group_size, damp=damp)

    def quantize(
        self,
        calib_loader: Iterable,
        layer_prefixes: Optional[Sequence[str]] = None,
        **kwargs,
    ) -> nn.Module:
        return quantize_gptq(
            self.model,
            calib_loader,
            bits=self.config.bits,
            group_size=self.config.group_size,
            damp=self.config.damp,
            layer_prefixes=layer_prefixes,
            **kwargs,
        )
