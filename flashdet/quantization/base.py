"""Shared quantization primitives for FlashDet.

This module intentionally has no optional dependencies: it implements the
observers, fake-quantization math and configuration objects used by the
static, dynamic and QAT backends.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class QuantConfig:
    """Quantization parameters shared by all backends."""

    bits: int = 8
    symmetric: bool = False
    per_channel: bool = False
    observer: str = "minmax"
    percentile: float = 99.99
    channel_axis: int = 0
    reduce_range: bool = False

    @property
    def qmin(self) -> int:
        if self.symmetric:
            return -(2 ** (self.bits - 1)) + 1
        return 0

    @property
    def qmax(self) -> int:
        if self.symmetric:
            return 2 ** (self.bits - 1) - 1
        return 2 ** self.bits - 1


def _reduce_dims(x: torch.Tensor, axis: int) -> Tuple[int, ...]:
    dims = tuple(i for i in range(x.ndim) if i != axis)
    return dims if dims else (axis,)


def _qparams_from_range(
    min_val: torch.Tensor,
    max_val: torch.Tensor,
    cfg: QuantConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    eps = torch.finfo(torch.float32).eps
    if cfg.symmetric:
        max_abs = torch.maximum(min_val.abs(), max_val.abs())
        scale = (max_abs / max(cfg.qmax, 1)).clamp(min=eps)
        zero_point = torch.zeros_like(scale)
        return scale, zero_point

    min_val = torch.minimum(min_val, torch.zeros_like(min_val))
    max_val = torch.maximum(max_val, torch.zeros_like(max_val))
    scale = ((max_val - min_val) / (cfg.qmax - cfg.qmin)).clamp(min=eps)
    zero_point = torch.round(cfg.qmin - min_val / scale).clamp(
        cfg.qmin, cfg.qmax
    )
    return scale, zero_point


class BaseObserver(nn.Module):
    """Base class for activation/weight observers."""

    def __init__(self, cfg: Optional[QuantConfig] = None):
        super().__init__()
        self.cfg = cfg or QuantConfig()
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))
        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("zero_point", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("initialized", torch.tensor(False))

    def reset(self):
        self.min_val.fill_(float("inf"))
        self.max_val.fill_(float("-inf"))
        self.initialized.fill_(False)

    def observe(self, x: torch.Tensor):
        raise NotImplementedError

    def calculate_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if not bool(self.initialized):
            return self.scale, self.zero_point
        scale, zero_point = _qparams_from_range(
            self.min_val, self.max_val, self.cfg
        )
        zero_point = zero_point.to(self.zero_point.dtype)
        if self.scale.shape != scale.shape:
            self.scale = scale.clone()
            self.zero_point = zero_point.clone()
        else:
            self.scale.copy_(scale)
            self.zero_point.copy_(zero_point)
        return self.scale, self.zero_point

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.observe(x)
        return x


class MinMaxObserver(BaseObserver):
    """Min/max observer — the default PTQ calibration method."""

    def observe(self, x: torch.Tensor):
        x = x.detach()
        if self.cfg.per_channel:
            dims = _reduce_dims(x, self.cfg.channel_axis)
            cur_min = x.amin(dim=dims)
            cur_max = x.amax(dim=dims)
        else:
            cur_min = x.amin()
            cur_max = x.amax()
        self.min_val = torch.minimum(self.min_val, cur_min)
        self.max_val = torch.maximum(self.max_val, cur_max)
        self.initialized.fill_(True)


class PercentileObserver(BaseObserver):
    """Clip at a percentile instead of the absolute min/max."""

    def __init__(self, cfg: Optional[QuantConfig] = None, max_samples: int = 100_000):
        super().__init__(cfg)
        self.max_samples = max_samples
        self._samples: List[torch.Tensor] = []

    def reset(self):
        super().reset()
        self._samples.clear()

    def observe(self, x: torch.Tensor):
        self._samples.append(x.detach().flatten().cpu())
        if sum(s.numel() for s in self._samples) > self.max_samples:
            joined = torch.cat(self._samples)
            self._samples = [joined[-self.max_samples:]]
        self._refresh()

    def _refresh(self):
        if not self._samples:
            return
        joined = torch.cat(self._samples)
        p = self.cfg.percentile
        lo = torch.quantile(joined, (100.0 - p) / 100.0)
        hi = torch.quantile(joined, p / 100.0)
        self.min_val = torch.minimum(self.min_val, lo.to(self.min_val.device))
        self.max_val = torch.maximum(self.max_val, hi.to(self.max_val.device))
        self.initialized.fill_(True)


class HistogramObserver(BaseObserver):
    """Histogram/MSE clipping observer (used by static PTQ)."""

    def __init__(
        self,
        cfg: Optional[QuantConfig] = None,
        bins: int = 2048,
        candidates: int = 128,
        max_samples: int = 100_000,
    ):
        super().__init__(cfg)
        self.bins = bins
        self.candidates = candidates
        self.max_samples = max_samples
        self._samples: List[torch.Tensor] = []

    def reset(self):
        super().reset()
        self._samples.clear()

    def observe(self, x: torch.Tensor):
        self._samples.append(x.detach().flatten().cpu())
        if sum(s.numel() for s in self._samples) > self.max_samples:
            joined = torch.cat(self._samples)
            self._samples = [joined[-self.max_samples:]]
        self._refresh()

    def _refresh(self):
        if not self._samples:
            return
        values = torch.cat(self._samples)
        if values.numel() == 0:
            return
        absmax = values.abs().max().item()
        if absmax == 0:
            self.min_val.fill_(0.0)
            self.max_val.fill_(0.0)
            self.initialized.fill_(True)
            return

        best_err = float("inf")
        best_clip = absmax
        for frac in torch.linspace(0.5, 1.0, self.candidates):
            clip = max(absmax * float(frac), 1e-8)
            clipped = values.clamp(-clip, clip)
            # Uniform quantize/dequantize approximation of the candidate clip.
            scale = 2 * clip / (self.cfg.qmax - self.cfg.qmin)
            q = torch.round(clipped / scale).clamp(self.cfg.qmin, self.cfg.qmax)
            approx = q * scale
            err = torch.mean((values - approx) ** 2).item()
            if err < best_err:
                best_err = err
                best_clip = clip

        self.min_val.fill_(-best_clip)
        self.max_val.fill_(best_clip)
        self.initialized.fill_(True)


class MSEObserver(HistogramObserver):
    """Alias for histogram/MSE clipping (commonly called MSE observer)."""


class EntropyObserver(HistogramObserver):
    """Histogram clipping observer (KL-style family, MSE implementation).

    Kept as a separate name because many quantization toolchains expose
    "entropy"/KL clipping as a selectable calibration algorithm.
    """


OBSERVERS = {
    "minmax": MinMaxObserver,
    "percentile": PercentileObserver,
    "histogram": HistogramObserver,
    "mse": MSEObserver,
    "entropy": EntropyObserver,
}


def build_observer(name: str = "minmax", cfg: Optional[QuantConfig] = None) -> BaseObserver:
    try:
        cls = OBSERVERS[name.lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unknown observer {name!r}. Choose from: {', '.join(OBSERVERS)}"
        ) from exc
    return cls(cfg)


def quantize_tensor(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    qmin: int,
    qmax: int,
) -> torch.Tensor:
    """Quantize a tensor to integer values (still stored as float/int32)."""
    return torch.clamp(torch.round(x / scale + zero_point), qmin, qmax)


def dequantize_tensor(
    q: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
) -> torch.Tensor:
    return (q - zero_point) * scale


def fake_quantize(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    qmin: int,
    qmax: int,
) -> torch.Tensor:
    """Fake-quantize with a straight-through estimator."""
    q = quantize_tensor(x, scale, zero_point, qmin, qmax)
    dq = dequantize_tensor(q, scale, zero_point)
    return x + (dq - x).detach()


class FakeQuantize(nn.Module):
    """Observer-backed fake quantization module for QAT."""

    def __init__(self, cfg: Optional[QuantConfig] = None, observer: str = None):
        super().__init__()
        self.cfg = cfg or QuantConfig()
        self.observer = build_observer(observer or self.cfg.observer, self.cfg)
        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        self.observer.observe(x)
        scale, zero_point = self.observer.calculate_qparams()
        return fake_quantize(
            x, scale, zero_point, self.cfg.qmin, self.cfg.qmax
        )


class LSQFakeQuantize(FakeQuantize):
    """Learned Step Size Quantization (LSQ) fake quantizer."""

    def __init__(self, cfg: Optional[QuantConfig] = None):
        super().__init__(cfg)
        self._scale_param: Optional[nn.Parameter] = None

    def _ensure_scale(self, scale: torch.Tensor):
        if self._scale_param is None:
            self._scale_param = nn.Parameter(scale.detach().clone())
        return self._scale_param

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        with torch.no_grad():
            self.observer.observe(x)
            scale, zero_point = self.observer.calculate_qparams()
        scale = self._ensure_scale(scale)
        return fake_quantize(
            x, scale.abs().clamp(min=1e-8), zero_point,
            self.cfg.qmin, self.cfg.qmax,
        )


class PACTActivation(nn.Module):
    """PACT: learnable activation clipping for QAT."""

    def __init__(self, init_alpha: float = 8.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.abs().clamp(min=1e-3)
        return torch.minimum(torch.maximum(x, torch.zeros_like(x)), alpha)


def quant_error(x: torch.Tensor, dq: torch.Tensor) -> Dict[str, float]:
    """Return common reconstruction metrics for a quantization pair."""
    diff = (x - dq).float()
    mse = torch.mean(diff ** 2).item()
    signal = torch.mean(x.float() ** 2).item()
    return {
        "mse": mse,
        "rmse": math.sqrt(max(mse, 0.0)),
        "snr_db": 10.0 * math.log10(max(signal, 1e-12) / max(mse, 1e-12)),
    }
