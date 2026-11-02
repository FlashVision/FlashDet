"""Static INT8 quantization for FlashDet.

Two paths are provided:

- :func:`quantize_static` — calibration-based INT8 fake quantization that works
  on FlashDet's real graph and supports selective layer groups. Weights and
  activations are rounded to INT8, then dequantized for float execution.
- :func:`quantize_static_ao` — PyTorch eager-mode quantized kernels. This
  requires a model with QuantStub/DeQuantStub boundaries and a graph that is
  compatible with eager quantization.

For deployment, export the calibrated/fake-quant model to ONNX, TensorRT,
OpenVINO or a hardware-specific quantizer.
"""

from __future__ import annotations

import copy
import logging
from collections import defaultdict
from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn

from flashdet.quantization.base import (
    QuantConfig,
    build_observer,
    fake_quantize,
)

logger = logging.getLogger(__name__)


def _iter_calibration_batches(loader: Iterable):
    for batch in loader:
        if isinstance(batch, (tuple, list)):
            yield batch[0]
        elif isinstance(batch, dict):
            yield batch.get("image", batch.get("images", batch))
        else:
            yield batch


@torch.no_grad()
def calibrate(
    model: nn.Module,
    calib_loader: Iterable,
    num_batches: Optional[int] = None,
    device: str = "cpu",
) -> int:
    """Run representative data through the model in eval mode."""
    model.eval().to(device)
    seen = 0
    for images in _iter_calibration_batches(calib_loader):
        if num_batches is not None and seen >= num_batches:
            break
        model(images.to(device))
        seen += 1
    logger.info("Quantization calibration finished (%d batches)", seen)
    return seen


def _matches(name: str, prefixes: Optional[Sequence[str]]) -> bool:
    if not prefixes:
        return True
    return any(name == p or name.startswith(p + ".") for p in prefixes)


def _selected_modules(
    model: nn.Module,
    layer_prefixes: Optional[Sequence[str]],
    keep_float_prefixes: Sequence[str],
) -> list:
    selected = []
    for name, module in model.named_modules():
        if not name or len(list(module.children())) > 0:
            continue
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue
        if _matches(name, layer_prefixes) and not _matches(name, keep_float_prefixes):
            selected.append((name, module))
    return selected


def _weight_qparams(weight: torch.Tensor, per_channel: bool):
    cfg = QuantConfig(symmetric=True, per_channel=per_channel, channel_axis=0)
    observer = build_observer("minmax", cfg)
    observer.observe(weight)
    scale, zero_point = observer.calculate_qparams()
    if per_channel and weight.ndim > 1:
        shape = [scale.numel()] + [1] * (weight.ndim - 1)
        scale = scale.reshape(shape)
        zero_point = zero_point.reshape(shape)
    return (scale, zero_point), cfg


def quantize_static(
    model: nn.Module,
    calib_loader: Iterable,
    backend: str = "fbgemm",
    layer_prefixes: Optional[Sequence[str]] = None,
    keep_float_prefixes: Sequence[str] = ("head",),
    per_channel: bool = True,
    num_batches: Optional[int] = None,
    inplace: bool = False,
    device: str = "cpu",
    observer: str = "minmax",
) -> nn.Module:
    """Calibration-based static INT8 quantization (fake-quant execution)."""
    work = model if inplace else copy.deepcopy(model)
    work.eval().to(device)
    selected = _selected_modules(work, layer_prefixes, keep_float_prefixes)
    if not selected:
        logger.warning("No modules matched the static quantization policy")
        return work

    samples = defaultdict(list)
    hooks = []

    def _make_hook(name):
        def hook(_module, inputs):
            samples[name].append(inputs[0].detach().flatten().cpu())
        return hook

    for name, module in selected:
        hooks.append(module.register_forward_pre_hook(_make_hook(name)))

    seen = 0
    with torch.no_grad():
        for images in _iter_calibration_batches(calib_loader):
            if num_batches is not None and seen >= num_batches:
                break
            work(images.to(device))
            seen += 1

    for hook in hooks:
        hook.remove()

    if seen == 0:
        raise ValueError("calib_loader produced no batches")

    for name, module in selected:
        values = torch.cat(samples[name])
        act_cfg = QuantConfig(symmetric=False, per_channel=False)
        act_observer = build_observer(observer, act_cfg)
        act_observer.observe(values)
        scale, zero_point = act_observer.calculate_qparams()
        scale = scale.to(module.weight.device)
        zero_point = zero_point.to(module.weight.device)

        (w_scale, w_zero_point), w_cfg = _weight_qparams(
            module.weight.detach(), per_channel=per_channel
        )
        module.weight.data.copy_(fake_quantize(
            module.weight.data, w_scale.to(module.weight.device),
            w_zero_point.to(module.weight.device),
            w_cfg.qmin, w_cfg.qmax,
        ))

        def _activation_hook(_module, inputs, s=scale, z=zero_point, c=act_cfg):
            x = inputs[0]
            return (fake_quantize(x, s, z, c.qmin, c.qmax),)

        module.register_forward_pre_hook(_activation_hook)
        module._flashdet_quantized = True
        logger.debug("Static-quantized %s", name)

    logger.info(
        "Static quantization applied to %d layers (observer=%s, per_channel=%s)",
        len(selected), observer, per_channel,
    )
    return work


def set_layer_qconfig(
    model: nn.Module,
    qconfig,
    layer_prefixes: Optional[Sequence[str]] = None,
    keep_float_prefixes: Sequence[str] = ("head",),
):
    """Assign qconfig only to leaves under the selected prefixes."""
    keep_float = tuple(keep_float_prefixes or ())
    for name, module in model.named_modules():
        if not name:
            continue
        selected = _matches(name, layer_prefixes) and not _matches(name, keep_float)
        if len(list(module.children())) == 0:
            module.qconfig = qconfig if selected else None
    return model


def quantize_static_ao(
    model: nn.Module,
    calib_loader: Iterable,
    backend: str = "fbgemm",
    layer_prefixes: Optional[Sequence[str]] = None,
    keep_float_prefixes: Sequence[str] = ("head",),
    num_batches: Optional[int] = None,
    inplace: bool = False,
    device: str = "cpu",
) -> nn.Module:
    """PyTorch eager-mode static quantization (advanced; requires stubs)."""
    import torch.ao.quantization as tq

    if backend not in ("fbgemm", "qnnpack"):
        raise ValueError("backend must be 'fbgemm' or 'qnnpack'")
    torch.backends.quantized.engine = backend

    work = model if inplace else copy.deepcopy(model)
    work.eval()
    set_layer_qconfig(
        work, tq.get_default_qconfig(backend),
        layer_prefixes, keep_float_prefixes,
    )
    work = tq.prepare(work, inplace=True)
    calibrate(work, calib_loader, num_batches=num_batches, device=device)
    return tq.convert(work, inplace=True)


def quantize_static_layers(
    model: nn.Module,
    calib_loader: Iterable,
    layers: Sequence[str] = ("backbone", "neck"),
    **kwargs,
) -> nn.Module:
    """Static quantization restricted to named layer groups."""
    return quantize_static(model, calib_loader, layer_prefixes=layers, **kwargs)
