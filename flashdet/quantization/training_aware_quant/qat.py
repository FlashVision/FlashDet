"""Quantization-Aware Training (QAT) prepare/convert helpers."""

from __future__ import annotations

import copy
import logging
from typing import Optional, Sequence

import torch
import torch.nn as nn

from flashdet.quantization.static_quantization.ptq import set_layer_qconfig

logger = logging.getLogger(__name__)


def prepare_qat(
    model: nn.Module,
    backend: str = "fbgemm",
    layer_prefixes: Optional[Sequence[str]] = None,
    keep_float_prefixes: Sequence[str] = ("head",),
    inplace: bool = False,
) -> nn.Module:
    """Insert fake-quant observers and switch the model to train mode."""
    import torch.ao.quantization as tq

    if backend not in ("fbgemm", "qnnpack"):
        raise ValueError("backend must be 'fbgemm' or 'qnnpack'")
    torch.backends.quantized.engine = backend

    work = model if inplace else copy.deepcopy(model)
    work.train()
    set_layer_qconfig(
        work,
        tq.get_default_qat_qconfig(backend),
        layer_prefixes=layer_prefixes,
        keep_float_prefixes=keep_float_prefixes,
    )
    return tq.prepare_qat(work, inplace=True)


def convert_qat(
    model: nn.Module,
    example_input: torch.Tensor = None,
    inplace: bool = False,
) -> nn.Module:
    """Convert a trained QAT model to INT8 inference form.

    Eager conversion requires QuantStub/DeQuantStub boundaries. FlashDet's
    custom multi-output head does not have them, so without ``example_input``
    this returns the trained fake-quant model (simulated INT8). Pass a sample
    input to attempt eager conversion and automatically fall back if the
    converted graph cannot run.
    """
    import torch.ao.quantization as tq

    work = model if inplace else copy.deepcopy(model)
    work.eval()
    if example_input is None:
        logger.info(
            "No example_input supplied; returning trained fake-quant QAT model. "
            "Export to ONNX/TensorRT for hardware INT8."
        )
        return work

    try:
        converted = tq.convert(work, inplace=True)
        with torch.no_grad():
            converted(example_input)
        return converted
    except Exception as exc:
        logger.warning(
            "Eager QAT conversion failed (%s); returning the trained "
            "fake-quant model. Export to ONNX/TensorRT for hardware INT8.",
            exc,
        )
        return work


def qat_finetune(
    model: nn.Module,
    train_step,
    epochs: int = 1,
    backend: str = "fbgemm",
) -> nn.Module:
    """Small convenience wrapper: prepare, run ``train_step(model)``, convert.

    ``train_step`` must return a scalar loss and call ``loss.backward()`` /
    optimizer steps itself, or simply perform one training iteration.
    """
    prepared = prepare_qat(model, backend=backend)
    for _ in range(max(0, epochs)):
        train_step(prepared)
    return convert_qat(prepared)
