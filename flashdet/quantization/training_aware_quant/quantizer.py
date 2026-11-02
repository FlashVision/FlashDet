"""Class-based QAT interface."""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

from .qat import convert_qat, prepare_qat


class QATQuantizer:
    """Prepare a model for QAT and convert it after fine-tuning."""

    def __init__(self, model: nn.Module, backend: str = "fbgemm"):
        self.model = model
        self.backend = backend
        self.stage = "float"

    def prepare(
        self,
        layer_prefixes: Optional[Sequence[str]] = None,
        keep_float_prefixes: Sequence[str] = ("head",),
    ) -> nn.Module:
        self.model = prepare_qat(
            self.model,
            backend=self.backend,
            layer_prefixes=layer_prefixes,
            keep_float_prefixes=keep_float_prefixes,
        )
        self.stage = "qat_prepared"
        return self.model

    def convert(self, example_input: torch.Tensor = None) -> nn.Module:
        if self.stage != "qat_prepared":
            raise RuntimeError("prepare() must be called before convert()")
        self.model = convert_qat(self.model, example_input=example_input)
        self.stage = "quantized"
        return self.model


def qat_train(
    model: nn.Module,
    train_step,
    epochs: int = 1,
    layer_prefixes: Optional[Sequence[str]] = None,
    example_input: torch.Tensor = None,
) -> nn.Module:
    """Prepare, fine-tune and convert a model for QAT."""
    quantizer = QATQuantizer(model)
    quantizer.prepare(layer_prefixes=layer_prefixes)
    for _ in range(max(0, epochs)):
        train_step(quantizer.model)
    return quantizer.convert(example_input=example_input)
