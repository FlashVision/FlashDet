"""FlashDet-specific layer-aware quantization."""

from __future__ import annotations

import copy
from typing import Iterable, Optional

import torch
import torch.nn as nn

from flashdet.quantization.dynamic_quantization import (
    quantize_dynamic,
    quantize_weight_only,
)
from flashdet.quantization.static_quantization import quantize_static
from flashdet.quantization.training_aware_quant import convert_qat, prepare_qat

from .layer_config import DEFAULT_POLICY, QuantLayerPolicy
from .report import quantization_report


class QuantFlashDet:
    """Quantize FlashDet backbone/neck/head with per-group policies.

    Example::

        from flashdet.quantization.quantflashdet import QuantFlashDet

        quantizer = QuantFlashDet(model)
        qmodel = quantizer.quantize("static", calib_loader=loader)
        print(quantizer.report(qmodel))
    """

    def __init__(
        self,
        model: nn.Module,
        policy: Optional[QuantLayerPolicy] = None,
        backend: str = "fbgemm",
    ):
        self.model = model
        self.policy = policy or DEFAULT_POLICY
        self.backend = backend
        self.stage = "float"

    def quantize(
        self,
        method: Optional[str] = None,
        calib_loader: Optional[Iterable] = None,
        num_batches: Optional[int] = None,
        inplace: bool = False,
    ) -> nn.Module:
        """Apply the policy (or a single override method) to the model.

        For ``method="qat"`` this returns a prepared model; call
        :meth:`convert` after QAT fine-tuning.
        """
        policy = self.policy
        if method is not None:
            method = method.lower()
            if method not in ("none", "static", "dynamic", "weight_only", "qat"):
                raise ValueError(f"Unknown quantization method {method!r}")
            policy = QuantLayerPolicy(
                backbone=method, neck=method, head=method
            )

        work = self.model if inplace else copy.deepcopy(self.model)
        work.eval()

        static_groups = policy.groups_for("static")
        dynamic_groups = policy.groups_for("dynamic")
        weight_only_groups = policy.groups_for("weight_only")
        qat_groups = policy.groups_for("qat")

        if qat_groups:
            work = prepare_qat(
                work,
                backend=self.backend,
                layer_prefixes=qat_groups,
                keep_float_prefixes=policy.keep_float(),
                inplace=True,
            )
            self.stage = "qat_prepared"
            return work

        if static_groups:
            if calib_loader is None:
                raise ValueError("calib_loader is required for static quantization")
            # Keep every non-static group in float during prepare/convert.
            keep_float = tuple(
                g for g in ("backbone", "neck", "head") if g not in static_groups
            )
            work = quantize_static(
                work,
                calib_loader=calib_loader,
                backend=self.backend,
                layer_prefixes=static_groups,
                keep_float_prefixes=keep_float,
                num_batches=num_batches,
                inplace=True,
            )

        if dynamic_groups:
            work = quantize_dynamic(
                work, layer_prefixes=dynamic_groups, inplace=True,
            )
        if weight_only_groups:
            work = quantize_weight_only(
                work, layer_prefixes=weight_only_groups, inplace=True,
            )

        self.stage = "quantized"
        return work

    def convert(
        self,
        model: Optional[nn.Module] = None,
        example_input: Optional[torch.Tensor] = None,
    ) -> nn.Module:
        """Convert a QAT-prepared model to INT8 inference form."""
        if self.stage != "qat_prepared":
            raise RuntimeError("convert() is only valid after QAT preparation")
        result = convert_qat(model or self.model, example_input=example_input)
        self.stage = "quantized"
        return result

    def report(self, model: Optional[nn.Module] = None) -> dict:
        return quantization_report(model or self.model, policy=self.policy)
