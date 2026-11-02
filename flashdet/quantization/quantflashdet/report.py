"""Quantization reports for FlashDet layer groups."""

from __future__ import annotations

from collections import Counter
from typing import Dict

import torch.nn as nn

from .layer_config import QuantLayerPolicy


def _group_for(name: str) -> str:
    root = name.split(".", 1)[0]
    return root if root in ("backbone", "neck", "head") else "other"


def quantization_report(
    model: nn.Module,
    policy: QuantLayerPolicy = None,
) -> Dict:
    """Summarize module types and quantized layers per FlashDet group."""
    counts: Dict[str, Counter] = {}
    for name, module in model.named_modules():
        if not name:
            continue
        group = _group_for(name)
        counts.setdefault(group, Counter())[type(module).__name__] += 1

    quantized_markers = (
        "Quantized", "Int8", "Dynamic", "FakeQuant", "QAT",
    )
    quantized = {group: 0 for group in counts}
    for name, module in model.named_modules():
        if not name:
            continue
        group = _group_for(name)
        flagged = bool(getattr(module, "_flashdet_quantized", False))
        typed = any(marker in type(module).__name__ for marker in quantized_markers)
        if flagged or typed:
            quantized[group] += 1

    return {
        "policy": policy.as_dict() if policy is not None else None,
        "modules_by_group": {
            group: dict(counter) for group, counter in counts.items()
        },
        "quantized_modules": quantized,
    }
