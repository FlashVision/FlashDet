"""Structural reparameterization fusion for the ``rep`` block family.

Reuses the native ``fuse()`` implementations (collapses multi-branch
training-time blocks into single convolutions) and only counts what changed.
"""

from __future__ import annotations

import torch.nn as nn

from flashdet.fusing.base import module_op_counts


class RepFusionPass:
    """Call ``fuse()`` on every module that provides it.

    Covers the FlashDet native MultiScaleConv/FusedDWConv collapse and the
    Pico-Zoo backbone's ``fuse()`` (PicoBlock, RepBlock, StrideDown, ...).
    """

    name = "rep_fusion"

    def apply(self, model: nn.Module) -> dict:
        ops_before = module_op_counts(model)
        changed = 0
        for _, module in list(model.named_modules()):
            fuse = getattr(module, "fuse", None)
            if callable(fuse) and type(module).__name__ not in _SKIP:
                try:
                    result = fuse()
                    changed += 1 if result is not None else 0
                except Exception:
                    continue
        ops_after = module_op_counts(model)
        return {
            "name": self.name,
            "fused": changed,
            "ops_before": ops_before,
            "ops": ops_after,
        }


_SKIP = {"FlashDet", "FlashDetPico", "PicoZooBackbone", "PicoZooNeck",
         "FlashDetMicro", "FlashDetector"}
