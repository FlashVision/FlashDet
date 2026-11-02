"""State-dict-only save/load support for fused models.

Fusion rewrites module attribute paths (a folded ``conv -> bn -> act`` triple
becomes one ``FusedConvBnAct``), so a fused state dict no longer matches the
raw architecture.  Instead of pickling module objects — which the secure
``weights_only`` loader rejects — we save the fused tensors together with the
config and a small manifest, and replay the (structurally deterministic)
passes when loading.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

__all__ = [
    "PASS_REGISTRY",
    "apply_fusion_manifest",
    "passes_from_names",
    "save_fused_checkpoint",
]


def _pass_factories(example_input):
    """Map pass name -> zero-argument factory bound to ``example_input``."""
    from flashdet.fusing.kernel_fusing.fused_ops import ConvBnActFusionPass
    from flashdet.fusing.layer_fusing.conv_bn import (
        ConvBnFusionPass,
        LinearBnFusionPass,
    )
    from flashdet.fusing.layer_fusing.linear_pair import LinearPairFusionPass
    from flashdet.fusing.layer_fusing.rep_fusing import RepFusionPass

    return {
        "rep_fusion": RepFusionPass,
        "conv_bn_act": lambda: ConvBnActFusionPass(example_input=example_input),
        "conv_bn": lambda: ConvBnFusionPass(example_input=example_input),
        "linear_pair": lambda: LinearPairFusionPass(
            example_input=example_input),
        "linear_bn": lambda: LinearBnFusionPass(example_input=example_input),
    }


PASS_REGISTRY = (
    "rep_fusion",
    "conv_bn_act",
    "conv_bn",
    "linear_pair",
    "linear_bn",
)


def passes_from_names(names: list[str], example_input=None) -> list:
    """Instantiate fusion passes from their saved names, in order."""
    factories = _pass_factories(example_input)
    unknown = [n for n in names if n not in factories]
    if unknown:
        raise ValueError(f"Unknown fusion pass(es) in checkpoint: {unknown}")
    return [factories[name]() for name in names]


def apply_fusion_manifest(model: nn.Module, fusion: dict,
                          default_size: int = 320) -> nn.Module:
    """Rebuild the fused architecture described by a checkpoint manifest.

    The passes are deterministic given the architecture and probe shape, so
    replaying them recreates exactly the module paths the fused tensors were
    saved with.
    """
    from flashdet.fusing.base import run_fusions

    names = fusion.get("passes") or []
    if not names:
        return model
    size = int(fusion.get("input_size") or default_size)
    probe = torch.randn(1, 3, size, size)
    passes = passes_from_names(names, example_input=probe)
    fused, _ = run_fusions(model, passes, example_input=probe, verify=False)
    return fused


def save_fused_checkpoint(model: nn.Module, path: str, *, config: dict,
                          input_size: int, report=None) -> str:
    """Save a fused model as a plain, safely-loadable state dict."""
    state = {
        k.replace("_orig_mod.", "", 1): v
        for k, v in model.state_dict().items()
    }
    names = [p.get("name") for p in report.passes] if report is not None else []
    names = [n for n in names if n]
    unknown = [n for n in names if n not in PASS_REGISTRY]
    if unknown:
        # The manifest is replayed on load; unknown names would make the
        # checkpoint unloadable, so fail loudly at save time instead.
        raise ValueError(
            f"cannot save fused checkpoint with unknown pass(es): {unknown}; "
            f"known passes are {list(PASS_REGISTRY)}"
        )
    payload = {
        "model_state_dict": state,
        "config": dict(config or {}),
        "fusion": {
            "passes": [n for n in names if n],
            "input_size": int(input_size),
        },
    }
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    torch.save(payload, path)
    return path
