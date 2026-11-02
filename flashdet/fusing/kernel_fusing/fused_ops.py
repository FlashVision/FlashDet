"""Kernel fusion: whole op-groups become one module call.

The headline use case: deployment runs of FlashDet execute the pattern
``conv -> batchnorm -> activation`` everywhere.  In eager mode that's three
kernel launches and two intermediate tensors per pattern; fused it becomes one
module with a single small forward.  The BatchNorm is folded into the conv
weights (same math as :mod:`flashdet.fusing.layer_fusing.conv_bn`) and the
activation moves into the fused module.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from flashdet.fusing.graph_fusing.fx_utils import (
    EditContext,
    call_module_counts,
    call_target_counts,
    trace_inference,
)
from flashdet.fusing.layer_fusing.conv_bn import fold_bn_into_conv

_ACT_MODULE_TYPES = (nn.ReLU, nn.LeakyReLU, nn.SiLU, nn.Identity)
_ACT_FUNCTIONAL = (torch.nn.functional.relu, torch.nn.functional.leaky_relu,
                   torch.nn.functional.silu)


class FusedConvBnAct(nn.Module):
    """Single-module conv(+)folded-bn→activation; one call in a traced graph."""

    def __init__(self, conv: nn.Module, act: nn.Module | None = None,
                 act_kwargs: dict | None = None):
        super().__init__()
        self.conv = conv
        self.act = act
        self.act_kwargs = dict(act_kwargs or {})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.act is None:
            return x
        return self.act(x, **self.act_kwargs)


class ConvBnActFusionPass:
    """Replace conv→BN→act triples with a single fused module call.

    Falls back to plain conv→BN folding when no activation follows, so the
    pass can run over the whole model in one shot.
    """

    name = "conv_bn_act"

    def __init__(self, example_input: torch.Tensor | None = None):
        self._example_input = example_input

    def apply(self, model: nn.Module) -> dict:
        wrapper, gm = trace_inference(model, self._example_input)
        edits = EditContext(wrapper)
        fused_act = self._fold_triples(gm, edits)
        gm.graph.eliminate_dead_code()
        gm.recompile()
        return {"name": self.name, "fused": fused_act,
                "ops": call_module_counts(gm)}

    def _act_from_node(self, gm, node) -> tuple[nn.Module | None, dict] | None:
        if node.op != "call_module":
            return None
        module = gm.get_submodule(node.target)
        if isinstance(module, _ACT_MODULE_TYPES):
            return module, {}
        return None

    def _fold_triples(self, gm, edits: EditContext) -> int:
        count = 0
        targets = call_target_counts(gm)
        for node in list(gm.graph.nodes):
            if node.op != "call_module":
                continue
            bn = gm.get_submodule(node.target)
            if not isinstance(bn,
                              (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                continue
            conv_node = node.args[0]
            if conv_node.op != "call_module":
                continue
            conv = gm.get_submodule(conv_node.target)
            if not isinstance(conv, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                continue
            if (targets[conv_node.target] != 1
                    or targets[node.target] != 1):
                # Module reused at several call sites (e.g. one `self.act`
                # applied in two branches) — replacing the attribute would
                # also rewrite the other call sites.
                continue
            extra_conv_consumers = [
                u for u in conv_node.users if u is not node
            ]
            if extra_conv_consumers:
                continue

            bn_users = list(node.users)
            act_node = None
            act_module = None
            if len(bn_users) == 1:
                entry = self._act_from_node(gm, bn_users[0])
                # The activation can only be absorbed when its module is
                # not shared with another call site: absorbing it replaces
                # the attribute with Identity everywhere.
                if entry is not None and targets[bn_users[0].target] == 1:
                    act_node = bn_users[0]
                    act_module = entry[0]

            if act_node is not None:
                fused = FusedConvBnAct(
                    fold_bn_into_conv(conv, bn), act_module,
                )
                edits.replace(conv_node.target, fused)
                edits.replace(node.target, nn.Identity())
                edits.replace(act_node.target, nn.Identity())

                act_node.replace_all_uses_with(conv_node)
                gm.graph.erase_node(act_node)
                node.replace_all_uses_with(conv_node)
                gm.graph.erase_node(node)
                gm.recompile()
                count += 1
                continue

            # No absorbable activation — fold the numerically-safe conv→BN
            # part: every BN consumer is rewritten to read the folded conv
            # output, which is equal to the BN output.
            edits.replace(conv_node.target, fold_bn_into_conv(conv, bn))
            edits.replace(node.target, nn.Identity())
            node.replace_all_uses_with(conv_node)
            gm.graph.erase_node(node)
            gm.recompile()
            count += 1
        return count


def fused_residual_act(x: torch.Tensor, y: torch.Tensor,
                       activation: str = "leaky_relu", slope: float = 0.1,
                       inplace: bool = True) -> torch.Tensor:
    """Residual add + activation in one functional call (fewer allocs)."""
    total = x + y
    if activation == "leaky_relu":
        return torch.nn.functional.leaky_relu(total, slope, inplace=inplace)
    if activation == "relu":
        return torch.nn.functional.relu(total, inplace=inplace)
    if activation == "silu":
        return torch.nn.functional.silu(total, inplace=inplace)
    return total


class FusedDWPointwise(nn.Module):
    """Depthwise → pointwise pair as one module (single attribute hop)."""

    def __init__(self, dw: nn.Module, pw: nn.Module):
        super().__init__()
        self.dw = dw
        self.pw = pw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))
