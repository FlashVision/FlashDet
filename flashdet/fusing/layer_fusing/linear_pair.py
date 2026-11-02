"""Layer fusion: merge back-to-back Linear projections.

Two stacked Linear layers without a non-linearity between them compute the
same thing as a single Linear:

    y2 = W2 (W1 x + b1) + b2 = (W2 W1) x + (W2 b1 + b2)

Common around small head MLPs where dropout or activation runs at inference.
"""

from __future__ import annotations

import torch.nn as nn

from flashdet.fusing.graph_fusing.fx_utils import (
    EditContext,
    call_module_counts,
    call_target_counts,
    trace_inference,
)

_ACT_TYPES = (
    nn.ReLU, nn.LeakyReLU, nn.SiLU, nn.GELU, nn.Hardswish, nn.ELU, nn.Identity,
)


def merge_linears(first: nn.Linear, second: nn.Linear) -> nn.Linear:
    """Return a single Linear equivalent to ``second(first(x))``."""
    merged = nn.Linear(
        first.in_features, second.out_features,
        bias=second.bias is not None,
        device=second.weight.device, dtype=second.weight.dtype,
    )
    weight = second.weight.detach() @ first.weight.detach()
    if second.bias is not None:
        bias = second.bias.detach().clone()
        if first.bias is not None:
            bias += second.weight.detach() @ first.bias.detach()
    elif first.bias is not None:
        bias = second.weight.detach() @ first.bias.detach()
    else:
        bias = None
    merged.weight.data.copy_(weight)
    if bias is not None:
        if merged.bias is None:
            merged.bias = nn.Parameter(bias, requires_grad=False)
        else:
            merged.bias.data.copy_(bias)
    return merged


class LinearPairFusionPass:
    """Merge adjacent Linear→Linear nodes with nothing between them."""

    name = "linear_pair"

    def __init__(self, example_input=None):
        self._example_input = example_input

    def apply(self, model: nn.Module) -> dict:
        wrapper, gm = trace_inference(model, self._example_input)
        edits = EditContext(wrapper)
        count = self._merge_pairs(gm, edits)
        gm.graph.eliminate_dead_code()
        gm.recompile()
        return {"name": self.name, "fused": count,
                "ops": call_module_counts(gm)}

    @staticmethod
    def _through_identity(gm, node):
        """Skip folded-BatchNorm placeholders (Identity) between the pair."""
        while (getattr(node, "op", None) == "call_module"
               and isinstance(gm.get_submodule(node.target), nn.Identity)):
            node = node.args[0]
        return node

    def _merge_pairs(self, gm, edits: EditContext) -> int:
        count = 0
        targets = call_target_counts(gm)
        for node in list(gm.graph.nodes):
            if node.op != "call_module":
                continue
            second = gm.get_submodule(node.target)
            if not isinstance(second, nn.Linear):
                continue
            first_node = self._through_identity(gm, node.args[0])
            if getattr(first_node, "op", None) != "call_module":
                continue
            first = gm.get_submodule(first_node.target)
            if not isinstance(first, nn.Linear):
                continue
            if (targets[first_node.target] != 1
                    or targets[node.target] != 1):
                continue
            merged = merge_linears(first, second)
            edits.replace(first_node.target, merged)
            edits.replace(node.target, nn.Identity())
            node.replace_all_uses_with(first_node)
            gm.graph.erase_node(node)
            gm.recompile()
            count += 1
        return count
