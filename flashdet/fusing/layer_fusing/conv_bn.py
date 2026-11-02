"""Layer fusion: fold BatchNorm into its convolution or Linear weight.

Math (eval mode, per output channel ``c``):

    s_c   = sqrt(running_var_c + eps)
    gamma = bn.weight, beta = bn.bias, mu = bn.running_mean

    conv_out = W x + b
    y        = gamma (conv_out - mu) / s + beta
             = (W * gamma/s) x + (gamma (b - mu)/s + beta)

Depthwise/grouped convolutions work because scaling applies per output
channel, which is always dim 0 of the weight tensor.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from flashdet.fusing.graph_fusing.fx_utils import (
    EditContext,
    call_module_counts,
    module_call_sites,
    trace_inference,
    used_once,
)

_CONV_TYPES = (nn.Conv1d, nn.Conv2d, nn.Conv3d)
_BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def fold_bn_into_conv(conv: nn.Module, bn: nn.Module) -> nn.Module:
    """Fold ``bn`` into ``conv`` numerically and return the same module."""
    if conv.groups > 1 and conv.in_channels != conv.groups:
        raise NotImplementedError(
            "Grouped convolutions are only folded when groups == in_channels "
            "(depthwise) or groups == 1"
        )
    s = torch.sqrt(bn.running_var + bn.eps)
    gamma = bn.weight.detach()
    beta = bn.bias.detach()
    shape = [conv.weight.shape[0]] + [1] * (conv.weight.dim() - 1)
    scale = (gamma / s).reshape(shape)

    conv.weight.data.mul_(scale)
    if conv.bias is None:
        conv.bias = nn.Parameter(
            torch.zeros(conv.weight.shape[0], device=conv.weight.device,
                        dtype=conv.weight.dtype), requires_grad=False)
        conv.bias.data.copy_(
            beta - gamma * bn.running_mean / s)
    else:
        conv.bias.data.copy_(
            (conv.bias.detach() - bn.running_mean) * gamma / s + beta)
    return conv


def fold_bn_into_linear(linear: nn.Module, bn: nn.Module) -> nn.Module:
    s = torch.sqrt(bn.running_var + bn.eps)
    gamma = bn.weight.detach()
    beta = bn.bias.detach()
    if linear.bias is None:
        linear.bias = nn.Parameter(
            torch.zeros(linear.weight.shape[0], device=linear.weight.device,
                        dtype=linear.weight.dtype), requires_grad=False)
        linear.bias.data.copy_(beta - gamma * bn.running_mean / s)
    else:
        linear.bias.data.copy_(
            (linear.bias.detach() - bn.running_mean) * gamma / s + beta)
    linear.weight.data.mul_((gamma / s).unsqueeze(1))
    return linear


class ConvBnFusionPass:
    """Fold every conv→BatchNorm pair in the traced inference graph.

    A pair is only folded when the convolution output feeds *exclusively*
    into the normalization layer, so residual taps or shared conv outputs
    are left untouched.
    """

    name = "conv_bn"

    def __init__(self, example_input: torch.Tensor | None = None):
        self._example_input = example_input

    def apply(self, model: nn.Module) -> dict:
        wrapper, gm = trace_inference(model, self._example_input)
        edits = EditContext(wrapper)
        fused = self._fold_pairs(gm, edits)
        gm.graph.eliminate_dead_code()
        gm.recompile()
        return {"name": self.name, "fused": fused,
                "ops": call_module_counts(gm)}

    def _fold_pairs(self, gm, edits: EditContext) -> int:
        count = 0
        sites = module_call_sites(gm)
        for node in list(gm.graph.nodes):
            if node.op != "call_module":
                continue
            bn = gm.get_submodule(node.target)
            if not isinstance(bn, _BN_TYPES):
                continue
            conv_node = node.args[0]
            if conv_node.op != "call_module":
                continue
            conv = gm.get_submodule(conv_node.target)
            if not isinstance(conv, _CONV_TYPES):
                continue
            if not used_once(sites, conv) or not used_once(sites, bn):
                # The module is shared by several call sites in the forward
                # (e.g. one `self.act` applied in two branches).  Folding
                # mutates the module and the graph together, so every call
                # site must correspond to exactly one node for this to stay
                # numerically equivalent.  Shared modules are left alone.
                continue
            extra_consumers = [u for u in conv_node.users if u is not node]
            if extra_consumers:
                # Something besides the BN taps the conv output directly —
                # folding would change what they see.  Leave this pair as is.
                continue
            edits.replace(conv_node.target, fold_bn_into_conv(conv, bn))
            edits.replace(node.target, nn.Identity())
            node.replace_all_uses_with(conv_node)
            gm.graph.erase_node(node)
            gm.recompile()
            count += 1
        return count


class LinearBnFusionPass:
    """Fold Linear→BatchNorm1d pairs (usually head projections)."""

    name = "linear_bn"

    def __init__(self, example_input: torch.Tensor | None = None):
        self._example_input = example_input

    def apply(self, model: nn.Module) -> dict:
        wrapper, gm = trace_inference(model, self._example_input)
        edits = EditContext(wrapper)
        count = self._fold_pairs(gm, edits)
        gm.graph.eliminate_dead_code()
        gm.recompile()
        return {"name": self.name, "fused": count,
                "ops": call_module_counts(gm)}

    def _fold_pairs(self, gm, edits: EditContext) -> int:
        count = 0
        sites = module_call_sites(gm)
        for node in list(gm.graph.nodes):
            if node.op != "call_module":
                continue
            bn = gm.get_submodule(node.target)
            if not isinstance(bn, _BN_TYPES):
                continue
            linear_node = node.args[0]
            if linear_node.op != "call_module":
                continue
            linear = gm.get_submodule(linear_node.target)
            if not isinstance(linear, nn.Linear):
                continue
            if not used_once(sites, linear) or not used_once(sites, bn):
                continue
            extra_consumers = [u for u in linear_node.users if u is not node]
            if extra_consumers:
                continue
            edits.replace(linear_node.target, fold_bn_into_linear(linear, bn))
            edits.replace(node.target, nn.Identity())
            node.replace_all_uses_with(linear_node)
            gm.graph.erase_node(node)
            gm.recompile()
            count += 1
        return count
