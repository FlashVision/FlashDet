"""Tracing utilities shared by graph-level fusion passes."""

from __future__ import annotations

import torch
import torch.nn as nn


class _DetectorWrapper(nn.Module):
    """Single-tensor view of a detector for tracing."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        return torch.cat([out["o2o_cls"], out["o2o_reg"]], dim=-1)


class _TensorWrapper(nn.Module):
    """Pass-through wrapper for models that already return tensors."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def guess_example_input(model: nn.Module,
                        example_input: torch.Tensor | None = None):
    """Find a reasonable probe input for a model when the caller gave none."""
    if example_input is not None:
        return example_input

    for module in model.modules():
        if isinstance(module, nn.Conv2d) and module.in_channels > 0:
            return torch.randn(1, module.in_channels, 96, 96)
        if isinstance(module, nn.Linear) and module.in_features > 0:
            return torch.randn(1, module.in_features)
    raise ValueError("Could not infer an example input for tracing")


def _build_wrapper(model: nn.Module, example_input: torch.Tensor) -> nn.Module:
    """Probe the model's output form eagerly, then pick a trace-safe wrapper."""
    with torch.no_grad():
        out = model(example_input)
    if isinstance(out, dict) and "o2o_cls" in out and "o2o_reg" in out:
        return _DetectorWrapper(model)
    if torch.is_tensor(out):
        return _TensorWrapper(model)
    raise ValueError(
        "Model must return a tensor or an o2o_cls/o2o_reg dict for tracing; "
        f"got {type(out)!r}"
    )


def trace_inference(model: nn.Module, example_input: torch.Tensor | None = None):
    """Trace ``model``'s inference path as a torch.fx GraphModule.

    Shares module storage with ``model``, so edits apply to the live model —
    callers should own a deepcopy (``run_fusions`` does).
    """
    import torch.fx as fx

    probe = guess_example_input(model, example_input)
    wrapper = _build_wrapper(model, probe)
    wrapper.eval()
    with torch.no_grad():
        traced = fx.symbolic_trace(wrapper)
    graph_module = fx.GraphModule(wrapper, traced.graph)
    graph_module.eval()
    return wrapper, graph_module


def module_for_target(gm: nn.Module, target: str) -> nn.Module:
    return gm.get_submodule(target)


class EditContext:
    """Attribute swaps applied to the *real* model, not FX's re-parent copy.

    A traced GraphModule re-parents top-level child modules into its own
    copy; editing through ``gm.get_submodule`` would silently mutate the copy
    instead of the user's model. This context performs ``setattr`` through
    the original wrapper, which shares leaf modules with the model.
    """

    def __init__(self, wrapper: nn.Module):
        self.wrapper = wrapper

    def replace(self, target: str, module: nn.Module) -> None:
        """Point the attribute at ``target`` to ``module`` on the live model."""
        parent_path, _, attr = target.rpartition(".")
        if parent_path:
            parent = self.wrapper.get_submodule(parent_path)
        else:
            parent = self.wrapper
        setattr(parent, attr, module)


def call_target_counts(gm) -> dict:
    """Map call_module target path -> number of call sites in ``gm``."""
    counts: dict[str, int] = {}
    for node in gm.graph.nodes:
        if node.op == "call_module":
            counts[node.target] = counts.get(node.target, 0) + 1
    return counts


def call_module_counts(gm) -> dict:
    """Number of call_module nodes per leaf type (post-graph view)."""

    counts: dict[str, int] = {}
    for node in gm.graph.nodes:
        if node.op != "call_module":
            continue
        module = gm.get_submodule(node.target)
        counts[type(module).__name__] = counts.get(
            type(module).__name__, 0) + 1
    return counts
