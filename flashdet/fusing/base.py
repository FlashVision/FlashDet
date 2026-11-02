"""Fusion framework: shared pass/report plumbing.

A fusion pass inspects a model, removes redundant work (a BatchNorm folded
into its convolution, a pair of stacked Linear layers, multi-branch
reparameterizable blocks) and reports what changed.  All passes run on a
deep copy so the original model is never mutated.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
import torch.nn as nn

__all__ = [
    "FusionPass",
    "FusionReport",
    "param_count",
    "module_op_counts",
    "run_fusions",
]


class FusionPass(ABC):
    """One named, composable model optimization."""

    name: str = "pass"

    @abstractmethod
    def apply(self, model: nn.Module) -> dict:
        """Apply the pass in place to ``model`` and return change stats."""

    def validate(self, model: nn.Module) -> None:
        """Optional hook run before the pass is applied."""


@dataclass
class FusionReport:
    """Aggregated result of running several fusion passes."""

    passes: list[dict] = field(default_factory=list)
    params_before: int = 0
    params_after: int = 0
    ops_before: dict = field(default_factory=dict)
    ops_after: dict = field(default_factory=dict)
    parity_ok: bool | None = None
    parity_max_abs_diff: float | None = None

    @property
    def total_changes(self) -> int:
        return sum(int(p.get("fused", p.get("changed", 0)))
                   for p in self.passes)

    def to_dict(self) -> dict:
        return {
            "passes": self.passes,
            "params_before": self.params_before,
            "params_after": self.params_after,
            "params_delta": self.params_after - self.params_before,
            "ops_before": self.ops_before,
            "ops_after": self.ops_after,
            "total_changes": self.total_changes,
            "parity_ok": self.parity_ok,
            "parity_max_abs_diff": self.parity_max_abs_diff,
        }

    def __str__(self) -> str:
        lines = []
        for p in self.passes:
            fused = int(p.get("fused", p.get("changed", 0)))
            if fused:
                lines.append(f"  {p['name']}: {fused} fused")
        if self.parity_ok is not None:
            state = "OK" if self.parity_ok else "MISMATCH"
            lines.append(f"  parity: {state} "
                         f"(max|Δ|={self.parity_max_abs_diff:.6g})")
        delta = self.params_after - self.params_before
        lines.append(f"  params: {self.params_before:,} -> "
                     f"{self.params_after:,} ({delta:+,})")
        return "\n".join(lines)


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def module_op_counts(model: nn.Module) -> dict:
    """Count leaf module types that do real computation."""
    counts: dict[str, int] = {}
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            counts["conv1d"] = counts.get("conv1d", 0) + 1
        elif isinstance(m, nn.Conv2d):
            counts["conv2d"] = counts.get("conv2d", 0) + 1
        elif isinstance(m, nn.Conv3d):
            counts["conv3d"] = counts.get("conv3d", 0) + 1
        elif isinstance(m, nn.Linear):
            counts["linear"] = counts.get("linear", 0) + 1
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            counts["batchnorm"] = counts.get("batchnorm", 0) + 1
    return counts


def run_fusions(model: nn.Module, passes: list[FusionPass],
                example_input: torch.Tensor | None = None,
                verify: bool = True) -> tuple[nn.Module, FusionReport]:
    """Apply passes to a deep copy of ``model`` and verify numeric parity.

    Returns the fused model and a :class:`FusionReport`.  When ``verify`` is
    true (and ``example_input`` is given) the fused forward is compared
    against the original within a small tolerance; the fused model is kept
    either way, but the parity result is reported honestly.
    """
    # Keep the caller's model untouched: the reference and the fused working
    # copy are both owned by this function.
    original = copy.deepcopy(model).eval()
    fused = copy.deepcopy(model).eval()

    report = FusionReport(
        params_before=param_count(original),
        ops_before=module_op_counts(original),
    )

    for fuse_pass in passes:
        stats = fuse_pass.apply(fused)
        stats.setdefault("name", getattr(fuse_pass, "name",
                                         type(fuse_pass).__name__))
        report.passes.append(stats)
    fused.eval()

    if example_input is not None and verify:
        with torch.no_grad():
            before = original(example_input)
            after = fused(example_input)
        if isinstance(before, dict):
            keys = sorted(before)
            diff = max(float((before[k] - after[k]).abs().max())
                       for k in keys if torch.is_tensor(before[k]))
        else:
            diff = float((before - after).abs().max())
        report.parity_max_abs_diff = diff
        report.parity_ok = diff < 5e-3
    report.params_after = param_count(fused)
    report.ops_after = module_op_counts(fused)
    return fused, report
