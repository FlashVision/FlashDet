"""FLOPs/MACs counter — measure computational complexity of FlashDet models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class FLOPsCounter:
    """Count FLOPs (floating-point operations) and MACs for a PyTorch model.

    Uses hook-based forward pass analysis to compute per-layer and total
    FLOPs without requiring external dependencies.

    Parameters
    ----------
    model : nn.Module | None
        PyTorch model instance. Provide either this or ``model_path``.
    model_path : str | Path | None
        Path to a ``.pth`` checkpoint. Loaded if ``model`` is None.
    input_size : int | tuple[int, int]
        Network input resolution.
    device : str
        Computation device.
    """

    def __init__(
        self,
        model=None,
        model_path: Optional[Union[str, Path]] = None,
        input_size: Union[int, Tuple[int, int]] = 320,
        device: str = "cpu",
    ):
        if model is None and model_path is None:
            raise ValueError("Provide either model or model_path")

        self._model = model
        self._model_path = Path(model_path) if model_path else None
        self.input_size = (input_size, input_size) if isinstance(input_size, int) else tuple(input_size)
        self.device = device

    def count(self) -> Dict[str, Any]:
        """Count total and per-layer FLOPs/MACs.

        Returns
        -------
        dict
            ``{"total_flops", "total_macs", "total_params",
              "flops_readable", "macs_readable", "params_readable",
              "per_layer": [...]}``.
        """
        import torch
        import torch.nn as nn

        model = self._get_model()
        model.eval()

        layer_stats: List[Dict[str, Any]] = []
        hooks = []

        def _hook_fn(name: str, module: nn.Module):
            def hook(mod, inp, out):
                flops = self._compute_layer_flops(mod, inp, out)
                params = sum(p.numel() for p in mod.parameters())
                out_shape = tuple(out.shape) if isinstance(out, torch.Tensor) else None
                layer_stats.append({
                    "name": name,
                    "type": type(mod).__name__,
                    "flops": flops,
                    "macs": flops // 2,
                    "params": params,
                    "output_shape": out_shape,
                })
            return hook

        for name, module in model.named_modules():
            if len(list(module.children())) > 0:
                continue
            hooks.append(module.register_forward_hook(_hook_fn(name, module)))

        dummy = torch.randn(1, 3, *self.input_size, device=self.device)
        with torch.no_grad():
            model(dummy)

        for h in hooks:
            h.remove()

        total_flops = sum(l["flops"] for l in layer_stats)
        total_macs = total_flops // 2
        total_params = sum(p.numel() for p in model.parameters())

        for layer in layer_stats:
            layer["flops_pct"] = round(layer["flops"] / max(total_flops, 1) * 100, 2)

        return {
            "total_flops": total_flops,
            "total_macs": total_macs,
            "total_params": total_params,
            "flops_readable": self._human_readable(total_flops),
            "macs_readable": self._human_readable(total_macs),
            "params_readable": self._human_readable(total_params),
            "per_layer": layer_stats,
        }

    def summary(self) -> str:
        """Return a human-readable FLOPs summary table."""
        stats = self.count()
        lines = [
            f"{'Layer':<45} {'Type':<16} {'FLOPs':>12} {'%':>6} {'Params':>10}",
            "-" * 95,
        ]
        for layer in stats["per_layer"]:
            if layer["flops"] > 0:
                lines.append(
                    f"{layer['name']:<45} {layer['type']:<16} "
                    f"{self._human_readable(layer['flops']):>12} "
                    f"{layer['flops_pct']:>5.1f}% "
                    f"{layer['params']:>10,}"
                )
        lines.append("-" * 95)
        lines.append(
            f"{'TOTAL':<45} {'':<16} "
            f"{stats['flops_readable']:>12} {'100.0%':>6} "
            f"{stats['total_params']:>10,}"
        )
        lines.append("")
        lines.append(f"FLOPs: {stats['flops_readable']}  |  "
                     f"MACs: {stats['macs_readable']}  |  "
                     f"Params: {stats['params_readable']}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Per-layer FLOPs estimation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_layer_flops(module, inputs, output) -> int:
        import torch.nn as nn

        if isinstance(module, (nn.Conv2d, nn.Conv1d)):
            return FLOPsCounter._conv_flops(module, output)
        elif isinstance(module, nn.Linear):
            return FLOPsCounter._linear_flops(module, output)
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm, nn.LayerNorm)):
            return FLOPsCounter._norm_flops(module, output)
        elif isinstance(module, (nn.ReLU, nn.ReLU6, nn.SiLU, nn.GELU, nn.LeakyReLU, nn.Sigmoid, nn.Hardswish)):
            return FLOPsCounter._activation_flops(output)
        elif isinstance(module, (nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
            return FLOPsCounter._pool_flops(module, output)
        elif isinstance(module, nn.ConvTranspose2d):
            return FLOPsCounter._convtranspose_flops(module, output)
        return 0

    @staticmethod
    def _conv_flops(module, output) -> int:
        import torch
        batch = output.shape[0]
        out_channels = output.shape[1]
        spatial = int(np.prod(output.shape[2:]))
        kernel_ops = int(np.prod(module.kernel_size)) * (module.in_channels // module.groups)
        flops = batch * out_channels * spatial * kernel_ops * 2
        if module.bias is not None:
            flops += batch * out_channels * spatial
        return flops

    @staticmethod
    def _linear_flops(module, output) -> int:
        batch_size = int(np.prod(output.shape[:-1]))
        flops = batch_size * module.in_features * module.out_features * 2
        if module.bias is not None:
            flops += batch_size * module.out_features
        return flops

    @staticmethod
    def _norm_flops(module, output) -> int:
        import torch
        numel = output.numel()
        return numel * 4  # mean, var, normalize, affine

    @staticmethod
    def _activation_flops(output) -> int:
        return output.numel()

    @staticmethod
    def _pool_flops(module, output) -> int:
        return output.numel()

    @staticmethod
    def _convtranspose_flops(module, output) -> int:
        import torch
        batch = output.shape[0]
        out_channels = output.shape[1]
        spatial = int(np.prod(output.shape[2:]))
        kernel_ops = int(np.prod(module.kernel_size)) * (module.in_channels // module.groups)
        flops = batch * out_channels * spatial * kernel_ops * 2
        if module.bias is not None:
            flops += batch * out_channels * spatial
        return flops

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _get_model(self):
        if self._model is not None:
            return self._model
        import torch
        self._model = torch.load(str(self._model_path), map_location=self.device, weights_only=False)
        if hasattr(self._model, "eval"):
            self._model.eval()
        return self._model

    @staticmethod
    def _human_readable(num: int) -> str:
        if num >= 1e12:
            return f"{num / 1e12:.2f}T"
        if num >= 1e9:
            return f"{num / 1e9:.2f}G"
        if num >= 1e6:
            return f"{num / 1e6:.2f}M"
        if num >= 1e3:
            return f"{num / 1e3:.2f}K"
        return str(num)
