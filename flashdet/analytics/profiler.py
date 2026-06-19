"""Profiler — layer-wise latency and memory analysis for FlashDet models."""

from __future__ import annotations

import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


class Profiler:
    """Profile a FlashDet PyTorch model layer-by-layer.

    Hooks into every ``nn.Module`` child to record per-layer forward time
    and (optionally) memory consumption.

    Parameters
    ----------
    model_path : str | Path
        Path to a FlashDet ``.pth`` checkpoint.
    device : str
        ``"cuda"`` or ``"cpu"``.
    input_size : int | tuple[int, int]
        Network input resolution.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        device: str = "cuda",
        input_size: Union[int, tuple] = 320,
    ):
        self.model_path = Path(model_path)
        self.device = device
        self.input_size = (input_size, input_size) if isinstance(input_size, int) else tuple(input_size)

        self._model: Optional[Any] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, warmup: int = 5, iterations: int = 20) -> List[Dict[str, Any]]:
        """Profile the model and return per-layer statistics.

        Returns
        -------
        list[dict]
            Each dict has keys:
            ``{"name", "type", "time_ms", "time_pct", "params", "output_shape"}``.
        """
        import torch

        model = self._load_model()
        dummy = torch.randn(1, 3, *self.input_size, device=self.device)

        timings: OrderedDict[str, List[float]] = OrderedDict()
        shapes: Dict[str, tuple] = {}
        hooks = []

        def _make_hook(name: str):
            def _hook(module, inp, out):
                if self.device == "cuda":
                    torch.cuda.synchronize()
                timings[name].append(time.perf_counter())
                if isinstance(out, torch.Tensor):
                    shapes[name] = tuple(out.shape)
                elif isinstance(out, (tuple, list)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
                    shapes[name] = tuple(out[0].shape)
            return _hook

        for name, module in model.named_modules():
            if len(list(module.children())) > 0:
                continue  # only leaf modules
            timings[name] = []
            hooks.append(module.register_forward_hook(_make_hook(name)))

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                model(dummy)

        # Timed runs — record wall-clock per forward hook invocation
        layer_times: OrderedDict[str, List[float]] = OrderedDict()
        for name in timings:
            layer_times[name] = []

        for _ in range(iterations):
            for name in timings:
                timings[name].clear()

            if self.device == "cuda":
                torch.cuda.synchronize()
            with torch.no_grad():
                model(dummy)

            sorted_names = sorted(timings.keys(), key=lambda n: timings[n][0] if timings[n] else float("inf"))
            for i, name in enumerate(sorted_names):
                if not timings[name]:
                    continue
                t_end = timings[name][0]
                if i == 0:
                    t_start = t_end
                    layer_times[name].append(0.0)
                else:
                    prev_name = sorted_names[i - 1]
                    t_start = timings[prev_name][0] if timings[prev_name] else t_end
                    layer_times[name].append((t_end - t_start) * 1000)

        for h in hooks:
            h.remove()

        # Aggregate
        results: List[Dict[str, Any]] = []
        total_ms = 0.0
        for name in layer_times:
            vals = layer_times[name]
            mean_ms = float(np.mean(vals)) if vals else 0.0
            total_ms += mean_ms

        for name in layer_times:
            vals = layer_times[name]
            mean_ms = float(np.mean(vals)) if vals else 0.0
            module = dict(model.named_modules()).get(name)
            n_params = sum(p.numel() for p in module.parameters()) if module is not None else 0
            mod_type = type(module).__name__ if module is not None else "Unknown"
            results.append({
                "name": name,
                "type": mod_type,
                "time_ms": round(mean_ms, 4),
                "time_pct": round(mean_ms / total_ms * 100, 2) if total_ms > 0 else 0.0,
                "params": n_params,
                "output_shape": shapes.get(name),
            })

        return results

    def summary(self, warmup: int = 5, iterations: int = 20) -> str:
        """Return a human-readable profiling summary table."""
        rows = self.run(warmup=warmup, iterations=iterations)

        lines = [
            f"{'Layer':<50} {'Type':<18} {'Time(ms)':>10} {'%':>7} {'Params':>10}",
            "-" * 100,
        ]
        for r in rows:
            lines.append(
                f"{r['name']:<50} {r['type']:<18} {r['time_ms']:>10.4f} "
                f"{r['time_pct']:>6.2f}% {r['params']:>10,}"
            )
        total_ms = sum(r["time_ms"] for r in rows)
        total_params = sum(r["params"] for r in rows)
        lines.append("-" * 100)
        lines.append(
            f"{'TOTAL':<50} {'':<18} {total_ms:>10.4f} {'100.00%':>7} {total_params:>10,}"
        )
        return "\n".join(lines)

    def memory_report(self) -> Dict[str, float]:
        """Return GPU memory usage summary (CUDA only).

        Returns
        -------
        dict
            ``{"allocated_mb", "reserved_mb", "peak_mb"}``
        """
        import torch

        if not torch.cuda.is_available() or self.device != "cuda":
            return {"allocated_mb": 0.0, "reserved_mb": 0.0, "peak_mb": 0.0}

        model = self._load_model()
        dummy = torch.randn(1, 3, *self.input_size, device=self.device)
        torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            model(dummy)
        torch.cuda.synchronize()

        return {
            "allocated_mb": round(torch.cuda.memory_allocated() / 1024 ** 2, 2),
            "reserved_mb": round(torch.cuda.memory_reserved() / 1024 ** 2, 2),
            "peak_mb": round(torch.cuda.max_memory_allocated() / 1024 ** 2, 2),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_model(self):
        if self._model is not None:
            return self._model
        import torch
        self._model = torch.load(str(self.model_path), map_location=self.device)
        if hasattr(self._model, "eval"):
            self._model.eval()
        return self._model
