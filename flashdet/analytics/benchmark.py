"""Benchmark — measure FlashDet model speed, size and parameter count."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


class Benchmark:
    """Benchmark FlashDet model speed and resource usage.

    Parameters
    ----------
    model_path : str | Path
        Path to a saved FlashDet checkpoint (``.pth`` / ``.onnx``).
    device : str
        ``"cuda"`` or ``"cpu"``.
    input_size : int | tuple[int, int]
        Network input resolution.  A single int is treated as
        ``(input_size, input_size)``.
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
        self._is_onnx: bool = self.model_path.suffix.lower() == ".onnx"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, warmup: int = 10, iterations: int = 100) -> Dict[str, float]:
        """Run a speed benchmark.

        Returns
        -------
        dict
            ``{"fps": …, "latency_ms": …, "params": …, "model_size_mb": …}``
        """
        model = self._load_model()
        dummy = self._make_dummy_input()

        if self._is_onnx:
            return self._bench_onnx(model, dummy, warmup, iterations)
        return self._bench_pytorch(model, dummy, warmup, iterations)

    def compare(self, model_paths: List[Union[str, Path]]) -> List[Dict[str, Any]]:
        """Compare multiple models side by side.

        Parameters
        ----------
        model_paths : list[str | Path]
            Paths to additional models to compare against the primary model.

        Returns
        -------
        list[dict]
            One result dict per model (primary model first).
        """
        all_paths = [self.model_path] + [Path(p) for p in model_paths]
        results: List[Dict[str, Any]] = []
        for p in all_paths:
            bm = Benchmark(p, device=self.device, input_size=self.input_size)
            res = bm.run()
            res["model"] = str(p.name)
            results.append(res)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self):
        if self._model is not None:
            return self._model

        if self._is_onnx:
            import onnxruntime as ort
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.device == "cuda" else ["CPUExecutionProvider"]
            self._model = ort.InferenceSession(str(self.model_path), providers=providers)
        else:
            import torch
            self._model = torch.load(str(self.model_path), map_location=self.device)
            if hasattr(self._model, "eval"):
                self._model.eval()
        return self._model

    def _make_dummy_input(self) -> np.ndarray:
        return np.random.rand(1, 3, *self.input_size).astype(np.float32)

    def _bench_pytorch(self, model, dummy: np.ndarray, warmup: int, iterations: int) -> Dict[str, float]:
        import torch

        tensor = torch.from_numpy(dummy).to(self.device)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                model(tensor)
        if self.device == "cuda":
            torch.cuda.synchronize()

        # Timed runs
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(iterations):
                model(tensor)
        if self.device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        latency_ms = (elapsed / iterations) * 1000
        fps = iterations / elapsed

        params = sum(p.numel() for p in model.parameters()) if hasattr(model, "parameters") else 0
        size_mb = self.model_path.stat().st_size / (1024 * 1024)

        return {
            "fps": round(fps, 2),
            "latency_ms": round(latency_ms, 3),
            "params": params,
            "model_size_mb": round(size_mb, 2),
        }

    def _bench_onnx(self, session, dummy: np.ndarray, warmup: int, iterations: int) -> Dict[str, float]:
        input_name = session.get_inputs()[0].name

        for _ in range(warmup):
            session.run(None, {input_name: dummy})

        start = time.perf_counter()
        for _ in range(iterations):
            session.run(None, {input_name: dummy})
        elapsed = time.perf_counter() - start

        latency_ms = (elapsed / iterations) * 1000
        fps = iterations / elapsed
        size_mb = self.model_path.stat().st_size / (1024 * 1024)

        return {
            "fps": round(fps, 2),
            "latency_ms": round(latency_ms, 3),
            "params": 0,  # ONNX doesn't expose param count easily
            "model_size_mb": round(size_mb, 2),
        }
