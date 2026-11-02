#!/usr/bin/env python3
"""Concurrent load test for the FlashDet inference flow.

Simulates N independent users hitting a single loaded predictor.  It also injects
malformed payloads so you can see whether the input-validation layer rejects
them cleanly or the process crashes.

Examples:
    python scripts/load_test.py --users 1000 --concurrency 32 --size 128
    python scripts/load_test.py --users 5000 --concurrency 64 --size 320
    python scripts/load_test.py --model workspace/pico/model_best_inference.pth
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Running `python scripts/load_test.py` puts scripts/ first on sys.path, which
# can shadow the local checkout with an installed flashdet package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402  (import after sys.path bootstrap)


def _make_model_checkpoint(size: int, backbone: str, workdir: str) -> str:
    from flashdet.models.architectures.flashdet import FlashDet
    from flashdet.utils.checkpoint import save_inference_weights

    model = FlashDet(num_classes=5, size="p", backbone_type=backbone).eval()
    path = os.path.join(workdir, "load_test_pico.pth")
    save_inference_weights(
        model,
        path,
        config={
            "num_classes": 5,
            "model_size": "p",
            "architecture": "flashdet",
            "backbone_type": backbone,
            "neck_type": model.neck_type,
            "input_size": [size, size],
            "class_names": [f"c{i}" for i in range(5)],
        },
    )
    return path


def _image(size: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 256, (size, size, 3), dtype=np.uint8)


def _chaos_payload(case: int, size: int) -> object:
    """Return an intentionally malformed payload."""
    cases = [
        np.zeros((size, size, 2), dtype=np.uint8),        # wrong channels
        np.zeros((size, size, 3, 1), dtype=np.uint8),     # wrong rank
        np.zeros((0, size, 3), dtype=np.uint8),           # zero height
        [[1, 2, 3]],                                      # not an ndarray
    ]
    return cases[case % len(cases)]


def _run_one(predictor, size: int, seed: int, chaos_every: int) -> dict:
    rng = np.random.default_rng(seed)
    is_chaos = chaos_every > 0 and seed % chaos_every == 0
    payload = _chaos_payload(seed, size) if is_chaos else _image(size, rng)
    start = time.perf_counter()
    try:
        predictor(payload)
        elapsed = time.perf_counter() - start
        return {
            "ok": not is_chaos,
            "rejected": False,
            "kind": "chaos" if is_chaos else "normal",
            "latency": elapsed,
        }
    except (ValueError, TypeError) as exc:
        elapsed = time.perf_counter() - start
        return {
            "ok": is_chaos,
            "rejected": is_chaos,
            "kind": "chaos" if is_chaos else "normal",
            "latency": elapsed,
            "error": type(exc).__name__,
            "message": str(exc)[:180],
        }
    except Exception as exc:  # unexpected failure
        elapsed = time.perf_counter() - start
        return {
            "ok": False,
            "rejected": False,
            "kind": "chaos" if is_chaos else "normal",
            "latency": elapsed,
            "error": type(exc).__name__,
            "message": str(exc)[:300],
        }


def _percentile(sorted_values, pct: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, int(round((pct / 100.0) * (len(sorted_values) - 1))))
    return sorted_values[idx]


def main() -> int:
    parser = argparse.ArgumentParser(description="FlashDet concurrent load test")
    parser.add_argument("--users", type=int, default=1000, help="number of simulated users")
    parser.add_argument("--concurrency", type=int, default=32, help="parallel workers")
    parser.add_argument("--size", type=int, default=128, help="input image size")
    parser.add_argument("--backbone", default="repvit", help="Pico backbone for the test model")
    parser.add_argument("--model", default=None, help="optional real inference checkpoint")
    parser.add_argument("--chaos-every", type=int, default=50,
                        help="every Nth user sends a malformed payload (0 disables)")
    parser.add_argument("--output", default=None, help="optional JSON report path")
    args = parser.parse_args()

    import torch  # noqa: F401  (ensures torch is configured before predictor import)
    from flashdet.engine.inference.predictor import Predictor

    tmpdir = None
    if args.model:
        model_path = args.model
    else:
        tmpdir = tempfile.mkdtemp(prefix="flashdet_load_")
        model_path = _make_model_checkpoint(args.size, args.backbone, tmpdir)
    if args.model and args.size != 128:
        # A real checkpoint carries its own input size in config.
        pass

    load_start = time.perf_counter()
    predictor = Predictor(model_path=model_path, device="cpu")
    load_seconds = time.perf_counter() - load_start

    # Warm up once so lazy allocations do not skew the first user.
    predictor(_image(args.size, np.random.default_rng(0)))

    started = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(_run_one, predictor, args.size, i, args.chaos_every)
            for i in range(args.users)
        ]
        for fut in as_completed(futures):
            results.append(fut.result())
    wall = time.perf_counter() - started

    normal = [r for r in results if r["kind"] == "normal"]
    chaos = [r for r in results if r["kind"] == "chaos"]
    latencies = sorted(r["latency"] for r in normal if r.get("ok"))
    errors = Counter(
        f"{r.get('error', 'unknown')}: {r.get('message', '')}"
        for r in results if not r.get("ok")
    )

    report = {
        "users": args.users,
        "concurrency": args.concurrency,
        "input_size": args.size,
        "backbone": args.backbone if not args.model else "checkpoint",
        "model_load_seconds": round(load_seconds, 4),
        "wall_seconds": round(wall, 4),
        "requests_per_second": round(args.users / wall, 2) if wall else 0.0,
        "latency_ms": {
            "p50": round(_percentile(latencies, 50) * 1000, 2),
            "p95": round(_percentile(latencies, 95) * 1000, 2),
            "p99": round(_percentile(latencies, 99) * 1000, 2),
            "max": round(max(latencies) * 1000, 2) if latencies else 0.0,
        },
        "normal_ok": sum(1 for r in normal if r.get("ok")),
        "normal_failed": sum(1 for r in normal if not r.get("ok")),
        "chaos_rejected": sum(1 for r in chaos if r.get("rejected")),
        "chaos_unexpected": sum(1 for r in chaos if not r.get("ok")),
        "errors": dict(errors),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2),
    }

    print(json.dumps(report, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report written to {args.output}")

    if tmpdir:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    return 0 if report["normal_failed"] == 0 and report["chaos_unexpected"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
