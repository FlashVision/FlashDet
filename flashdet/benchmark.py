"""Lightweight benchmarking for a trained FlashDet checkpoint.

Reports inference size, parameters, FP16 weight size, latency and throughput.
Optionally evaluates mAP on a COCO-format validation split so accuracy and
speed are measured from the same model file.

Usage from the CLI:

    flashdet benchmark --model best.pth --input-size 320
    flashdet benchmark --model best.pth --val-images data/valid
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

FP16_BYTES_PER_PARAM = 2.0
PARAM_BYTES_PER_PARAM = 4.0


def payload_dtype_gb(params: int) -> float:
    return params * FP16_BYTES_PER_PARAM / (1024 ** 2)


def latency_ms(model: torch.nn.Module, input_size: int, device: str = "cpu",
               warmup: int = 20, repeats: int = 100) -> float:
    """Median single-image forward latency in milliseconds."""
    model.eval()
    x = torch.randn(1, 3, input_size, input_size, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model.predict(x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        samples: list[float] = []
        for _ in range(repeats):
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model.predict(x)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    median = samples[len(samples) // 2]
    p95 = samples[int(len(samples) * 0.95) - 1]
    logger.info("latency median=%.3f ms p95=%.3f ms", median, p95)
    return median


def evaluate_map(model_path: str, val_images: str, device: str = "cpu",
                 conf: float = 0.05, batch_size: int = 8,
                 input_size: int | None = None) -> dict:
    """Run mAP evaluation on a COCO-format validation split."""
    from flashdet.engine.evaluation.validator import Validator

    validator = Validator(
        model_path=model_path,
        val_images=val_images,
        device=device,
        conf_thresh=conf,
        batch_size=batch_size,
        input_size=input_size,
    )
    out = validator.validate()
    summary = {
        "mAP": out.get("mAP", 0.0),
        "num_classes": len(validator.class_names or []),
        "val_loss": out.get("val_loss"),
    }
    coco = out.get("coco_summary") or {}
    summary["mAP50"] = coco.get("map50", out.get("mAP", 0.0))
    summary["mAP5095"] = coco.get("map5095")
    per_class = out.get("AP_per_class")
    if per_class:
        summary["AP_per_class"] = per_class
    return summary


def benchmark_checkpoint(model_path: str, input_size: int | None = None,
                         device: str = "cpu", warmup: int = 20,
                         repeats: int = 100, fuse: bool = True,
                         val_images: str | None = None,
                         conf: float = 0.05) -> dict:
    """Benchmark a checkpoint and return a JSON-serialisable summary."""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(model_path)

    from flashdet.utils.checkpoint import load_inference_model

    model = load_inference_model(model_path, device=device)

    resolved, arch = _resolve_input_size(model_path, input_size)

    if fuse:
        raw = model.module if hasattr(model, "module") else model
        if hasattr(raw, "fuse"):
            try:
                raw.fuse()
            except Exception as exc:  # fusion is best-effort
                logger.warning("fuse() failed, running unfused: %s", exc)

    params = sum(p.numel() for p in model.parameters())
    state = model.state_dict()
    median = latency_ms(model, resolved, device, warmup, repeats)

    summary = {
        "model": os.path.basename(model_path),
        "architecture": arch,
        "input_size": resolved,
        "params": params,
        "params_fp16_mb": round(params * FP16_BYTES_PER_PARAM / (1024 ** 2), 2),
        "state_size_mb": round(
            sum(t.numel() * PARAM_BYTES_PER_PARAM for t in state.values())
            / (1024 ** 2), 2
        ),
        "device": device,
        "latency_ms": round(median, 3),
        "throughput_fps": round(1000.0 / median, 1),
    }

    if val_images:
        if not os.path.isdir(val_images):
            raise FileNotFoundError(val_images)
        summary["eval"] = evaluate_map(
            model_path, val_images, device=device, conf=conf,
        )

    return summary


def _resolve_input_size(model_path: str, input_size: int | None) -> tuple[int, str]:
    """Use the training input size and architecture from the checkpoint when
    not explicitly overridden on the command line."""
    from flashdet.utils.checkpoint import safe_torch_load

    ckpt = safe_torch_load(model_path, map_location="cpu")
    config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    arch = config.get("architecture", "flashdet")
    configured = config.get("input_size") if isinstance(config, dict) else None
    if isinstance(configured, (list, tuple)):
        configured = configured[0] if configured else None
    return int(input_size or configured or 320), arch
    size = config.get("input_size")
    if isinstance(size, (list, tuple)):
        size = size[0]
    return int(input_size or size or 320), arch


def save_report(summary: dict, output: str | None) -> str | None:
    """Optionally persist the benchmark summary as JSON."""
    if not output:
        return None
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(summary, f, indent=2)
    return output
