#!/usr/bin/env python
"""Measure the FP32 -> FP16 -> fused -> INT8 deployment pipeline.

Runs every stage on the same checkpoint and validation split, then prints a
markdown table with parameters, on-disk size, forward latency, FPS,
mAP@0.5 and decoded-box IoU against the FP32 baseline:

    python scripts/benchmark_pipeline.py \
        --model best.pth --val-images data/valid --input-size 320

Stages measured:

    fp32          original PyTorch checkpoint
    fp16          half-precision weights on disk (same architecture)
    fused         layer/kernel fusion applied, saved as a checkpoint
    onnx          FP32 ONNX export
    onnx-fused    FP32 ONNX export of the fused checkpoint
    int8          INT8 QDQ ONNX quantized from the FP32 ONNX
    int8-fused    INT8 QDQ ONNX quantized from the fused ONNX

Latency is forward-only (decode/NMS excluded) and measured on CPU by default.
The FP16 row casts the model to half; consumer CPUs have no FP16 arithmetic
acceleration, so treat it as a storage/GPU format.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from flashdet.deployment import (
    build_model_from_checkpoint,
    export_onnx,
    input_size_from_checkpoint,
    onnx_quantize_int8,
)
from flashdet.fusing import default_passes, run_fusions
from flashdet.fusing.checkpoint import save_fused_checkpoint
from flashdet.utils.checkpoint import safe_torch_load


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------


def _median_latency(fn, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


def _torch_predict(model: torch.nn.Module, dtype=None):
    def run(images: torch.Tensor) -> torch.Tensor:
        if dtype is not None:
            images = images.to(dtype)
        with torch.no_grad():
            out = model.forward(images)
        if isinstance(out, dict):
            return torch.cat([out["o2o_cls"], out["o2o_reg"]], dim=-1)
        return out

    return run


def _onnx_predict(session):
    def run(images: torch.Tensor) -> torch.Tensor:
        result = session.run(
            None, {"images": images.numpy().astype(np.float32)}
        )[0]
        return torch.from_numpy(result)

    return run


def _anchors_for(model: torch.nn.Module, size: int):
    """Anchor grid for the exported output layout (same for every stage)."""
    device = next(model.parameters()).device
    with torch.no_grad():
        feats = model.forward(torch.zeros(1, 3, size, size, device=device))
    return model._get_anchors(feats["feat_sizes"], device)


def _decode(raw: torch.Tensor, anchors, num_classes: int, size: int,
            conf: float):
    from flashdet.utils.bbox import decode_batch_nms_free

    return decode_batch_nms_free(
        raw[..., :num_classes].float(),
        raw[..., num_classes:].float(),
        anchors[0], anchors[1],
        img_hw=(size, size),
        score_thr=conf,
        max_det=300,
    )


def _evaluate(predict, anchors, loader, num_classes: int, size: int,
              conf: float):
    """mAP@0.5 plus decoded detections kept for IoU comparisons."""
    from flashdet.utils.metrics import compute_map

    all_preds: list[dict] = []
    all_gts: list[dict] = []
    kept: list[tuple[np.ndarray, np.ndarray]] = []

    for images, gt_meta in loader:
        dets = _decode(predict(images), anchors, num_classes, size, conf)
        for b, (boxes, labels) in enumerate(dets):
            all_preds.append({
                "boxes": boxes[:, :4].numpy(),
                "scores": boxes[:, 4].numpy(),
                "labels": labels.numpy(),
            })
            all_gts.append({
                "boxes": gt_meta["gt_bboxes"][b],
                "labels": gt_meta["gt_labels"][b],
            })
            kept.append((boxes[:, :4].numpy(), labels.numpy()))

    result = compute_map(
        all_preds, all_gts, iou_threshold=0.5, num_classes=num_classes,
    )
    return float(result["mAP"]), kept


def _box_iou(ref, cand, num_probes: int = 8) -> float:
    """Class-aware mean IoU of each baseline box with its best counterpart."""
    ious: list[float] = []
    for (rb, rl), (cb, cl) in list(zip(ref, cand))[:num_probes]:
        if rb.size == 0:
            continue
        if cb.size == 0:
            ious.extend([0.0] * len(rb))
            continue
        x1 = np.maximum(rb[:, None, 0], cb[None, :, 0])
        y1 = np.maximum(rb[:, None, 1], cb[None, :, 1])
        x2 = np.minimum(rb[:, None, 2], cb[None, :, 2])
        y2 = np.minimum(rb[:, None, 3], cb[None, :, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        area_r = (rb[:, 2] - rb[:, 0]) * (rb[:, 3] - rb[:, 1])
        area_c = (cb[:, 2] - cb[:, 0]) * (cb[:, 3] - cb[:, 1])
        union = area_r[:, None] + area_c[None, :] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
        iou = np.where(rl[:, None] == cl[None, :], iou, 0.0)
        best = iou.max(axis=1)
        ious.extend(best.tolist())
    return float(np.mean(ious)) if ious else 0.0


def _file_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 ** 2)


def _save_fp16(checkpoint_path: str, output_path: str) -> str:
    """Write a half-precision copy of a state-dict checkpoint."""
    payload = safe_torch_load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("FP16 conversion needs a state-dict checkpoint")
    key = "model_state_dict" if "model_state_dict" in payload else "state_dict"
    state = payload.get(key)
    if not isinstance(state, dict):
        raise ValueError("FP16 conversion needs a state dict")
    payload[key] = {
        name: value.half()
        if torch.is_tensor(value) and value.is_floating_point() else value
        for name, value in state.items()
    }
    payload["half"] = True
    torch.save(payload, output_path)
    return output_path


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run(args) -> dict:
    os.makedirs(args.workdir, exist_ok=True)
    size = args.input_size or input_size_from_checkpoint(args.model)
    conf = args.conf

    raw_ckpt = safe_torch_load(args.model, map_location="cpu")
    config = raw_ckpt.get("config", {}) if isinstance(raw_ckpt, dict) else {}

    # --- fused checkpoint -------------------------------------------------
    base_model = build_model_from_checkpoint(args.model, fuse=False)
    probe = torch.randn(1, 3, size, size)
    passes = default_passes(example_input=probe)
    fused_model, report = run_fusions(
        base_model, passes, example_input=probe, verify=True)
    if not report.parity_ok:
        raise RuntimeError(
            f"fusion parity failed: {report.parity_max_abs_diff}")
    fused_path = os.path.join(args.workdir, "fused.pth")
    save_fused_checkpoint(fused_model, fused_path, config=config,
                          input_size=size, report=report)
    fused_half_path = os.path.join(args.workdir, "fused_fp16.pth")
    save_fused_checkpoint(fused_model.half(), fused_half_path, config=config,
                          input_size=size, report=report)

    # --- FP16 weights -----------------------------------------------------
    fp16_path = args.fp16_model or os.path.join(args.workdir, "fp16.pth")
    if not args.fp16_model:
        _save_fp16(args.model, fp16_path)

    # --- ONNX exports -----------------------------------------------------
    onnx_path = os.path.join(args.workdir, "model.onnx")
    onnx_fused_path = os.path.join(args.workdir, "model_fused.onnx")
    export_onnx(args.model, onnx_path, fuse=False, simplify=True)
    export_onnx(fused_path, onnx_fused_path, fuse=False, simplify=True)

    calib = args.calib_images or args.val_images
    int8_path = os.path.join(args.workdir, "model_int8.onnx")
    int8_fused_path = os.path.join(args.workdir, "model_fused_int8.onnx")
    if calib:
        onnx_quantize_int8(onnx_path, int8_path, calib_images=calib,
                           max_batches=args.calib_batches)
        onnx_quantize_int8(onnx_fused_path, int8_fused_path,
                           calib_images=calib, max_batches=args.calib_batches)

    # --- stage models -----------------------------------------------------
    fp32_model = build_model_from_checkpoint(args.model, fuse=False).eval()
    fp32_half = build_model_from_checkpoint(args.model, fuse=False).eval().half()
    fused_eval = build_model_from_checkpoint(fused_path, fuse=False).eval()
    fused_half = build_model_from_checkpoint(
        fused_half_path, fuse=False).eval().half()

    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("onnxruntime is required for the ONNX stages") from exc

    def session(path):
        return ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    onnx_stages = [
        ("onnx", "FP32 ONNX", onnx_path, session(onnx_path), fp32_model),
        ("onnx-fused", "FP32 ONNX (fused)", onnx_fused_path,
         session(onnx_fused_path), fused_eval),
    ]
    if calib:
        onnx_stages += [
            ("int8", "INT8 ONNX", int8_path, session(int8_path), fp32_model),
            ("int8-fused", "INT8 ONNX (fused)", int8_fused_path,
             session(int8_fused_path), fused_eval),
        ]

    # --- shared validation data ------------------------------------------
    from flashdet.data.dataloader import create_dataloader

    loader = create_dataloader(
        img_dir=args.val_images,
        ann_file=os.path.join(args.val_images, "_annotations.coco.json"),
        batch_size=args.batch_size,
        input_size=(size, size),
        num_workers=0,
        is_train=False,
    )
    num_classes = int(getattr(fp32_model, "num_classes", 0))
    anchors = _anchors_for(fp32_model, size)

    # Baseline predictions also define the IoU reference.
    base_map, base_dets = _evaluate(
        _torch_predict(fp32_model), anchors, loader, num_classes, size, conf)

    x = torch.randn(1, 3, size, size)
    half_input = x.half()
    rows = []

    def add(name, label, path, model, predict, ref_dets, input_tensor,
            map_value):
        params = sum(p.numel() for p in model.parameters())
        latency = _median_latency(
            lambda p=predict, t=input_tensor: p(t),
            args.warmup, args.repeats,
        )
        rows.append({
            "stage": name,
            "label": label,
            "params": params,
            "size_mb": round(_file_mb(path), 2),
            "latency_ms": round(latency, 2),
            "fps": round(1000.0 / latency, 1) if latency else None,
            "map50": round(map_value, 4),
            "box_iou": round(_box_iou(base_dets, ref_dets, args.probes), 3),
        })

    add("fp32", "FP32 PyTorch", args.model, fp32_model,
        _torch_predict(fp32_model), base_dets, x, base_map)

    fused_map, fused_dets = _evaluate(
        _torch_predict(fused_eval), anchors, loader, num_classes, size, conf)
    add("fused", "FP32 PyTorch (fused)", fused_path, fused_eval,
        _torch_predict(fused_eval), fused_dets, x, fused_map)

    fp16_map, fp16_dets = _evaluate(
        _torch_predict(fp32_half, torch.float16), anchors, loader,
        num_classes, size, conf)
    add("fp16", "FP16 PyTorch", fp16_path, fp32_half,
        _torch_predict(fp32_half, torch.float16), fp16_dets, half_input,
        fp16_map)

    fused_half_map, fused_half_dets = _evaluate(
        _torch_predict(fused_half, torch.float16), anchors, loader,
        num_classes, size, conf)
    add("fp16-fused", "FP16 PyTorch (fused)", fused_half_path, fused_half,
        _torch_predict(fused_half, torch.float16), fused_half_dets,
        half_input, fused_half_map)

    for name, label, path, sess, model in onnx_stages:
        predict = _onnx_predict(sess)
        map_value, dets = _evaluate(
            predict, anchors, loader, num_classes, size, conf)
        add(name, label, path, model, predict, dets, x, map_value)

    return {
        "model": args.model,
        "input_size": size,
        "conf": conf,
        "fusion": report.to_dict(),
        "rows": rows,
    }


def _markdown(result: dict) -> str:
    lines = [
        "| Stage | Params | Size (MB) | Latency (ms) | FPS | mAP@0.5 | Box IoU |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["rows"]:
        lines.append(
            f"| {row['label']} | {row['params']:,} | {row['size_mb']:.2f} | "
            f"{row['latency_ms']:.2f} | {row['fps']:.1f} | "
            f"{row['map50']:.3f} | {row['box_iou']:.3f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="FP32 checkpoint")
    parser.add_argument("--val-images", required=True,
                        help="COCO-format validation split")
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--fp16-model", default=None,
                        help="Existing FP16 checkpoint (created if omitted)")
    parser.add_argument("--calib-images", default=None,
                        help="Calibration folder (defaults to --val-images)")
    parser.add_argument("--calib-batches", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--probes", type=int, default=8,
                        help="Images used for the box-IoU comparison")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--workdir", default="workspace/pipeline")
    parser.add_argument("--output", default=None,
                        help="Write the JSON result and markdown table here")
    parser.add_argument("--json", action="store_true",
                        help="Print JSON instead of markdown")
    args = parser.parse_args()

    result = run(args)
    table = _markdown(result)
    print(table)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".",
                    exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        md_path = os.path.splitext(args.output)[0] + ".md"
        with open(md_path, "w", encoding="utf-8") as handle:
            handle.write(table + "\n")
        print(f"\nSaved: {args.output} and {md_path}")
    if args.json:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
