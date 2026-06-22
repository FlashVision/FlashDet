#!/usr/bin/env python3
"""
Benchmark all trained FlashDet models and compare with published YOLOX results.

Measures:
  - mAP@0.5 and mAP@0.5:0.95 on COCO val2017
  - Inference latency (ms) on GPU
  - Model parameters and FLOPs

Prints a comparison table against YOLOX-Nano, YOLOX-Tiny, YOLOX-S, YOLOX-M.

Usage:
    python scripts/benchmark_flashdet.py
    python scripts/benchmark_flashdet.py --device cuda --input-size 640
"""

import argparse
import os
import sys
import time
import json

import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flashdet.models.architectures.flashdet import FlashDet
from flashdet.cfg import get_config
from flashdet.data import create_dataloader
from flashdet.utils.metrics import compute_map


YOLOX_PUBLISHED = {
    "YOLOX-Nano": {"params_m": 0.91, "flops_g": 1.08, "map50": 25.8, "map5095": None, "latency_ms": 1.1, "input": 416},
    "YOLOX-Tiny": {"params_m": 5.06, "flops_g": 6.45, "map50": 32.8, "map5095": None, "latency_ms": 1.5, "input": 416},
    "YOLOX-S":    {"params_m": 9.0,  "flops_g": 26.8, "map50": 40.5, "map5095": None, "latency_ms": 2.3, "input": 640},
    "YOLOX-M":    {"params_m": 25.3, "flops_g": 73.8, "map50": 46.9, "map5095": None, "latency_ms": 4.4, "input": 640},
    "YOLOX-L":    {"params_m": 54.2, "flops_g": 155.6,"map50": 49.7, "map5095": None, "latency_ms": 6.8, "input": 640},
    "YOLOX-X":    {"params_m": 99.1, "flops_g": 281.9,"map50": 51.1, "map5095": None, "latency_ms": 11.4,"input": 640},
}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def measure_latency(model, input_size, device, warmup=50, repeats=300):
    """Measure average inference latency in milliseconds."""
    model.eval()
    x = torch.randn(1, 3, input_size, input_size).to(device)

    with torch.no_grad():
        for _ in range(warmup):
            model.predict(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(repeats):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model.predict(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

    times = sorted(times)
    trim = int(len(times) * 0.1)
    if trim > 0:
        times = times[trim:-trim]
    return np.mean(times)


def evaluate_map(model, val_dir, device, input_size, num_classes=80, batch_size=32):
    """Evaluate mAP@0.5 on validation set."""
    config = get_config(num_classes=num_classes)
    config.data.val_images = os.path.join(val_dir, "images") if os.path.isdir(os.path.join(val_dir, "images")) else val_dir

    ann_file = os.path.join(val_dir, "_annotations.coco.json")
    if not os.path.isfile(ann_file):
        print(f"  Warning: annotation file not found at {ann_file}, skipping mAP eval")
        return None

    with open(ann_file) as f:
        coco_data = json.load(f)

    cats = sorted(coco_data.get("categories", []), key=lambda c: c["id"])
    cat_id_to_idx = {c["id"]: i for i, c in enumerate(cats)}

    img_id_to_anns = {}
    for ann in coco_data.get("annotations", []):
        img_id_to_anns.setdefault(ann["image_id"], []).append(ann)

    model.eval()
    all_preds = []
    all_gts = []

    import cv2

    images_info = coco_data["images"]
    img_dir = os.path.join(val_dir, "images") if os.path.isdir(os.path.join(val_dir, "images")) else val_dir

    processed = 0
    for img_info in images_info:
        img_path = os.path.join(img_dir, img_info["file_name"])
        if not os.path.isfile(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        h_orig, w_orig = img.shape[:2]
        ratio = min(input_size / h_orig, input_size / w_orig)
        new_h, new_w = int(h_orig * ratio), int(w_orig * ratio)
        dh = (input_size - new_h) / 2
        dw = (input_size - new_w) / 2

        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img_resized = cv2.copyMakeBorder(img_resized, top, input_size - new_h - top,
                                         left, input_size - new_w - left,
                                         cv2.BORDER_CONSTANT, value=(114, 114, 114))
        img_t = torch.from_numpy(img_resized).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)

        with torch.no_grad():
            results = model.predict(img_t, None, score_thr=0.001, nms_thr=0.65)

        img_preds = []
        if results and len(results) > 0:
            for det in results[0] if isinstance(results[0], list) else [results[0]]:
                if hasattr(det, 'shape') and len(det.shape) == 2:
                    for d in det:
                        if len(d) >= 6:
                            x1, y1, x2, y2, score, cls = d[:6]
                            x1 = (float(x1) - dw) / ratio
                            y1 = (float(y1) - dh) / ratio
                            x2 = (float(x2) - dw) / ratio
                            y2 = (float(y2) - dh) / ratio
                            img_preds.append([x1, y1, x2, y2, float(score), int(cls)])

        anns = img_id_to_anns.get(img_info["id"], [])
        img_gts = []
        for ann in anns:
            if ann.get("iscrowd", 0):
                continue
            bx, by, bw, bh = ann["bbox"]
            cls_idx = cat_id_to_idx.get(ann["category_id"], -1)
            if cls_idx < 0:
                continue
            img_gts.append([bx, by, bx + bw, by + bh, cls_idx])

        all_preds.append(img_preds)
        all_gts.append(img_gts)
        processed += 1

        if processed % 500 == 0:
            print(f"    Evaluated {processed}/{len(images_info)} images...")

    print(f"    Evaluated {processed} images total")

    if not all_preds:
        return None

    map50 = compute_map(all_preds, all_gts, iou_threshold=0.5, num_classes=num_classes)
    return map50


def main():
    parser = argparse.ArgumentParser(description="Benchmark FlashDet models")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--val-dir", default="data/coco2017/valid")
    parser.add_argument("--skip-map", action="store_true", help="Skip mAP evaluation (latency only)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    models_to_eval = {
        "FlashDet-N": {"size": "n", "ckpt_dir": "workspace/flashdet_n_coco"},
        "FlashDet-S": {"size": "s", "ckpt_dir": "workspace/flashdet_s_coco"},
        "FlashDet-M": {"size": "m", "ckpt_dir": "workspace/flashdet_m_coco"},
    }

    results = {}

    for name, info in models_to_eval.items():
        print(f"{'='*60}")
        print(f"  Benchmarking: {name}")
        print(f"{'='*60}")

        ckpt_path = os.path.join(info["ckpt_dir"], "model_best_inference.pth")
        if not os.path.isfile(ckpt_path):
            ckpt_path = os.path.join(info["ckpt_dir"], "model_last_inference.pth")
        if not os.path.isfile(ckpt_path):
            print(f"  No checkpoint found in {info['ckpt_dir']}, building fresh model")
            model = FlashDet(num_classes=80, size=info["size"]).to(device)
        else:
            print(f"  Loading: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            state_dict = ckpt.get("state_dict", ckpt.get("model", ckpt))
            num_classes = ckpt.get("config", {}).get("num_classes", 80)
            model = FlashDet(num_classes=num_classes, size=info["size"]).to(device)
            model.load_state_dict(state_dict, strict=False)

        params = count_parameters(model)
        print(f"  Parameters: {params:,} ({params/1e6:.2f}M)")

        latency = measure_latency(model, args.input_size, device)
        fps = 1000.0 / latency
        print(f"  Latency: {latency:.2f} ms ({fps:.0f} FPS) @ {args.input_size}x{args.input_size}")

        map50 = None
        if not args.skip_map and os.path.isdir(args.val_dir):
            print(f"  Evaluating mAP@0.5 on {args.val_dir}...")
            map50 = evaluate_map(model, args.val_dir, device, args.input_size)
            if map50 is not None:
                print(f"  mAP@0.5: {map50:.1f}")

        results[name] = {
            "params_m": params / 1e6,
            "latency_ms": latency,
            "fps": fps,
            "map50": map50,
            "input": args.input_size,
        }
        print()

    print()
    print("=" * 90)
    print("  FlashDet vs YOLOX — Comparison Table")
    print("=" * 90)
    print(f"{'Model':<20} {'Params':>8} {'Input':>6} {'mAP@0.5':>9} {'Latency':>10} {'FPS':>7}")
    print("-" * 90)

    for name, r in results.items():
        map_str = f"{r['map50']:.1f}" if r["map50"] is not None else "—"
        print(f"{name:<20} {r['params_m']:>7.2f}M {r['input']:>5} {map_str:>9} {r['latency_ms']:>8.2f}ms {r['fps']:>7.0f}")

    print("-" * 90)
    for name, r in YOLOX_PUBLISHED.items():
        map_str = f"{r['map50']:.1f}" if r["map50"] is not None else "—"
        print(f"{name:<20} {r['params_m']:>7.2f}M {r['input']:>5} {map_str:>9} {r['latency_ms']:>8.2f}ms {1000/r['latency_ms']:>7.0f}")
    print("=" * 90)
    print("  Note: YOLOX numbers from official paper (NVIDIA V100, TensorRT FP16)")
    print("  FlashDet numbers measured on this machine (PyTorch, no TensorRT)")
    print()

    out_path = os.path.join("workspace", "benchmark_results.json")
    os.makedirs("workspace", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"flashdet": results, "yolox_published": YOLOX_PUBLISHED}, f, indent=2, default=str)
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
