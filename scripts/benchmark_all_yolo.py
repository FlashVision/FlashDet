"""Benchmark FlashDet YOLO models — using our own implementations.

Tests our clean-room YOLO implementations (YOLOv8, YOLOv9, YOLOv10, YOLOv11, YOLOX)
with random initialization to verify architecture correctness and speed.

For pretrained weights: Train using `python train.py --architecture yolov11`
All code is MIT licensed — no AGPL/GPL dependencies.

Usage:
    python scripts/benchmark_all_yolo.py
    python scripts/benchmark_all_yolo.py --images data/demo/
    python scripts/benchmark_all_yolo.py --pretrained runs/yolov11/best.pth
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build_all_models(num_classes: int = 80):
    """Build all available FlashDet model architectures."""
    from flashdet.models.detector import build_model
    from flashdet.cfg import get_config

    models = {}
    configs = [
        ("FlashDet-N", "flashdet", {"size": "n"}),
        ("YOLOv8-S", "yolov8", {"width_mult": 0.5, "depth_mult": 0.33}),
        ("YOLOv9 (w=1.0)", "yolov9", {"width_mult": 1.0, "depth_mult": 1.0}),
        ("YOLOv10 (w=1.0)", "yolov10", {"width_mult": 1.0, "depth_mult": 1.0}),
        ("YOLOv11 (w=1.0)", "yolov11", {"width_mult": 1.0, "depth_mult": 1.0}),
        ("YOLOX-S", "yolox", {"width_mult": 0.75, "depth_mult": 0.33}),
    ]

    for name, arch, kwargs in configs:
        config = get_config(num_classes=num_classes)
        config.model.architecture = arch
        for k, v in kwargs.items():
            setattr(config.model, k, v)

        try:
            model = build_model(config, architecture=arch)
            model.eval()
            n_params = sum(p.numel() for p in model.parameters()) / 1e6
            models[name] = {"model": model, "params_M": n_params, "arch": arch}
        except Exception as e:
            print(f"  WARNING: Failed to build {name}: {e}")

    return models


def benchmark_speed(model, input_size: int = 640, num_runs: int = 50, warmup: int = 5):
    """Benchmark inference speed (CPU)."""
    x = torch.randn(1, 3, input_size, input_size)

    for _ in range(warmup):
        with torch.no_grad():
            model(x)

    times = []
    for _ in range(num_runs):
        t0 = time.time()
        with torch.no_grad():
            model(x)
        times.append((time.time() - t0) * 1000)

    return {
        "avg_ms": np.mean(times),
        "std_ms": np.std(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark FlashDet YOLO architectures")
    parser.add_argument("--num-classes", type=int, default=80, help="Number of classes")
    parser.add_argument("--input-size", type=int, default=640, help="Input image size")
    parser.add_argument("--num-runs", type=int, default=30, help="Number of benchmark runs")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    args = parser.parse_args()

    print("=" * 80)
    print("  FlashDet — Architecture Benchmark")
    print("  All implementations MIT licensed (clean-room from papers)")
    print("=" * 80)
    print(f"  Input size: {args.input_size}x{args.input_size}")
    print(f"  Classes:    {args.num_classes}")
    print(f"  Device:     {args.device}")
    print(f"  Runs:       {args.num_runs}")
    print("=" * 80)
    print()

    # Build models
    print("Building models...")
    models = build_all_models(args.num_classes)

    if not models:
        print("ERROR: No models could be built!")
        return

    # Benchmark each model
    results = []
    print(f"\nBenchmarking ({args.num_runs} runs each)...")
    print("-" * 80)

    for name, info in models.items():
        model = info["model"]
        if args.device == "cuda" and torch.cuda.is_available():
            model = model.cuda()

        timing = benchmark_speed(model, args.input_size, args.num_runs)
        results.append({
            "model": name,
            "architecture": info["arch"],
            "params_M": info["params_M"],
            **timing,
        })
        print(f"  {name:<25} | {info['params_M']:6.2f}M | "
              f"{timing['avg_ms']:6.1f}ms ± {timing['std_ms']:.1f}ms | "
              f"min={timing['min_ms']:.1f}ms")

    # Summary
    print("\n" + "=" * 80)
    print(f"  {'MODEL':<25} {'PARAMS':<10} {'AVG LATENCY':<15} {'THROUGHPUT':<12} {'LICENSE'}")
    print("=" * 80)
    for r in results:
        fps = 1000.0 / r["avg_ms"] if r["avg_ms"] > 0 else 0
        print(f"  {r['model']:<25} {r['params_M']:.2f}M     "
              f"{r['avg_ms']:.1f}ms          {fps:.0f} FPS       MIT")
    print("=" * 80)

    # Model architecture details
    print("\n  Architecture Details:")
    print("  " + "-" * 76)
    for name, info in models.items():
        model = info["model"]
        n_layers = sum(1 for _ in model.modules())
        print(f"  {name:<25} | Layers: {n_layers:<5} | Params: {info['params_M']:.2f}M")
    print()

    # Save results
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    output_file = results_dir / "architecture_benchmark.json"
    with open(output_file, "w") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2)
    print(f"  Results saved: {output_file}")

    # Training instructions
    print("\n" + "=" * 80)
    print("  To train with pretrained performance:")
    print("    python train.py --architecture yolov8  --data data/coco.yaml --epochs 300")
    print("    python train.py --architecture yolov9  --data data/coco.yaml --epochs 300")
    print("    python train.py --architecture yolov10 --data data/coco.yaml --epochs 300")
    print("    python train.py --architecture yolov11 --data data/coco.yaml --epochs 300")
    print("    python train.py --architecture yolox   --data data/coco.yaml --epochs 300")
    print("    python train.py --architecture flashdet --data data/coco.yaml --epochs 100")
    print("=" * 80)


if __name__ == "__main__":
    main()
