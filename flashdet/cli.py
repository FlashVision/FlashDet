"""FlashDet CLI — command-line interface for training, validation, prediction, export, and dataset download."""

import argparse
import os
import sys


def _colored(text, color):
    """Simple ANSI color helper."""
    colors = {"green": "\033[92m", "blue": "\033[94m", "yellow": "\033[93m", "red": "\033[91m", "bold": "\033[1m"}
    return f"{colors.get(color, '')}{text}\033[0m"


def _print_banner():
    print(_colored("FlashDet", "bold") + f" v{_get_version()}")
    print(_colored("Ultra-lightweight real-time object detection", "blue"))
    print()


def _get_version():
    from flashdet import __version__
    return __version__


def cmd_version(args):
    """Print version info."""
    _print_banner()


def cmd_settings(args):
    """Print system settings and environment info."""
    import torch
    import platform
    import numpy as np

    _print_banner()
    print(_colored("System", "bold"))
    print(f"  Python:      {platform.python_version()}")
    print(f"  OS:          {platform.system()} {platform.release()}")
    print(f"  Machine:     {platform.machine()}")
    print()
    print(_colored("Dependencies", "bold"))
    print(f"  PyTorch:     {torch.__version__}")
    print(f"  NumPy:       {np.__version__}")
    print(f"  CUDA:        {torch.version.cuda or 'Not available'}")
    print(f"  cuDNN:       {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
    print()
    print(_colored("Hardware", "bold"))
    if torch.cuda.is_available():
        print(f"  GPU:         {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        print(f"  VRAM:        {mem:.1f} GB")
    else:
        print("  GPU:         None (CPU only)")
    print(f"  CPU cores:   {__import__('os').cpu_count()}")


def cmd_check(args):
    """Verify installation — imports, GPU, and basic inference."""
    _print_banner()
    errors = []

    print(_colored("Checking installation...", "bold"))
    print()

    try:
        import flashdet  # noqa: F401
        print(f"  {_colored('✓', 'green')} flashdet package")
    except ImportError as e:
        print(f"  {_colored('✗', 'red')} flashdet package: {e}")
        errors.append(str(e))

    try:
        from flashdet.engine import Trainer, Validator  # noqa: F401
        print(f"  {_colored('✓', 'green')} engine (Trainer, Validator)")
    except ImportError as e:
        print(f"  {_colored('✗', 'red')} engine: {e}")
        errors.append(str(e))

    try:
        from flashdet.trackers import FlashTracker, MotionTracker, AppearanceTracker  # noqa: F401
        print(f"  {_colored('✓', 'green')} trackers (FlashTracker, MotionTracker, AppearanceTracker)")
    except ImportError as e:
        print(f"  {_colored('✗', 'red')} trackers: {e}")
        errors.append(str(e))

    try:
        from flashdet.solutions import ObjectCounter, SpeedEstimator, Heatmap  # noqa: F401
        print(f"  {_colored('✓', 'green')} solutions (ObjectCounter, SpeedEstimator, Heatmap, ...)")
    except ImportError as e:
        print(f"  {_colored('✗', 'red')} solutions: {e}")
        errors.append(str(e))

    try:
        from flashdet.analytics import Benchmark, Profiler  # noqa: F401
        print(f"  {_colored('✓', 'green')} analytics (Benchmark, Profiler)")
    except ImportError as e:
        print(f"  {_colored('✗', 'red')} analytics: {e}")
        errors.append(str(e))

    try:
        import torch
        from flashdet.cfg import get_config
        from flashdet.models import build_model
        cfg = get_config(model_size="m")
        model = build_model(cfg)
        model.eval()
        size = int(cfg.model.input_size[0])
        with torch.no_grad():
            model(torch.randn(1, 3, size, size))
        print(f"  {_colored('✓', 'green')} model forward pass (FlashDet-m, {size}px)")
    except Exception as e:
        print(f"  {_colored('✗', 'red')} model forward pass: {e}")
        errors.append(str(e))

    import torch
    if torch.cuda.is_available():
        print(f"  {_colored('✓', 'green')} CUDA ({torch.cuda.get_device_name(0)})")
    else:
        print(f"  {_colored('⚠', 'yellow')} No CUDA GPU (training will be slow)")

    print()
    if errors:
        print(_colored(f"✗ {len(errors)} check(s) failed", "red"))
        sys.exit(1)
    else:
        print(_colored("✓ All checks passed! FlashDet is ready.", "green"))


def cmd_download(args):
    """Download an open-source dataset."""
    from flashdet.data.download import download_dataset, list_datasets

    if args.list:
        _print_banner()
        datasets = list_datasets()
        print(_colored("Available datasets:", "bold"))
        print()
        for ds in datasets:
            print(f"  {_colored(ds['id'], 'green'):30s} {ds['name']}")
            print(f"  {'':30s} {ds['description']}")
            print(f"  {'':30s} Classes: {ds['classes']}, Format: {ds['format']}")
            print()
        return

    if not args.dataset:
        print(_colored("Error:", "red") + " --dataset is required (or use --list to see options)")
        sys.exit(1)

    download_dataset(
        dataset_id=args.dataset,
        output_dir=args.output,
        cache_dir=args.cache_dir,
    )


def cmd_train(args):
    """Train a FlashDet model."""
    if args.seed is not None:
        import random
        import numpy as np
        import torch
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    from flashdet.engine.training.trainer import Trainer

    if args.config:
        from flashdet.cfg import load_yaml_config
        cfg = load_yaml_config(args.config)
        print(f"{_colored('Config:', 'bold')} {args.config}")
        trainer = Trainer(
            config=cfg,
            device=args.device,
            skip_verify_annotations=getattr(args, "skip_verify_annotations", False),
        )
    else:
        if not args.train_images or not args.val_images:
            print(_colored("Error:", "red") + " --train-images and --val-images are required (or use --config)")
            sys.exit(1)
        kwargs = {
            "model_size": args.model_size,
            "architecture": args.architecture,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "device": args.device,
            "train_images": args.train_images,
            "val_images": args.val_images,
            "save_dir": args.save_dir,
            "backbone_type": getattr(args, "backbone", "lite"),
            "neck_type": getattr(args, "neck", None),
        }
        if args.lora:
            kwargs["lora"] = True
        if args.qlora:
            kwargs["qlora"] = True
        if args.amp:
            kwargs["amp"] = True
        if args.lr:
            kwargs["lr"] = args.lr
        if args.workers is not None:
            kwargs["workers"] = args.workers
        if args.mosaic:
            kwargs["mosaic"] = True
        if args.mixup:
            kwargs["mixup"] = True
        if getattr(args, "skip_verify_annotations", False):
            kwargs["skip_verify_annotations"] = True
        trainer = Trainer(**kwargs)

    trainer.train()


def cmd_predict(args):
    """Run inference on an image, video, or directory."""
    from flashdet.engine.inference import Predictor

    predictor = Predictor(
        model_path=args.model,
        device=args.device,
        conf_thresh=args.conf,
        nms_thresh=args.nms,
        input_size=args.input_size,
    )

    results = predictor.predict_image(args.source)
    if not results:
        print("No detections found.")
        return

    for det in results:
        bbox = det["bbox"]
        print(f"  {det['class_name']}: {det['confidence']:.2f} "
              f"[{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")
    print(f"\nTotal: {len(results)} detections")

    if args.output:
        import os
        import cv2
        from pathlib import Path
        os.makedirs(args.output, exist_ok=True)
        image = cv2.imread(args.source)
        if image is not None:
            for det in results:
                bbox = det["bbox"]
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                label = f"{det['class_name']} {det['confidence']:.2f}"
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            out_path = os.path.join(args.output, Path(args.source).name)
            cv2.imwrite(out_path, image)
            print(f"Saved annotated image to {out_path}")


def cmd_val(args):
    """Validate model on a dataset."""
    from flashdet.engine.evaluation.validator import Validator
    validator = Validator(
        model_path=args.model,
        val_images=args.val_images,
        device=args.device,
        conf_thresh=args.conf,
        nms_thresh=args.nms,
        input_size=args.input_size,
        batch_size=args.batch_size,
    )
    results = validator.validate()
    print(f"mAP@{validator.iou_threshold:.2f}: {results['mAP']:.4f}")
    print(f"Validation loss: {results['val_loss']:.4f}")
    coco = results.get("coco_summary")
    if coco and coco.get("map5095") is not None:
        print(f"mAP@0.5:0.95: {coco['map5095']:.4f}")
    if results.get("AP_per_class"):
        print("Per-class AP:")
        for cid, ap in sorted(results["AP_per_class"].items()):
            name = validator.class_names[cid] if cid < len(validator.class_names) else str(cid)
            print(f"  {name}: {ap:.4f}")


def cmd_benchmark(args):
    """Benchmark speed (and optionally accuracy) for a checkpoint."""
    from flashdet.benchmark import benchmark_checkpoint, save_report

    summary = benchmark_checkpoint(
        args.model,
        input_size=args.input_size,
        device=args.device,
        warmup=args.warmup,
        repeats=args.repeats,
        fuse=args.fuse,
        val_images=args.val_images,
        conf=args.conf,
    )
    print(f"\nBenchmark: {summary['model']}")
    print(f"  Architecture:       {summary['architecture']}")
    print(f"  Input size:         {summary['input_size']}x{summary['input_size']}")
    print(f"  Params:             {summary['params']:,}")
    print(f"  FP16 weights:       {summary['params_fp16_mb']:.2f} MB")
    print(f"  Latency (median):   {summary['latency_ms']:.2f} ms/image")
    print(f"  Throughput:         {summary['throughput_fps']:.1f} FPS")
    evals = summary.get("eval")
    if evals:
        print(f"  mAP@0.5:            {evals['mAP50']*100:.2f}%")
        if evals.get("mAP5095") is not None:
            print(f"  mAP@0.5:0.95:       {evals['mAP5095']*100:.2f}%")
        per_class = evals.get("AP_per_class")
        if per_class:
            print("  Per-class AP:")
            for cid, ap in sorted(per_class.items()):
                print(f"    {cid}: {ap:.4f}")
    out = save_report(summary, args.output)
    if out:
        print(f"\nReport saved: {out}")


def cmd_export(args):
    """Export model to ONNX (optionally INT8-quantized), with parity check."""
    from flashdet.deployment import (
        export_onnx, check_parity, onnx_quantize_int8, detection_parity,
    )

    output_path = export_onnx(
        args.model,
        args.output,
        fuse=args.fuse,
        simplify=args.simplify,
    )
    print(f"Exported to {output_path}")
    if args.check_parity:
        try:
            report = check_parity(args.model, output_path)
            state = "OK" if report["ok"] else "OUT OF TOLERANCE"
            print(f"Parity check: {state} "
                  f"(max|Δ|={report['max_abs_diff']:.5f}, "
                  f"mean|Δ|={report['mean_abs_diff']:.6f})")
        except ImportError as exc:
            print(f"Parity check skipped: {exc}")
    if args.int8:
        if not args.calib_images:
            print("INT8 skipped: --calib-images <dir> is required")
        else:
            int8_path = onnx_quantize_int8(
                output_path,
                calib_images=args.calib_images,
                max_batches=args.calib_batches,
            )
            size_onnx = os.path.getsize(output_path) / (1024 * 1024)
            size_int8 = os.path.getsize(int8_path) / (1024 * 1024)
            print(f"INT8 model: {int8_path}")
            print(f"  FP16/FP32 ONNX: {size_onnx:.2f} MB -> INT8: {size_int8:.2f} MB")
            try:
                q_report = check_parity(args.model, int8_path, rtol=5e-2, atol=5e-2)
                print(f"  INT8 vs PyTorch tensors: max|Δ|={q_report['max_abs_diff']:.4f}, "
                      f"mean|Δ|={q_report['mean_abs_diff']:.5f} (raw tensors can differ)")
                det_report = detection_parity(args.model, int8_path)
                print(f"  INT8 box IoU vs PyTorch: mean={det_report['mean_box_iou']:.3f} "
                      f"(min={det_report['min_box_iou']:.3f}) — ≥0.9 is production-ready")
            except ImportError as exc:
                print(f"  INT8 parity skipped: {exc}")


def cmd_fuse(args):
    """Fuse layers/kernels for deployment and report parity."""
    import torch
    from flashdet.deployment import (
        build_model_from_checkpoint,
        input_size_from_checkpoint,
    )
    from flashdet.fusing import default_passes, run_fusions
    from flashdet.utils.checkpoint import safe_torch_load

    model = build_model_from_checkpoint(args.model, fuse=False)
    size = args.input_size or input_size_from_checkpoint(args.model)
    probe = torch.randn(1, 3, size, size)

    passes = default_passes(example_input=probe)
    if args.no_rep:
        passes = [p for p in passes if p.name != "rep_fusion"]

    fused, report = run_fusions(
        model, passes, example_input=probe, verify=not args.no_verify)

    print(f"Fusing: {args.model} (input {size}x{size})")
    print(report)
    if args.no_verify:
        print("  parity: skipped (--no-verify)")
    elif not report.parity_ok:
        print("  warning: fused output differs from the original — inspect "
              "the report before deploying")

    if args.output:
        raw = safe_torch_load(args.model, map_location="cpu")
        config = raw.get("config", {}) if isinstance(raw, dict) else {}
        if not config:
            print("  warning: checkpoint has no config; the saved fused file "
                  "may not rebuild correctly")
        from flashdet.fusing.checkpoint import save_fused_checkpoint
        path = save_fused_checkpoint(
            fused, args.output, config=config or {},
            input_size=size, report=report,
        )
        print(f"Fused checkpoint: {path}")


def cmd_weights(args):
    """List / download registered demo checkpoints."""
    from flashdet.hub import list_weights, WEIGHTS, download_weights

    if args.list or not args.name:
        print("Available checkpoints:")
        for name in list_weights():
            item = WEIGHTS[name]
            print(f"\n  {name}")
            print(f"    {item['description']}")
            print(f"    Classes: {item.get('num_classes', '?')}")
        if not args.name:
            print("\nDownload with: flashdet weights --name <name>")
            return
    path = download_weights(args.name, save_dir=args.save_dir, force=args.force)
    print(f"Ready: {path}")


def cmd_datasets(args):
    """List available datasets and show dataset info."""
    cmd_download_args = argparse.Namespace(list=True, dataset=None, output=None, cache_dir=None)
    cmd_download(cmd_download_args)


def main():
    parser = argparse.ArgumentParser(
        prog="flashdet",
        description="FlashDet: Ultra-lightweight real-time object detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  flashdet check                              Verify installation
  flashdet download --list                    List available datasets
  flashdet download --dataset coco2017        Download COCO 2017 dataset
  flashdet download --dataset sample          Download tiny sample for testing
  flashdet train --train-images data/train --val-images data/val
  flashdet train --config configs/flashdet_n_320_coco.yaml
  flashdet train --config configs/yolov10_s_640_coco.yaml
  flashdet fuse --model weights/flashdet_p.pth --output fused.pth

Documentation: https://github.com/FlashVision/FlashDet
""",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # version
    subparsers.add_parser("version", help="Show version info")

    # settings
    subparsers.add_parser("settings", help="Show system settings (Python, PyTorch, CUDA, GPU)")

    # check
    subparsers.add_parser("check", help="Verify installation and run health check")

    # download
    dl_p = subparsers.add_parser("download", help="Download open-source datasets (COCO, VOC, etc.)")
    dl_p.add_argument("--list", action="store_true", help="List all available datasets")
    dl_p.add_argument("--dataset", default=None,
                       help="Dataset ID to download (e.g. coco2017, voc2007, sample)")
    dl_p.add_argument("--output", default=None,
                       help="Output directory (default: data/<dataset>)")
    dl_p.add_argument("--cache-dir", default=None,
                       help="Cache directory for archives (default: ~/.cache/flashdet/)")

    # datasets (alias for download --list)
    subparsers.add_parser("datasets", help="List available datasets for download")

    # train
    train_p = subparsers.add_parser("train", help="Train a FlashDet model")
    train_p.add_argument("--config", default=None, help="Path to YAML config (e.g. configs/flashdet_n_320_coco.yaml)")
    train_p.add_argument("--model-size", default="n", choices=["p", "n", "s", "m", "l", "x"],
                         help="Model variant (default: n)")
    train_p.add_argument("--architecture", default="flashdet",
                         choices=["flashdet", "yolov8", "yolov9", "yolov10", "yolov11", "yolox"],
                         help="Detection architecture (default: flashdet)")
    train_p.add_argument("--backbone", default=None,
                         help="Pico backbone (lite, pico_v2, or any Pico-Zoo name such as repvit/edgenext/mambaout)")
    train_p.add_argument("--neck", default=None,
                         help="Pico neck family (pan, rep, ghost, attention, mamba, lcnet) or a backbone name")
    train_p.add_argument("--epochs", type=int, default=100, help="Training epochs (default: 100)")
    train_p.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    train_p.add_argument("--lr", type=float, default=None, help="Learning rate")
    train_p.add_argument("--device", default="cuda", help="Device: cuda or cpu (default: cuda)")
    train_p.add_argument("--train-images", default=None, help="Path to training images")
    train_p.add_argument("--val-images", default=None, help="Path to validation images")
    train_p.add_argument("--save-dir", default="workspace/flashdet_output", help="Output directory")
    train_p.add_argument("--workers", type=int, default=None, help="DataLoader workers")
    train_p.add_argument("--lora", action="store_true", help="Enable LoRA fine-tuning")
    train_p.add_argument("--qlora", action="store_true", help="Enable QLoRA fine-tuning")
    train_p.add_argument("--amp", action="store_true", help="Enable mixed precision (FP16)")
    train_p.add_argument("--mosaic", action="store_true", help="Enable mosaic augmentation")
    train_p.add_argument("--mixup", action="store_true", help="Enable MixUp augmentation")
    train_p.add_argument("--skip-verify-annotations", action="store_true",
                         help="Skip pre-training dataloader/GT verification")
    train_p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    # predict
    pred_p = subparsers.add_parser("predict", help="Run inference on image/video/directory")
    pred_p.add_argument("--model", required=True, help="Path to .pth checkpoint")
    pred_p.add_argument("--source", required=True, help="Image path, video path, or directory")
    pred_p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    pred_p.add_argument("--nms", type=float, default=0.5, help="NMS IoU threshold (default: 0.5)")
    pred_p.add_argument("--input-size", type=int, default=None,
                        help="Input size (default: from checkpoint config)")
    pred_p.add_argument("--device", default="cuda", help="Device (default: cuda)")
    pred_p.add_argument("--output", default=None, help="Output directory for annotated results")

    # val
    val_p = subparsers.add_parser("val", help="Validate model on dataset")
    val_p.add_argument("--model", required=True, help="Path to .pth checkpoint")
    val_p.add_argument("--val-images", required=True, help="Path to validation images")
    val_p.add_argument("--device", default="cuda", help="Device (default: cuda)")
    val_p.add_argument("--conf", type=float, default=0.01, help="Confidence threshold (default: 0.01, matches training validation)")
    val_p.add_argument("--nms", type=float, default=0.6, help="NMS IoU threshold (default: 0.6)")
    val_p.add_argument("--input-size", type=int, default=None,
                       help="Input size (default: from checkpoint/config)")
    val_p.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")

    # export
    exp_p = subparsers.add_parser("export", help="Export model to ONNX format")
    exp_p.add_argument("--model", required=True, help="Path to .pth checkpoint")
    exp_p.add_argument("--output", default=None, help="Output path (default: alongside the checkpoint)")
    exp_p.add_argument("--simplify", action="store_true", help="Simplify ONNX graph (onnxslim or onnxsim)")
    exp_p.add_argument("--no-fuse", dest="fuse", action="store_false",
                       help="Skip reparameterization fusion before export")
    exp_p.add_argument("--check-parity", dest="check_parity", action="store_true", default=True,
                       help=argparse.SUPPRESS)
    exp_p.add_argument("--no-parity", dest="check_parity", action="store_false",
                       help="Skip the PyTorch vs ONNXRuntime parity check")
    exp_p.add_argument("--int8", action="store_true",
                       help="Also produce an INT8-quantized ONNX (requires --calib-images)")
    exp_p.add_argument("--calib-images", default=None,
                       help="Folder of calibration images for INT8 quantization")
    exp_p.add_argument("--calib-batches", type=int, default=16,
                       help="Calibration batches consumed (default: 16)")

    # fuse
    fuse_p = subparsers.add_parser(
        "fuse", help="Fuse layers/kernels for deployment (with parity check)")
    fuse_p.add_argument("--model", required=True, help="Path to .pth checkpoint")
    fuse_p.add_argument("--input-size", type=int, default=None,
                        help="Probe input size (default: from checkpoint config)")
    fuse_p.add_argument("--output", default=None,
                        help="Save the fused model as a replayable checkpoint")
    fuse_p.add_argument("--no-rep", action="store_true",
                        help="Skip structural reparameterization (rep blocks)")
    fuse_p.add_argument("--no-verify", dest="no_verify", action="store_true",
                        help="Skip the numeric parity verification")

    # weights
    w_p = subparsers.add_parser("weights", help="List or download ready-to-use checkpoints")
    w_p.add_argument("--list", action="store_true", help="List available checkpoints")
    w_p.add_argument("--name", default=None,
                     help="Registry name to download (e.g. p-demo-synth10)")
    w_p.add_argument("--save-dir", default=None, help="Download directory (default: ./weights)")
    w_p.add_argument("--force", action="store_true", help="Re-download even if present")

    # benchmark
    bench_p = subparsers.add_parser("benchmark", help="Benchmark speed and accuracy for a checkpoint")
    bench_p.add_argument("--model", required=True, help="Path to .pth checkpoint")
    bench_p.add_argument("--input-size", type=int, default=None,
                         help="Input size (default: from checkpoint config)")
    bench_p.add_argument("--device", default="cpu", help="Device (default: cpu)")
    bench_p.add_argument("--warmup", type=int, default=20, help="Warmup forwards (default: 20)")
    bench_p.add_argument("--repeats", type=int, default=100, help="Measured forwards (default: 100)")
    bench_p.add_argument("--no-fuse", dest="fuse", action="store_false",
                         help="Skip the reparameterization fusion step")
    bench_p.add_argument("--val-images", default=None,
                         help="Optional COCO-format validation split to also report mAP")
    bench_p.add_argument("--conf", type=float, default=0.05,
                         help="Confidence threshold for the accuracy pass (default: 0.05)")
    bench_p.add_argument("--output", default=None, help="Save the report as JSON")

    args = parser.parse_args()

    if args.command is None:
        _print_banner()
        parser.print_help()
        sys.exit(0)

    commands = {
        "version": cmd_version,
        "settings": cmd_settings,
        "check": cmd_check,
        "download": cmd_download,
        "datasets": cmd_datasets,
        "train": cmd_train,
        "predict": cmd_predict,
        "val": cmd_val,
        "export": cmd_export,
        "fuse": cmd_fuse,
        "benchmark": cmd_benchmark,
        "weights": cmd_weights,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
