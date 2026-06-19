#!/usr/bin/env python3
"""
Train FlashDet on PPE Detection Dataset.

Usage:
    python train.py
    python train.py --epochs 200 --batch-size 32
    python train.py --resume workspace/ppe_detector/checkpoint_last.pth
"""

import os
import sys
import argparse
import time
import json
import math
import cv2
import numpy as np

import copy
import torch

from flashdet.cfg import get_config
from flashdet.models import FlashDet, load_coco_pretrained
from flashdet.models.lora import apply_lora, apply_qlora, merge_lora_weights, get_lora_state_dict
from flashdet.data import create_dataloader, verify_dataset
from flashdet.utils import save_checkpoint, load_checkpoint, save_weights_only, save_inference_weights, setup_logger, AverageMeter
from flashdet.utils.metrics import compute_map
from flashdet.utils.torchtune_optim import (
    apply_activation_checkpointing,
    ActivationOffloadHook,
    create_optimizer,
    compile_model as torchtune_compile,
    log_memory_stats,
)


class ModelEMA:
    """Exponential Moving Average of model weights.

    Uses an adaptive decay schedule so the EMA converges quickly even with
    small datasets (few batches/epoch).  The effective decay ramps from ~0
    (EMA = model copy) up to *target_decay* over the first few thousand
    updates, using:

        effective_decay = min(target_decay, (1 + n) / (warmup + n))

    With the default warmup=2000, the decay reaches 0.9998 at ~10 000
    updates.  For a 30-batch/epoch dataset, this means the EMA is already
    well-converged after ~50 epochs instead of lagging behind for hundreds.
    """
    def __init__(self, model: torch.nn.Module, decay: float = 0.9998,
                 warmup: int = 2000):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.target_decay = decay
        self.warmup = warmup
        self.num_updates = 0
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @property
    def decay(self):
        return min(self.target_decay,
                   (1 + self.num_updates) / (self.warmup + self.num_updates))

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        self.num_updates += 1
        d = self.decay
        for ema_p, model_p in zip(self.ema.parameters(), model.parameters()):
            ema_p.data.mul_(d).add_(model_p.data, alpha=1.0 - d)
        for ema_b, model_b in zip(self.ema.buffers(), model.buffers()):
            ema_b.copy_(model_b)

    def state_dict(self):
        return {
            "ema_state": self.ema.state_dict(),
            "target_decay": self.target_decay,
            "warmup": self.warmup,
            "num_updates": self.num_updates,
        }

    def load_state_dict(self, state: dict):
        missing, unexpected = self.ema.load_state_dict(
            state["ema_state"], strict=False
        )
        if missing:
            import logging
            logging.getLogger(__name__).debug(
                "EMA missing keys (%d): %s%s",
                len(missing), missing[:3], "..." if len(missing) > 3 else ""
            )
        self.target_decay = state.get("target_decay",
                                      state.get("decay", self.target_decay))
        self.warmup = state.get("warmup", self.warmup)
        self.num_updates = state.get("num_updates", 0)


def _make_color_palette(n: int):
    """Generate a deterministic BGR color palette for N classes."""
    import colorsys
    palette = {}
    for i in range(n):
        hue = i / max(n, 1)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.9)
        palette[i] = (int(b * 255), int(g * 255), int(r * 255))  # BGR
    return palette


def _load_class_names_from_ann(ann_file: str):
    """Read class names from a COCO annotation file (order = sorted category IDs)."""
    try:
        with open(ann_file) as f:
            ann = json.load(f)
        cats = ann.get("categories", [])
        if not cats:
            return []
        cat_ids = sorted(c["id"] for c in cats)
        id_to_name = {c["id"]: c["name"] for c in cats}
        return [id_to_name[cid] for cid in cat_ids]
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return []


def save_visualization(model, images, gt_meta, save_path, epoch, batch_idx, device, config,
                       class_names=None, colors=None):
    """Save a GT-vs-Predictions panel for the first image in the batch."""
    from flashdet.utils.visualization import make_gt_pred_panel

    pred_model = model.module if hasattr(model, "module") else model
    pred_model.eval()
    try:
        with torch.no_grad():
            results = pred_model.predict(images, None, score_thr=0.3, nms_thr=0.5)
    except Exception:
        results = []
    pred_model.train()

    # Denormalise the first image (ImageNet RGB stats)
    img = images[0].cpu().numpy().transpose(1, 2, 0)  # CHW → HWC
    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([58.395, 57.12, 57.375])
    img = np.clip(img * std + mean, 0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Ground truth arrays
    gt_boxes = gt_labels = np.empty(0)
    if gt_meta and "gt_bboxes" in gt_meta and len(gt_meta["gt_bboxes"]) > 0:
        gt_boxes = gt_meta["gt_bboxes"][0]
        gt_labels = gt_meta["gt_labels"][0]
        if not isinstance(gt_boxes, np.ndarray) or len(gt_boxes) == 0:
            gt_boxes = np.empty((0, 4))
            gt_labels = np.empty(0)

    # Prediction arrays
    pred_boxes = np.empty((0, 4))
    pred_labels = np.empty(0, dtype=int)
    pred_scores = np.empty(0)
    if results and len(results) > 0:
        dets, lbs = results[0]
        if dets is not None and dets.numel() > 0:
            dets_np = dets.cpu().numpy()
            pred_boxes = dets_np[:, :4]
            pred_scores = dets_np[:, 4]
            pred_labels = lbs.cpu().numpy().astype(int)

    # Build colour dict keyed by class name for the panel renderer
    color_map = {}
    if class_names and colors:
        for idx, cname in enumerate(class_names):
            color_map[cname] = colors.get(idx, (255, 255, 255))

    panel = make_gt_pred_panel(
        img_bgr,
        gt_boxes, gt_labels.astype(int) if len(gt_labels) else gt_labels,
        pred_boxes, pred_labels, pred_scores,
        class_names=class_names,
        colors=color_map or None,
        title_extra=f"| Epoch {epoch}  Batch {batch_idx}",
    )

    # Save as RGB via PIL for correct colour in browsers / UI
    from PIL import Image
    panel_rgb = cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)
    Image.fromarray(panel_rgb).save(save_path, quality=95)

    latest_path = os.path.join(os.path.dirname(save_path), "latest_visualization.jpg")
    Image.fromarray(panel_rgb).save(latest_path, quality=95)


def train_one_epoch(model, dataloader, optimizer, device, epoch, logger, save_dir=None, config=None, ema=None,
                    class_names=None, colors=None, scaler=None, grad_accum=1):
    """Train for one epoch with optional AMP and gradient accumulation."""
    model.train()
    use_amp = scaler is not None

    loss_meter = AverageMeter("Loss")
    qfl_meter = AverageMeter("QFL")
    bbox_meter = AverageMeter("BBox")
    dfl_meter = AverageMeter("DFL")

    start_time = time.time()
    vis_dir = os.path.join(save_dir, "visualizations") if save_dir else None
    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)
        try:
            vis_files = sorted([f for f in os.listdir(vis_dir) if f.endswith('.jpg') and f != 'latest_visualization.jpg'])
            if len(vis_files) > 10:
                for old_file in vis_files[:-10]:
                    os.remove(os.path.join(vis_dir, old_file))
        except OSError:
            pass

    # Access underlying model for DataParallel
    raw_model = model.module if hasattr(model, 'module') else model

    zero_pos_batches = 0
    total_batches = 0

    for batch_idx, (images, gt_meta) in enumerate(dataloader):
        images = images.to(device)

        with torch.amp.autocast(device.type, enabled=use_amp):
            output = model(images, gt_meta, epoch=epoch)
            loss = output["loss"] / grad_accum

        loss_states = output["loss_states"]
        total_batches += 1
        if loss_states["num_pos"].item() == 0:
            zero_pos_batches += 1

        if torch.isnan(loss):
            logger.warning(f"NaN loss at batch {batch_idx}, skipping")
            continue

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Step optimizer every grad_accum batches (or on last batch)
        if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(dataloader):
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 35.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 35.0)
                optimizer.step()
            optimizer.zero_grad()

            if ema is not None:
                ema.update(raw_model)
        
        # Update meters (use unscaled loss for logging; `loss` is divided by grad_accum for backward)
        loss_meter.update(output["loss"].item())
        qfl_meter.update(loss_states["loss_qfl"].item())
        bbox_meter.update(loss_states["loss_bbox"].item())
        dfl_meter.update(loss_states["loss_dfl"].item())
        
        # Save visualization every 20 batches for more frequent updates
        if vis_dir and (batch_idx + 1) % 20 == 0:
            try:
                vis_path = os.path.join(vis_dir, f"epoch{epoch}_batch{batch_idx+1}.jpg")
                save_visualization(model, images, gt_meta, vis_path, epoch, batch_idx + 1, device, config,
                                   class_names=class_names, colors=colors)
                
                # Clean up old visualizations (keep only latest 10 images)
                vis_files = sorted([f for f in os.listdir(vis_dir) if f.endswith('.jpg') and f != 'latest_visualization.jpg'])
                if len(vis_files) > 10:
                    for old_file in vis_files[:-10]:
                        try:
                            os.remove(os.path.join(vis_dir, old_file))
                        except OSError:
                            pass
            except Exception as e:
                logger.warning(f"Failed to save visualization: {e}")
        
        # Log every 10 batches for real-time dashboard updates
        if (batch_idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            logger.info(
                f"Epoch [{epoch}] Batch [{batch_idx+1}/{len(dataloader)}] "
                f"Loss: {loss_meter.avg:.4f} (QFL: {qfl_meter.avg:.4f}, "
                f"BBox: {bbox_meter.avg:.4f}, DFL: {dfl_meter.avg:.4f}) "
                f"Pos: {loss_states['num_pos'].item():.0f} "
                f"Time: {elapsed:.1f}s"
            )
            # Flush to ensure dashboard sees updates immediately
            sys.stdout.flush()
    
    if zero_pos_batches > 0:
        logger.warning(
            f"DIAGNOSTIC: {zero_pos_batches}/{total_batches} batches had ZERO positive "
            f"assignments. If this is > 50%, the assigner cannot match predictions to GT "
            f"— loss WILL NOT decrease. Use --pretrained-coco or train without --lora."
        )

    return {
        "loss": loss_meter.avg,
        "loss_qfl": qfl_meter.avg,
        "loss_bbox": bbox_meter.avg,
        "loss_dfl": dfl_meter.avg
    }


@torch.no_grad()
def validate(model, dataloader, device, logger, ema=None, class_names=None):
    """
    Validate model — computes both loss and mAP@0.5.

    Uses EMA weights when provided (they give better accuracy).
    Main model is always restored to train() mode on exit.

    Returns:
        (val_loss, map50) — both as floats.
    """
    eval_model = ema.ema if ema is not None else model
    eval_model.eval()

    loss_meter = AverageMeter("Loss")
    qfl_meter  = AverageMeter("QFL")
    bbox_meter = AverageMeter("BBox")
    dfl_meter  = AverageMeter("DFL")

    all_preds = []
    all_gts   = []

    for images, gt_meta in dataloader:
        images = images.to(device)

        # ---- loss pass ----
        out = eval_model(images, gt_meta, epoch=0, compute_loss=True)
        loss_states = out["loss_states"]
        loss_meter.update(out["loss"].item())
        qfl_meter.update(loss_states["loss_qfl"].item())
        bbox_meter.update(loss_states["loss_bbox"].item())
        dfl_meter.update(loss_states["loss_dfl"].item())

        # ---- prediction pass for mAP (low threshold to capture all detections) ----
        results = eval_model.predict(images, None, score_thr=0.05, nms_thr=0.6)

        for i, (dets, lbs) in enumerate(results):
            gt_boxes  = gt_meta["gt_bboxes"][i]   # np.ndarray [N,4]
            gt_labels = gt_meta["gt_labels"][i]   # np.ndarray [N]

            if dets is not None and dets.numel() > 0:
                boxes_np  = dets[:, :4].cpu().numpy()
                scores_np = dets[:, 4].cpu().numpy()
                lbs_np    = lbs.cpu().numpy()
            else:
                boxes_np  = np.zeros((0, 4), dtype=np.float32)
                scores_np = np.zeros(0, dtype=np.float32)
                lbs_np    = np.zeros(0, dtype=np.int64)

            all_preds.append({"boxes": boxes_np, "scores": scores_np, "labels": lbs_np})
            all_gts.append({"boxes": gt_boxes, "labels": gt_labels})

    # ---- mAP @IoU=0.5 ----
    num_cls = len(class_names) if class_names else 10
    map_results = compute_map(all_preds, all_gts, iou_threshold=0.5, num_classes=num_cls)
    map50 = map_results["mAP"]

    # Per-class AP summary (only classes that have GT instances)
    ap_per_cls = map_results.get("AP_per_class", {})
    if class_names and ap_per_cls:
        per_cls_str = "  ".join(
            f"{class_names[cid]}={v:.3f}"
            for cid, v in sorted(ap_per_cls.items())
            if cid < len(class_names)
        )
        logger.info(f"  AP per class: {per_cls_str}")

    logger.info(
        f"Validation - Loss: {loss_meter.avg:.4f} "
        f"(QFL: {qfl_meter.avg:.4f}, BBox: {bbox_meter.avg:.4f}, DFL: {dfl_meter.avg:.4f}) "
        f"| mAP@0.5: {map50:.4f}"
    )

    # Always restore main model to train mode
    model.train()

    return loss_meter.avg, map50


def main():
    parser = argparse.ArgumentParser(description="Train FlashDet")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--workers", type=int, default=4, help="Data workers")
    parser.add_argument("--save-dir", default="workspace/ppe_detector", help="Save directory")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Warmup epochs")
    parser.add_argument("--patience", type=int, default=50,
                        help="Early stopping patience (epochs without mAP improvement). 0 disables.")
    parser.add_argument("--model-size", default="m", choices=["m", "m-1.5x", "m-0.5x"],
                        help="Model size: m (~1.17M params), m-1.5x (~2.44M params), m-0.5x (~0.5M ultra-lite)")
    parser.add_argument("--input-size", type=int, default=320, help="Input image size (320 or 416)")
    parser.add_argument("--finetune", default=None,
                        help="Path to a previous model checkpoint (inference or training) to fine-tune from. "
                             "Loads model weights only — optimizer/scheduler start fresh from epoch 0. "
                             "Handles FP16 checkpoints and missing aux_head keys automatically.")
    parser.add_argument("--pretrained-coco", action="store_true",
                        help="Load official FlashDet COCO pretrained weights for fine-tuning "
                             "(backbone + FPN + head regression). Much better than training from scratch.")
    parser.add_argument("--pretrained-ckpt", default=None,
                        help="Path to a local FlashDet COCO checkpoint file (overrides auto-download)")
    parser.add_argument("--class-file", default=None,
                        help="Path to a .txt file with class names (one per line). "
                             "Overrides annotation-based auto-detection.")
    parser.add_argument("--train-images", default=None,
                        help="Path to train images directory (overrides config)")
    parser.add_argument("--val-images", default=None,
                        help="Path to validation images directory (overrides config)")
    parser.add_argument("--amp", action="store_true",
                        help="Enable Automatic Mixed Precision (FP16) training")
    parser.add_argument("--multi-gpu", action="store_true",
                        help="Use all visible GPUs via DataParallel")
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")

    # --- torchtune-inspired training optimizations ---
    tt_group = parser.add_argument_group("torchtune optimizations",
                                          "Memory & performance techniques from torchtune")
    tt_group.add_argument("--activation-checkpointing", action="store_true",
                          help="Enable gradient/activation checkpointing (trade compute for memory)")
    tt_group.add_argument("--activation-offloading", action="store_true",
                          help="Offload activations to CPU during forward pass")
    tt_group.add_argument("--optimizer-in-bwd", action="store_true",
                          help="Fuse optimizer step into backward pass (reduces peak memory)")
    tt_group.add_argument("--use-8bit-optimizer", action="store_true",
                          help="Use bitsandbytes 8-bit AdamW (halves optimizer memory)")
    tt_group.add_argument("--compile", action="store_true",
                          help="Apply torch.compile for faster training (requires PyTorch >= 2.0)")
    tt_group.add_argument("--chunked-loss", action="store_true",
                          help="Compute focal/DFL losses in chunks for lower peak memory")
    tt_group.add_argument("--chunk-size", type=int, default=1024,
                          help="Chunk size for chunked loss computation (default: 1024)")

    # --- LoRA ---
    lora_group = parser.add_argument_group("LoRA", "Low-Rank Adaptation for efficient fine-tuning")
    lora_group.add_argument("--lora", action="store_true",
                            help="Enable LoRA fine-tuning (freezes backbone, trains low-rank adapters)")
    lora_group.add_argument("--lora-variant", default="standard",
                            choices=["standard", "dora", "lora_plus", "adalora", "ortho", "lora_fa"],
                            help="LoRA variant: standard, dora (weight-decomposed), "
                                 "lora_plus (asymmetric LR), adalora (adaptive rank), "
                                 "ortho (orthogonal), lora_fa (freeze A)")
    lora_group.add_argument("--lora-rank", type=int, default=8,
                            help="LoRA rank (default: 8)")
    lora_group.add_argument("--lora-alpha", type=float, default=16.0,
                            help="LoRA scaling alpha (default: 16.0)")
    lora_group.add_argument("--lora-dropout", type=float, default=0.05,
                            help="LoRA dropout (default: 0.05)")
    lora_group.add_argument("--lora-targets", nargs="+", default=["backbone", "fpn"],
                            help="Module names to apply LoRA to (default: backbone fpn)")
    lora_group.add_argument("--qlora", action="store_true",
                            help="Enable QLoRA (quantized base weights + LoRA adapters)")
    lora_group.add_argument("--qlora-dtype", default="int8", choices=["int8", "nf4"],
                            help="QLoRA quantization dtype (int8=no deps, nf4=requires bitsandbytes)")

    args = parser.parse_args()
    
    # Map model size to backbone config (matching official FlashDet specs)
    # Official FlashDet-m:     1.0x backbone, 96 fpn  -> ~1.17M params, 2.3MB FP16
    # Official FlashDet-m-1.5x: 1.5x backbone, 128 fpn -> ~2.44M params, 4.7MB FP16
    # Custom FlashDet-m-0.5x:  0.5x backbone, 96 fpn  -> ~0.49M params (ultra-lite)
    MODEL_SIZE_MAP = {
        "m": {"backbone": "1.0x", "fpn_channels": 96},        # Official FlashDet-m
        "m-1.5x": {"backbone": "1.5x", "fpn_channels": 128},  # Official FlashDet-m-1.5x
        "m-0.5x": {"backbone": "0.5x", "fpn_channels": 96},   # Ultra-lite version
    }
    model_cfg = MODEL_SIZE_MAP[args.model_size]
    input_size = (args.input_size, args.input_size)
    
    # Setup
    os.makedirs(args.save_dir, exist_ok=True)
    logger = setup_logger("FlashDet", args.save_dir)
    requested_device = args.device
    if torch.cuda.is_available():
        device = torch.device(requested_device)
    else:
        device = torch.device("cpu")
        req = str(requested_device).strip().lower()
        if req not in ("cpu", ""):
            logger.warning(
                "CUDA is not available; requested device %r was ignored, using CPU.",
                requested_device,
            )
    
    config = get_config()

    # Override data paths from CLI if provided
    if args.train_images:
        config.data.train_images = args.train_images
        config.data.train_annotations = os.path.join(args.train_images, "_annotations.coco.json")
    if args.val_images:
        config.data.val_images = args.val_images
        config.data.val_annotations = os.path.join(args.val_images, "_annotations.coco.json")

    # Resolve class names: explicit file > annotation JSON > config fallback
    class_names = None
    if args.class_file:
        with open(args.class_file, encoding="utf-8") as _cf:
            class_names = [l.strip() for l in _cf if l.strip()]
    if not class_names:
        class_names = _load_class_names_from_ann(config.data.train_annotations)
    if not class_names:
        class_names = config.class_names
    colors = _make_color_palette(len(class_names))
    num_classes = len(class_names)

    logger.info("=" * 60)
    logger.info("FlashDet Training")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Model Size: {args.model_size} (backbone: {model_cfg['backbone']}, fpn: {model_cfg['fpn_channels']})")
    logger.info(f"Input Size: {input_size}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Learning Rate: {args.lr}")
    logger.info(f"Save Dir: {args.save_dir}")
    logger.info(f"Classes ({num_classes}): {class_names}")
    
    # Verify dataset
    data_root = os.path.dirname(os.path.normpath(config.data.train_images))
    if not verify_dataset(data_root):
        logger.error("Dataset not found!")
        logger.error("Please download the dataset first:")
        logger.error("  Option 1: Download from Roboflow")
        logger.error("  Option 2: Download from Kaggle using: python src/data/prepare.py")
        sys.exit(1)
    
    # Create data loaders
    logger.info("\nLoading datasets...")
    train_loader = create_dataloader(
        img_dir=config.data.train_images,
        ann_file=config.data.train_annotations,
        batch_size=args.batch_size,
        input_size=input_size,
        num_workers=args.workers,
        is_train=True
    )
    
    val_loader = create_dataloader(
        img_dir=config.data.val_images,
        ann_file=config.data.val_annotations,
        batch_size=args.batch_size,
        input_size=input_size,
        num_workers=args.workers,
        is_train=False
    )
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    
    # Create model
    logger.info("\nBuilding model...")
    model = FlashDet(
        num_classes=num_classes,
        input_size=input_size,
        backbone_size=model_cfg["backbone"],
        fpn_channels=model_cfg["fpn_channels"],
        pretrained=config.model.backbone_pretrained,
        use_aux_head=True
    ).to(device)
    
    info = model.get_model_info()
    logger.info(f"Model: {info['name']}")
    logger.info(f"Inference params: {info['inference_params']:,} "
                f"({info['inference_params_mb']:.2f} MB FP32, "
                f"{info['inference_fp16_mb']:.2f} MB FP16)")
    logger.info(f"Training params:  {info['total_params']:,} "
                f"({info['params_mb']:.2f} MB, incl. aux head)")
    logger.info(f"Input Size: {info['input_size']}")

    # Warn if backbone pretrained weights failed to load
    raw_model_tmp = model.module if hasattr(model, 'module') else model
    if hasattr(raw_model_tmp.backbone, 'pretrained_loaded') and not raw_model_tmp.backbone.pretrained_loaded:
        logger.warning("Backbone is using RANDOM weights (pretrained download failed).")

    # --- torchtune: LoRA / QLoRA (apply before loading finetune/COCO weights) ---
    if args.qlora:
        logger.info("\n--- Applying QLoRA (variant=%s, dtype=%s) ---",
                    args.lora_variant, args.qlora_dtype)
        model = apply_qlora(
            model,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=args.lora_targets,
            quant_dtype=args.qlora_dtype,
            variant=args.lora_variant,
        )
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(f"QLoRA: {trainable:,} / {total:,} trainable params "
                    f"({100.0 * trainable / max(total, 1):.1f}%)")
    elif args.lora:
        logger.info("\n--- Applying LoRA (variant=%s) ---", args.lora_variant)
        model = apply_lora(
            model,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=args.lora_targets,
            variant=args.lora_variant,
        )
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(f"LoRA: {trainable:,} / {total:,} trainable params "
                    f"({100.0 * trainable / max(total, 1):.1f}%)")

    # Fine-tune from a previous checkpoint (inference or training)
    if args.finetune and not args.resume:
        logger.info(f"\nLoading fine-tune weights from: {args.finetune}")
        ckpt = torch.load(args.finetune, map_location=device, weights_only=False)
        src_sd = ckpt.get("model_state_dict", ckpt)
        # FP16 inference checkpoints: cast back to FP32
        src_sd = {k: v.float() if v.is_floating_point() else v for k, v in src_sd.items()}
        missing, unexpected = model.load_state_dict(src_sd, strict=False)
        loaded = len(src_sd) - len(unexpected)
        logger.info(f"  Loaded {loaded} weight tensors from fine-tune checkpoint")
        if missing:
            logger.info(f"  Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing)>5 else ''}")
            logger.info("  (expected — aux_head/aux_fpn are re-initialised for training)")
        if unexpected:
            logger.warning(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}")
    elif args.finetune and args.resume:
        logger.info("--finetune ignored because --resume is set")

    # Load COCO pretrained weights BEFORE DataParallel wrapping
    if args.pretrained_coco and not args.resume and not args.finetune:
        if model_cfg["backbone"] == "0.5x":
            logger.warning(
                "COCO pretrained weights are NOT available for the 0.5x model. "
                "Only 1.0x and 1.5x have official COCO checkpoints. "
                "The 0.5x model will train from scratch (slower convergence)."
            )
        else:
            logger.info("\nLoading COCO pretrained weights for fine-tuning...")
            try:
                result = load_coco_pretrained(
                    model,
                    backbone_size=model_cfg["backbone"],
                    fpn_channels=model_cfg["fpn_channels"],
                    input_size=args.input_size,
                    checkpoint_path=args.pretrained_ckpt,
                )
                logger.info(f"  Loaded {len(result['loaded'])} weight tensors from COCO checkpoint")
                logger.info(f"  Skipped {len(result['skipped'])} tensors (aux_head / cls layer / missing)")
            except ValueError as e:
                logger.warning(f"  COCO pretrained weights not available for this config: {e}")
                logger.warning("  Training from scratch instead (no pretrained weights).")
    elif args.pretrained_coco and args.resume:
        logger.info("--pretrained-coco ignored because --resume is set")

    # --- torchtune: Chunked Loss ---
    if args.chunked_loss:
        logger.info(f"\n--- Enabling Chunked Loss (torchtune-style, chunk_size={args.chunk_size}) ---")
        raw_head = model.head if hasattr(model, 'head') else None
        if raw_head is not None:
            raw_head.use_chunked_loss = True
            raw_head.chunk_size = args.chunk_size
            logger.info("Chunked loss enabled on detection head")

    # AMP scaler (only on CUDA; GradScaler("cuda", ...) is invalid on CPU)
    use_amp = False
    scaler = None
    if args.amp and device.type == "cuda":
        use_amp = True
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        logger.info("AMP: Mixed Precision (FP16) enabled")
    elif args.amp:
        logger.warning("AMP requested but device is not CUDA; mixed precision disabled.")

    # Gradient accumulation
    grad_accum = max(1, args.grad_accum)
    if grad_accum > 1:
        logger.info(f"Gradient Accumulation: {grad_accum} steps "
                    f"(effective batch = {args.batch_size * grad_accum})")

    # Multi-GPU via DataParallel (after pretrained loading, before optimizer/EMA)
    use_multi_gpu = args.multi_gpu and torch.cuda.device_count() > 1
    if use_multi_gpu:
        n_gpus = torch.cuda.device_count()
        logger.info(f"Multi-GPU: using {n_gpus} GPUs via DataParallel")
        model = torch.nn.DataParallel(model)
    elif args.multi_gpu:
        logger.info("Multi-GPU requested but only 1 GPU available, using single GPU")

    # raw_model is always the unwrapped model (used for EMA, saving, etc.)
    raw_model = model.module if use_multi_gpu else model

    # --- torchtune: Activation Checkpointing ---
    if args.activation_checkpointing:
        logger.info("\n--- Enabling Activation Checkpointing (torchtune-style) ---")
        apply_activation_checkpointing(raw_model)

    # --- torchtune: Activation Offloading ---
    offload_hook = None
    if args.activation_offloading:
        logger.info("\n--- Enabling Activation Offloading (torchtune-style) ---")
        offload_hook = ActivationOffloadHook()
        offload_hook.register(raw_model)

    # --- torchtune: torch.compile ---
    if args.compile:
        logger.info("\n--- Applying torch.compile (torchtune-style) ---")
        raw_model = torchtune_compile(raw_model)
        if not use_multi_gpu:
            model = raw_model

    # Log GPU memory before optimizer setup
    if device.type == "cuda":
        log_memory_stats(device, prefix="Pre-optimizer")

    # Optimizer and scheduler (with torchtune-style options)
    base_lr = args.lr  # Honour the user-specified LR exactly

    optimizer = create_optimizer(
        model,
        lr=base_lr,
        weight_decay=config.train.weight_decay,
        use_8bit=args.use_8bit_optimizer,
        optimizer_in_bwd=args.optimizer_in_bwd,
        betas=(0.9, 0.999),
    )
    
    # LR schedule: linear warmup then cosine annealing with eta_min=0.00005
    # eta_min matches official FlashDet config (prevents LR from going too low)
    eta_min = 0.00005
    eta_min_factor = eta_min / base_lr  # e.g. 0.00005 / 0.001 = 0.05

    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        else:
            progress = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return eta_min_factor + (1.0 - eta_min_factor) * cosine

    # When optimizer_in_bwd is used, the optimizer is fused into backward hooks.
    # We manually adjust LR via set_lr() instead of a scheduler.
    scheduler = None
    if not args.optimizer_in_bwd:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        logger.info("Scheduler disabled (optimizer fused into backward — LR adjusted manually)")
    
    logger.info(f"Base LR: {base_lr}, Weight Decay: {config.train.weight_decay}")
    logger.info(f"Warmup: {args.warmup_epochs} epochs, eta_min: {eta_min:.6f}")

    # --- torchtune optimizations summary ---
    tt_flags = []
    if args.activation_checkpointing:
        tt_flags.append("activation_ckpt")
    if args.activation_offloading:
        tt_flags.append("activation_offload")
    if args.optimizer_in_bwd:
        tt_flags.append("optimizer_in_bwd")
    if args.use_8bit_optimizer:
        tt_flags.append("8bit_adamw")
    if args.compile:
        tt_flags.append("torch.compile")
    if args.chunked_loss:
        tt_flags.append(f"chunked_loss(chunk={args.chunk_size})")
    if args.qlora:
        tt_flags.append(f"QLoRA(r={args.lora_rank}, alpha={args.lora_alpha}, dtype={args.qlora_dtype})")
    elif args.lora:
        tt_flags.append(f"LoRA(r={args.lora_rank}, alpha={args.lora_alpha})")
    if tt_flags:
        logger.info(f"torchtune optimizations: {', '.join(tt_flags)}")
    else:
        logger.info("torchtune optimizations: none (use --help to see options)")

    # EMA with adaptive warmup (always on the raw unwrapped model)
    ema = ModelEMA(raw_model, decay=0.9998, warmup=2000)
    logger.info(f"EMA enabled (target_decay=0.9998, warmup=2000, "
                f"~{len(train_loader)*5} iters in first 5 epochs)")

    # Resume
    start_epoch = 0
    best_loss = float("inf")
    # NOTE: best_map50 is initialised in the training-loop block below.

    if args.resume:
        ckpt = load_checkpoint(raw_model, args.resume, optimizer, scheduler, device)
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt.get("loss", float("inf"))
        # Restore EMA state if saved
        try:
            raw = torch.load(args.resume, map_location=device, weights_only=False)
        except Exception as e:
            logger.warning("Could not load checkpoint file for EMA/extra state: %s", e)
            raw = {}
        if raw and "ema_state_dict" in raw:
            ema.load_state_dict(raw["ema_state_dict"])
            logger.info(f"EMA state restored (num_updates={ema.num_updates}, "
                        f"current_decay={ema.decay:.6f})")
        else:
            ema = ModelEMA(raw_model, decay=0.9998, warmup=2000)
            logger.info("EMA warm-started from checkpoint weights")
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # model_config dict is embedded in every checkpoint so test.py can read
    # the correct class names and architecture without touching config.py.
    model_config = {
        "num_classes": num_classes,
        "input_size": input_size,
        "backbone_size": model_cfg["backbone"],
        "fpn_channels": model_cfg["fpn_channels"],
        "class_names": class_names,
    }

    # Training loop
    logger.info("\nStarting training...")
    logger.info("-" * 60)

    best_map50 = 0.0   # Best model selected by mAP@0.5, not by val loss
    epochs_without_improvement = 0

    for epoch in range(start_epoch, args.epochs):
        # For optimizer_in_bwd, manually compute and set the LR each epoch
        if args.optimizer_in_bwd:
            lr_factor = lr_lambda(epoch)
            current_lr = base_lr * lr_factor
            optimizer.set_lr(current_lr)
        else:
            current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs} "
                    f"(lr={current_lr:.6f}, ema_decay={ema.decay:.6f})")
        
        epoch_start = time.time()

        # Train
        train_losses = train_one_epoch(
            model, train_loader, optimizer, device, epoch + 1, logger,
            save_dir=args.save_dir, config=config, ema=ema,
            class_names=class_names, colors=colors,
            scaler=scaler, grad_accum=grad_accum,
        )

        epoch_time = time.time() - epoch_start
        # Show time estimate after the first epoch
        if epoch == start_epoch:
            remaining = args.epochs - (epoch + 1)
            est_total = epoch_time * remaining
            if est_total > 3600:
                est_str = f"{est_total / 3600:.1f}h"
            elif est_total > 60:
                est_str = f"{est_total / 60:.0f}m"
            else:
                est_str = f"{est_total:.0f}s"
            logger.info(
                f"Epoch time: {epoch_time:.1f}s | "
                f"Estimated remaining: {est_str} for {remaining} epochs"
            )
            if device.type == "cuda":
                log_memory_stats(device, prefix=f"Epoch {epoch+1}")

        # Validate using EMA weights (better accuracy than raw model)
        # val_interval controls how often we run the (relatively expensive) mAP pass.
        if (epoch + 1) % config.train.val_interval == 0:
            val_loss, map50 = validate(
                raw_model, val_loader, device, logger, ema=ema, class_names=class_names
            )

            # Track best val loss for reference
            if val_loss < best_loss:
                best_loss = val_loss

            # Save best model by mAP@0.5 (the proper detection metric)
            if map50 > best_map50:
                best_map50 = map50
                epochs_without_improvement = 0
                save_checkpoint(
                    raw_model, optimizer, epoch, val_loss,
                    os.path.join(args.save_dir, "checkpoint_best.pth"),
                    scheduler=scheduler,
                    config=model_config
                )
                save_inference_weights(
                    ema.ema,
                    os.path.join(args.save_dir, "model_best_inference.pth"),
                    config=model_config,
                    half=False
                )
                save_inference_weights(
                    ema.ema,
                    os.path.join(args.save_dir, "model_best_fp16.pth"),
                    config=model_config,
                    half=True
                )
                logger.info(f"Saved best model (EMA mAP@0.5: {best_map50:.4f}, val loss: {val_loss:.4f})")
            else:
                epochs_without_improvement += config.train.val_interval
                logger.info(
                    f"  No mAP improvement for {epochs_without_improvement} epochs "
                    f"(best={best_map50:.4f}, current={map50:.4f})"
                )

            # Early stopping
            if args.patience > 0 and epochs_without_improvement >= args.patience:
                logger.info(
                    f"\nEarly stopping triggered: no mAP improvement for "
                    f"{epochs_without_improvement} epochs (patience={args.patience})"
                )
                break

        # Save latest checkpoint (EMA state included in one atomic write)
        ckpt_path = os.path.join(args.save_dir, "checkpoint_last.pth")
        save_checkpoint(
            raw_model, optimizer, epoch, train_losses["loss"],
            ckpt_path,
            scheduler=scheduler,
            config=model_config,
            ema=ema,
        )

        save_inference_weights(
            ema.ema,
            os.path.join(args.save_dir, "model_last_inference.pth"),
            config=model_config,
            half=False
        )
        save_inference_weights(
            ema.ema,
            os.path.join(args.save_dir, "model_last_fp16.pth"),
            config=model_config,
            half=True
        )

        # Step scheduler (no-op when optimizer_in_bwd; LR is set manually above)
        if scheduler is not None:
            scheduler.step()
    
    # Save final inference weights at end of training
    logger.info("\nSaving final inference weights...")

    # If LoRA/QLoRA was used, save adapter weights separately and merge for inference
    if args.lora or args.qlora:
        lora_path = os.path.join(args.save_dir, "lora_adapters.pth")
        torch.save(get_lora_state_dict(ema.ema), lora_path)
        logger.info(f"LoRA adapter weights saved: {lora_path}")

        logger.info("Merging LoRA weights into base model for inference...")
        merge_lora_weights(ema.ema)

    # model_config was built once before the loop and reused throughout.
    # Save final EMA inference weights
    save_inference_weights(
        ema.ema,
        os.path.join(args.save_dir, "model_final_inference.pth"),
        config=model_config,
        half=False
    )
    save_inference_weights(
        ema.ema,
        os.path.join(args.save_dir, "model_final_fp16.pth"),
        config=model_config,
        half=True
    )

    # Clean up activation offloading hooks
    if offload_hook is not None:
        offload_hook.remove()

    # Final memory stats
    if device.type == "cuda":
        log_memory_stats(device, prefix="Training complete")
    
    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info(f"Best mAP@0.5:         {best_map50:.4f}")
    logger.info(f"Best Validation Loss: {best_loss:.4f}")
    logger.info(f"Checkpoints saved to: {args.save_dir}")
    logger.info(f"  - checkpoint_best.pth      (full training checkpoint)")
    logger.info(f"  - checkpoint_last.pth      (full training checkpoint)")
    logger.info(f"  - model_best_inference.pth (inference FP32, no aux_head)")
    logger.info(f"  - model_best_fp16.pth      (inference FP16, smallest)")
    logger.info(f"  - model_final_inference.pth (final epoch FP32)")
    logger.info(f"  - model_final_fp16.pth     (final epoch FP16)")
    if args.lora or args.qlora:
        logger.info(f"  - lora_adapters.pth        (LoRA adapter weights only)")
    if tt_flags:
        logger.info(f"torchtune optimizations used: {', '.join(tt_flags)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
