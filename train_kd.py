#!/usr/bin/env python3
"""
Knowledge Distillation training for FlashDet.

Trains a smaller student model by distilling knowledge from a larger
teacher model — inspired by torchtune's KD recipe adapted for
object detection.

Usage:
    # Distill from m-1.5x teacher to m-0.5x student
    python train_kd.py \
        --teacher-checkpoint workspace/teacher/model_best_inference.pth \
        --teacher-size m-1.5x \
        --model-size m-0.5x \
        --epochs 100

    # Distill with LoRA on student backbone
    python train_kd.py \
        --teacher-checkpoint workspace/teacher/model_best_inference.pth \
        --lora --lora-rank 8 \
        --kd-temperature 4.0

    # Logit-only distillation (no feature KD)
    python train_kd.py \
        --teacher-checkpoint workspace/teacher/model_best_inference.pth \
        --kd-feature-weight 0.0
"""

import os
import sys
import argparse
import time
import json
import math
import copy
import logging

import cv2
import numpy as np
import torch
import torch.nn as nn

from flashdet.cfg import get_config
from flashdet.models import FlashDet, load_coco_pretrained
from flashdet.models.lora import apply_lora, apply_qlora, merge_lora_weights, get_lora_state_dict
from flashdet.data import create_dataloader, verify_dataset
from flashdet.losses.kd_loss import KnowledgeDistillationLoss
from flashdet.utils import (
    save_checkpoint, load_checkpoint, save_weights_only,
    save_inference_weights, setup_logger, AverageMeter,
)
from flashdet.utils.metrics import compute_map
from flashdet.utils.torchtune_optim import (
    apply_activation_checkpointing,
    create_optimizer,
    log_memory_stats,
)


logger = logging.getLogger(__name__)


MODEL_SIZE_MAP = {
    "m":      {"backbone": "1.0x", "fpn_channels": 96},
    "m-1.5x": {"backbone": "1.5x", "fpn_channels": 128},
    "m-0.5x": {"backbone": "0.5x", "fpn_channels": 96},
}


class ModelEMA:
    """Exponential Moving Average of model weights (same as train.py)."""
    def __init__(self, model, decay=0.9998, warmup=2000):
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
    def update(self, model):
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

    def load_state_dict(self, state):
        self.ema.load_state_dict(state["ema_state"], strict=False)
        self.target_decay = state.get("target_decay", self.target_decay)
        self.warmup = state.get("warmup", self.warmup)
        self.num_updates = state.get("num_updates", 0)


def _load_class_names_from_ann(ann_file):
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


def load_teacher_model(
    checkpoint_path, model_size, num_classes, input_size, device
):
    """Load a pretrained teacher model for distillation."""
    cfg = MODEL_SIZE_MAP[model_size]

    teacher = FlashDet(
        num_classes=num_classes,
        input_size=input_size,
        backbone_size=cfg["backbone"],
        fpn_channels=cfg["fpn_channels"],
        pretrained=False,
        use_aux_head=False,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    src_sd = ckpt.get("model_state_dict", ckpt)
    src_sd = {k: v.float() if v.is_floating_point() else v for k, v in src_sd.items()}

    # Try to load model config from checkpoint for class count validation
    ckpt_config = ckpt.get("config", ckpt.get("model_config", {}))
    if isinstance(ckpt_config, dict):
        ckpt_nc = ckpt_config.get("num_classes", num_classes)
        if ckpt_nc != num_classes:
            logger.warning(
                "Teacher checkpoint has %d classes, student expects %d. "
                "Classification head weights will be partially loaded.",
                ckpt_nc, num_classes,
            )

    missing, unexpected = teacher.load_state_dict(src_sd, strict=False)
    loaded = len(src_sd) - len(unexpected)
    logger.info("Teacher: loaded %d weight tensors (missing=%d, unexpected=%d)",
                loaded, len(missing), len(unexpected))

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    return teacher


def train_one_epoch_kd(
    student, teacher, kd_criterion, dataloader, optimizer, device,
    epoch, _logger, kd_hard_weight=1.0, scaler=None, grad_accum=1,
    ema=None,
):
    """Train one epoch with Knowledge Distillation."""
    student.train()
    teacher.eval()
    use_amp = scaler is not None

    loss_meter = AverageMeter("Loss")
    hard_meter = AverageMeter("Hard")
    kd_meter = AverageMeter("KD")
    kd_cls_meter = AverageMeter("KD_Cls")
    kd_feat_meter = AverageMeter("KD_Feat")

    raw_student = student.module if hasattr(student, "module") else student
    start_time = time.time()

    for batch_idx, (images, gt_meta) in enumerate(dataloader):
        images = images.to(device)

        with torch.amp.autocast(device.type, enabled=use_amp):
            # Student forward (with features for KD)
            s_out = student(images, gt_meta, epoch=epoch, return_features=True)
            hard_loss = s_out["loss"]
            s_preds = s_out["preds"]
            s_fpn = s_out["fpn_features"]

            # Teacher forward (no grad)
            with torch.no_grad():
                t_out = teacher(images, return_features=True)
                t_preds = t_out["preds"]
                t_fpn = t_out["fpn_features"]

            # KD losses
            kd_result = kd_criterion(
                student_preds=s_preds,
                teacher_preds=t_preds.detach(),
                student_fpn_feats=s_fpn,
                teacher_fpn_feats=[f.detach() for f in t_fpn],
                num_classes=raw_student.num_classes,
                reg_max=raw_student.head.reg_max,
            )
            kd_loss = kd_result["kd_loss"]

            total_loss = (kd_hard_weight * hard_loss + kd_loss) / grad_accum

        if torch.isnan(total_loss):
            _logger.warning("NaN loss at batch %d, skipping", batch_idx)
            continue

        if scaler:
            scaler.scale(total_loss).backward()
        else:
            total_loss.backward()

        if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(dataloader):
            if scaler:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(raw_student.parameters(), 35.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(raw_student.parameters(), 35.0)
                optimizer.step()
            optimizer.zero_grad()

            if ema is not None:
                ema.update(raw_student)

        loss_meter.update((kd_hard_weight * hard_loss + kd_loss).item())
        hard_meter.update(hard_loss.item())
        kd_meter.update(kd_loss.item())
        kd_cls_meter.update(kd_result["kd_cls_loss"].item())
        kd_feat_meter.update(kd_result["kd_feature_loss"].item())

        if (batch_idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            _logger.info(
                "Epoch [%d] Batch [%d/%d] "
                "Total: %.4f (Hard: %.4f, KD: %.4f [cls: %.4f, feat: %.4f]) "
                "Time: %.1fs",
                epoch, batch_idx + 1, len(dataloader),
                loss_meter.avg, hard_meter.avg, kd_meter.avg,
                kd_cls_meter.avg, kd_feat_meter.avg, elapsed,
            )
            sys.stdout.flush()

    return {
        "loss": loss_meter.avg,
        "hard_loss": hard_meter.avg,
        "kd_loss": kd_meter.avg,
        "kd_cls_loss": kd_cls_meter.avg,
        "kd_feature_loss": kd_feat_meter.avg,
    }


@torch.no_grad()
def validate(model, dataloader, device, _logger, ema=None, class_names=None):
    """Validate model (same as train.py)."""
    eval_model = ema.ema if ema is not None else model
    eval_model.eval()

    loss_meter = AverageMeter("Loss")
    all_preds = []
    all_gts = []

    for images, gt_meta in dataloader:
        images = images.to(device)
        out = eval_model(images, gt_meta, epoch=0, compute_loss=True)
        loss_meter.update(out["loss"].item())

        results = eval_model.predict(images, None, score_thr=0.05, nms_thr=0.6)
        for i, (dets, lbs) in enumerate(results):
            gt_boxes = gt_meta["gt_bboxes"][i]
            gt_labels = gt_meta["gt_labels"][i]

            if dets is not None and dets.numel() > 0:
                boxes_np = dets[:, :4].cpu().numpy()
                scores_np = dets[:, 4].cpu().numpy()
                lbs_np = lbs.cpu().numpy()
            else:
                boxes_np = np.zeros((0, 4), dtype=np.float32)
                scores_np = np.zeros(0, dtype=np.float32)
                lbs_np = np.zeros(0, dtype=np.int64)

            all_preds.append({"boxes": boxes_np, "scores": scores_np, "labels": lbs_np})
            all_gts.append({"boxes": gt_boxes, "labels": gt_labels})

    num_cls = len(class_names) if class_names else 10
    map_results = compute_map(all_preds, all_gts, iou_threshold=0.5, num_classes=num_cls)
    map50 = map_results["mAP"]

    ap_per_cls = map_results.get("AP_per_class", {})
    if class_names and ap_per_cls:
        per_cls_str = "  ".join(
            f"{class_names[cid]}={v:.3f}"
            for cid, v in sorted(ap_per_cls.items())
            if cid < len(class_names)
        )
        _logger.info("  AP per class: %s", per_cls_str)

    _logger.info("Validation — Loss: %.4f | mAP@0.5: %.4f", loss_meter.avg, map50)
    model.train()
    return loss_meter.avg, map50


def main():
    parser = argparse.ArgumentParser(
        description="Knowledge Distillation training for FlashDet "
                    "(torchtune-inspired)",
    )

    # --- Teacher ---
    t_group = parser.add_argument_group("Teacher Model")
    t_group.add_argument("--teacher-checkpoint", required=True,
                         help="Path to teacher model checkpoint (.pth)")
    t_group.add_argument("--teacher-size", default="m-1.5x",
                         choices=["m", "m-1.5x", "m-0.5x"],
                         help="Teacher model size (default: m-1.5x)")

    # --- Student ---
    s_group = parser.add_argument_group("Student Model")
    s_group.add_argument("--model-size", default="m", choices=["m", "m-1.5x", "m-0.5x"],
                         help="Student model size (default: m)")
    s_group.add_argument("--input-size", type=int, default=320,
                         help="Input image size")
    s_group.add_argument("--pretrained-coco", action="store_true",
                         help="Load COCO pretrained weights for student")
    s_group.add_argument("--finetune", default=None,
                         help="Student checkpoint to fine-tune from")

    # --- KD settings ---
    kd_group = parser.add_argument_group("Knowledge Distillation")
    kd_group.add_argument("--kd-temperature", type=float, default=4.0,
                          help="KD softmax temperature (default: 4.0)")
    kd_group.add_argument("--kd-logit-weight", type=float, default=1.0,
                          help="Weight for logit-level KD loss (default: 1.0)")
    kd_group.add_argument("--kd-feature-weight", type=float, default=0.5,
                          help="Weight for feature-level KD loss (default: 0.5)")
    kd_group.add_argument("--kd-hard-weight", type=float, default=1.0,
                          help="Weight for hard (GT) detection loss (default: 1.0)")

    # --- LoRA on student ---
    lora_group = parser.add_argument_group("LoRA (on student)")
    lora_group.add_argument("--lora", action="store_true",
                            help="Apply LoRA to student backbone")
    lora_group.add_argument("--qlora", action="store_true",
                            help="Apply QLoRA (quantized LoRA) to student")
    lora_group.add_argument("--lora-rank", type=int, default=8)
    lora_group.add_argument("--lora-alpha", type=float, default=16.0)
    lora_group.add_argument("--lora-dropout", type=float, default=0.05)
    lora_group.add_argument("--lora-targets", nargs="+", default=["backbone", "fpn"])

    # --- Training ---
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--save-dir", default="workspace/kd_experiment")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--activation-checkpointing", action="store_true")
    parser.add_argument("--class-file", default=None)
    parser.add_argument("--train-images", default=None)
    parser.add_argument("--val-images", default=None)

    args = parser.parse_args()

    student_cfg = MODEL_SIZE_MAP[args.model_size]
    teacher_cfg = MODEL_SIZE_MAP[args.teacher_size]
    input_size = (args.input_size, args.input_size)

    os.makedirs(args.save_dir, exist_ok=True)
    _logger = setup_logger("KD-Training", args.save_dir)

    if torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")
        if args.device not in ("cpu", ""):
            _logger.warning("CUDA not available; using CPU.")

    config = get_config()
    if args.train_images:
        config.data.train_images = args.train_images
        config.data.train_annotations = os.path.join(args.train_images, "_annotations.coco.json")
    if args.val_images:
        config.data.val_images = args.val_images
        config.data.val_annotations = os.path.join(args.val_images, "_annotations.coco.json")

    # Resolve class names
    class_names = None
    if args.class_file:
        with open(args.class_file, encoding="utf-8") as f:
            class_names = [l.strip() for l in f if l.strip()]
    if not class_names:
        class_names = _load_class_names_from_ann(config.data.train_annotations)
    if not class_names:
        class_names = config.class_names
    num_classes = len(class_names)

    _logger.info("=" * 60)
    _logger.info("FlashDet Knowledge Distillation (torchtune-style)")
    _logger.info("=" * 60)
    _logger.info("Device:           %s", device)
    _logger.info("Teacher:          %s (backbone=%s, fpn=%d)",
                 args.teacher_size, teacher_cfg["backbone"], teacher_cfg["fpn_channels"])
    _logger.info("Student:          %s (backbone=%s, fpn=%d)",
                 args.model_size, student_cfg["backbone"], student_cfg["fpn_channels"])
    _logger.info("Input Size:       %s", input_size)
    _logger.info("KD Temperature:   %.1f", args.kd_temperature)
    _logger.info("KD Logit Weight:  %.2f", args.kd_logit_weight)
    _logger.info("KD Feature Weight: %.2f", args.kd_feature_weight)
    _logger.info("Hard Loss Weight: %.2f", args.kd_hard_weight)
    _logger.info("Classes (%d):     %s", num_classes, class_names)

    # Verify dataset
    data_root = os.path.dirname(os.path.normpath(config.data.train_images))
    if not verify_dataset(data_root):
        _logger.error("Dataset not found!")
        sys.exit(1)

    # Data loaders
    _logger.info("\nLoading datasets...")
    train_loader = create_dataloader(
        img_dir=config.data.train_images,
        ann_file=config.data.train_annotations,
        batch_size=args.batch_size,
        input_size=input_size,
        num_workers=args.workers,
        is_train=True,
    )
    val_loader = create_dataloader(
        img_dir=config.data.val_images,
        ann_file=config.data.val_annotations,
        batch_size=args.batch_size,
        input_size=input_size,
        num_workers=args.workers,
        is_train=False,
    )
    _logger.info("Train batches: %d", len(train_loader))
    _logger.info("Val batches:   %d", len(val_loader))

    # ── Build Teacher ──
    _logger.info("\nLoading teacher model from: %s", args.teacher_checkpoint)
    teacher = load_teacher_model(
        args.teacher_checkpoint, args.teacher_size,
        num_classes, input_size, device,
    )
    t_params = sum(p.numel() for p in teacher.parameters())
    _logger.info("Teacher params: %d (%.2f MB)", t_params, t_params * 4 / 1024**2)

    # ── Build Student ──
    _logger.info("\nBuilding student model...")
    student = FlashDet(
        num_classes=num_classes,
        input_size=input_size,
        backbone_size=student_cfg["backbone"],
        fpn_channels=student_cfg["fpn_channels"],
        pretrained=True,
        use_aux_head=True,
    ).to(device)

    # Optional COCO pretrain for student
    if args.pretrained_coco and not args.resume and not args.finetune:
        if student_cfg["backbone"] != "0.5x":
            _logger.info("Loading COCO pretrained weights for student...")
            try:
                load_coco_pretrained(
                    student, backbone_size=student_cfg["backbone"],
                    fpn_channels=student_cfg["fpn_channels"],
                    input_size=args.input_size,
                )
            except ValueError as e:
                _logger.warning("COCO pretrained not available: %s", e)

    # Optional fine-tune init
    if args.finetune and not args.resume:
        _logger.info("Loading fine-tune weights for student from: %s", args.finetune)
        ckpt = torch.load(args.finetune, map_location=device, weights_only=False)
        src_sd = ckpt.get("model_state_dict", ckpt)
        src_sd = {k: v.float() if v.is_floating_point() else v for k, v in src_sd.items()}
        student.load_state_dict(src_sd, strict=False)

    # LoRA / QLoRA on student
    if args.qlora:
        _logger.info("\n--- Applying QLoRA to student ---")
        from flashdet.models.lora import apply_qlora
        student = apply_qlora(
            student, rank=args.lora_rank, alpha=args.lora_alpha,
            dropout=args.lora_dropout, target_modules=args.lora_targets,
        )
    elif args.lora:
        _logger.info("\n--- Applying LoRA to student ---")
        student = apply_lora(
            student, rank=args.lora_rank, alpha=args.lora_alpha,
            dropout=args.lora_dropout, target_modules=args.lora_targets,
        )

    s_info = student.get_model_info()
    s_trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    s_total = sum(p.numel() for p in student.parameters())
    _logger.info("Student: %d / %d trainable params (%.1f%%)",
                 s_trainable, s_total, 100.0 * s_trainable / max(s_total, 1))

    if args.activation_checkpointing:
        _logger.info("Enabling activation checkpointing on student")
        apply_activation_checkpointing(student)

    # ── KD Loss ──
    kd_criterion = KnowledgeDistillationLoss(
        temperature=args.kd_temperature,
        logit_weight=args.kd_logit_weight,
        feature_weight=args.kd_feature_weight,
        student_channels=student_cfg["fpn_channels"],
        teacher_channels=teacher_cfg["fpn_channels"],
        num_levels=len(student.strides),
    ).to(device)

    # AMP
    scaler = None
    if args.amp and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        _logger.info("AMP enabled")

    grad_accum = max(1, args.grad_accum)

    raw_student = student.module if hasattr(student, "module") else student

    # Optimizer
    all_params = list(student.parameters()) + list(kd_criterion.parameters())
    trainable_params = [p for p in all_params if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr,
        weight_decay=config.train.weight_decay, betas=(0.9, 0.999),
    )

    # LR schedule
    base_lr = args.lr
    eta_min = 5e-5
    eta_min_factor = eta_min / base_lr

    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return eta_min_factor + (1.0 - eta_min_factor) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # EMA
    ema = ModelEMA(raw_student, decay=0.9998, warmup=2000)

    # Resume
    start_epoch = 0
    best_map50 = 0.0
    best_loss = float("inf")

    if args.resume:
        ckpt = load_checkpoint(raw_student, args.resume, optimizer, scheduler, device)
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt.get("loss", float("inf"))
        raw_ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if raw_ckpt and "ema_state_dict" in raw_ckpt:
            ema.load_state_dict(raw_ckpt["ema_state_dict"])
        else:
            ema = ModelEMA(raw_student, decay=0.9998, warmup=2000)
        _logger.info("Resumed from epoch %d", start_epoch)

    model_config = {
        "num_classes": num_classes,
        "input_size": input_size,
        "backbone_size": student_cfg["backbone"],
        "fpn_channels": student_cfg["fpn_channels"],
        "class_names": class_names,
        "kd_teacher_size": args.teacher_size,
    }

    if device.type == "cuda":
        log_memory_stats(device, prefix="Pre-training")

    # ── Training loop ──
    _logger.info("\nStarting KD training...")
    _logger.info("-" * 60)

    epochs_without_improvement = 0

    for epoch in range(start_epoch, args.epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        _logger.info("\nEpoch %d/%d (lr=%.6f, ema_decay=%.6f)",
                     epoch + 1, args.epochs, current_lr, ema.decay)

        epoch_start = time.time()

        train_losses = train_one_epoch_kd(
            student, teacher, kd_criterion,
            train_loader, optimizer, device,
            epoch + 1, _logger,
            kd_hard_weight=args.kd_hard_weight,
            scaler=scaler, grad_accum=grad_accum,
            ema=ema,
        )

        epoch_time = time.time() - epoch_start
        if epoch == start_epoch:
            remaining = args.epochs - (epoch + 1)
            est = epoch_time * remaining
            if est > 3600:
                est_str = f"{est / 3600:.1f}h"
            elif est > 60:
                est_str = f"{est / 60:.0f}m"
            else:
                est_str = f"{est:.0f}s"
            _logger.info("Epoch time: %.1fs | Est remaining: %s", epoch_time, est_str)
            if device.type == "cuda":
                log_memory_stats(device, prefix=f"Epoch {epoch + 1}")

        # Validate
        if (epoch + 1) % config.train.val_interval == 0:
            val_loss, map50 = validate(
                raw_student, val_loader, device, _logger,
                ema=ema, class_names=class_names,
            )

            if val_loss < best_loss:
                best_loss = val_loss

            if map50 > best_map50:
                best_map50 = map50
                epochs_without_improvement = 0
                save_checkpoint(
                    raw_student, optimizer, epoch, val_loss,
                    os.path.join(args.save_dir, "checkpoint_best.pth"),
                    scheduler=scheduler, config=model_config,
                )
                save_inference_weights(
                    ema.ema,
                    os.path.join(args.save_dir, "model_best_inference.pth"),
                    config=model_config, half=False,
                )
                save_inference_weights(
                    ema.ema,
                    os.path.join(args.save_dir, "model_best_fp16.pth"),
                    config=model_config, half=True,
                )
                _logger.info("Saved best model (mAP@0.5: %.4f)", best_map50)
            else:
                epochs_without_improvement += config.train.val_interval
                _logger.info("  No improvement for %d epochs (best=%.4f, current=%.4f)",
                             epochs_without_improvement, best_map50, map50)

            if args.patience > 0 and epochs_without_improvement >= args.patience:
                _logger.info("\nEarly stopping after %d epochs without improvement",
                             epochs_without_improvement)
                break

        # Save latest
        save_checkpoint(
            raw_student, optimizer, epoch, train_losses["loss"],
            os.path.join(args.save_dir, "checkpoint_last.pth"),
            scheduler=scheduler, config=model_config, ema=ema,
        )
        save_inference_weights(
            ema.ema,
            os.path.join(args.save_dir, "model_last_inference.pth"),
            config=model_config, half=False,
        )

        scheduler.step()

    # ── Save final ──
    _logger.info("\nSaving final weights...")

    if args.lora or args.qlora:
        lora_path = os.path.join(args.save_dir, "lora_adapters.pth")
        torch.save(get_lora_state_dict(ema.ema), lora_path)
        _logger.info("LoRA adapters saved: %s", lora_path)
        merge_lora_weights(ema.ema)

    save_inference_weights(
        ema.ema,
        os.path.join(args.save_dir, "model_final_inference.pth"),
        config=model_config, half=False,
    )
    save_inference_weights(
        ema.ema,
        os.path.join(args.save_dir, "model_final_fp16.pth"),
        config=model_config, half=True,
    )

    if device.type == "cuda":
        log_memory_stats(device, prefix="Training complete")

    _logger.info("\n" + "=" * 60)
    _logger.info("Knowledge Distillation Training Complete!")
    _logger.info("Teacher:          %s", args.teacher_size)
    _logger.info("Student:          %s", args.model_size)
    _logger.info("Best mAP@0.5:     %.4f", best_map50)
    _logger.info("Best Val Loss:    %.4f", best_loss)
    _logger.info("Checkpoints:      %s", args.save_dir)
    if args.lora or args.qlora:
        _logger.info("  - lora_adapters.pth")
    _logger.info("=" * 60)


if __name__ == "__main__":
    main()
