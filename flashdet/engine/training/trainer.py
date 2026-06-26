"""FlashDet Trainer — wraps the full training loop into a reusable class."""

import os
import copy
import math
import json
import logging
from typing import Dict, List, Optional, Any

import numpy as np
import torch
import torch.nn as nn

from flashdet.cfg import get_config
from flashdet.models import FlashDet, load_coco_pretrained
from flashdet.models.detector import build_model, ARCHITECTURE_REGISTRY
from flashdet.models.lora import (
    apply_lora, apply_qlora, merge_lora_weights, get_lora_state_dict,
)
from flashdet.data import create_dataloader, verify_dataset
from flashdet.utils import (
    save_checkpoint, load_checkpoint, save_inference_weights, setup_logger, AverageMeter,
)
from flashdet.utils.metrics import compute_map
from flashdet.utils.torchtune_optim import (
    apply_activation_checkpointing,
    ActivationOffloadHook,
    create_optimizer,
    compile_model as torchtune_compile,
)
from flashdet.engine.core.callbacks import CallbackList, Callback
from flashdet.engine.core.ema import ModelEMA

logger = logging.getLogger(__name__)


MODEL_SIZE_MAP = {
    "m": {"backbone": "1.0x", "fpn_channels": 96},
    "m-1.5x": {"backbone": "1.5x", "fpn_channels": 128},
    "m-0.5x": {"backbone": "0.5x", "fpn_channels": 96},
}

FLASHDET_SIZES = frozenset({"p", "n", "s", "m", "l", "x"})
_LEGACY_FLASHDET_SIZE = {"m": "n", "m-0.5x": "n", "m-1.5x": "s"}


def resolve_flashdet_size(model_size: str) -> str:
    """Map trainer model_size to FlashDet ``size`` argument."""
    if model_size in FLASHDET_SIZES:
        return model_size
    return _LEGACY_FLASHDET_SIZE.get(model_size, "n")


def resolve_model_cfg(model_size: str) -> Dict[str, Any]:
    """Checkpoint metadata for legacy and modern FlashDet sizes."""
    if model_size in MODEL_SIZE_MAP:
        return dict(MODEL_SIZE_MAP[model_size])
    return {
        "backbone": model_size,
        "fpn_channels": 64 if model_size == "p" else None,
        "size": resolve_flashdet_size(model_size),
    }


class Trainer:
    """High-level trainer for FlashDet.

    Example::

        from flashdet import Trainer

        trainer = Trainer(
            epochs=100,
            batch_size=32,
            model_size="m",
            pretrained_coco=True,
            lora=True,
            amp=True,
        )
        trainer.train()
    """

    def __init__(
        self,
        # Basic training
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 0.001,
        workers: int = 4,
        save_dir: str = "workspace/flashdet_output",
        resume: Optional[str] = None,
        device: str = "cuda",
        warmup_epochs: int = 5,
        patience: int = 50,
        # Model
        model_size: str = "m",
        input_size: int = 320,
        architecture: str = "flashdet",
        finetune: Optional[str] = None,
        pretrained_coco: bool = False,
        pretrained_ckpt: Optional[str] = None,
        # Data
        class_file: Optional[str] = None,
        train_images: Optional[str] = None,
        val_images: Optional[str] = None,
        # Performance
        amp: bool = False,
        multi_gpu: bool = False,
        grad_accum: int = 1,
        # torchtune optimizations
        activation_checkpointing: bool = False,
        activation_offloading: bool = False,
        optimizer_in_bwd: bool = False,
        use_8bit_optimizer: bool = False,
        compile: bool = False,
        chunked_loss: bool = False,
        chunk_size: int = 1024,
        # LoRA
        lora: bool = False,
        lora_variant: str = "standard",
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        lora_targets: Optional[List[str]] = None,
        qlora: bool = False,
        qlora_dtype: str = "int8",
        # Backbone (Pico only)
        backbone_type: str = "shufflenet",
        # Augmentations
        mosaic: bool = False,
        mixup: bool = False,
        copy_paste: bool = False,
        # Config override
        config: Any = None,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.workers = workers
        self.save_dir = save_dir
        self.resume = resume
        self.warmup_epochs = warmup_epochs
        self.patience = patience
        self.model_size = model_size
        self.input_size = (input_size, input_size)
        self.finetune = finetune
        self.pretrained_coco = pretrained_coco
        self.pretrained_ckpt = pretrained_ckpt
        self.class_file = class_file
        self.train_images = train_images
        self.val_images = val_images
        self.amp = amp
        self.multi_gpu = multi_gpu
        self.grad_accum = max(1, grad_accum)
        self.activation_checkpointing = activation_checkpointing
        self.activation_offloading = activation_offloading
        self.optimizer_in_bwd = optimizer_in_bwd
        self.use_8bit_optimizer = use_8bit_optimizer
        self.compile = compile
        self.chunked_loss = chunked_loss
        self.chunk_size = chunk_size
        self.lora = lora
        self.lora_variant = lora_variant
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_targets = lora_targets or ["backbone", "fpn"]
        self.qlora = qlora
        self.qlora_dtype = qlora_dtype
        self.architecture = architecture
        self.backbone_type = backbone_type
        self.mosaic = mosaic
        self.mixup = mixup
        self.copy_paste = copy_paste

        self._config = config or get_config()

        # When initialised from a YAML config, apply config values as defaults
        # for any parameter the caller did not explicitly set.
        if config is not None:
            self._apply_config_defaults(config)

        self._model_cfg = resolve_model_cfg(self.model_size)

        # Resolve device
        if torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")
            if device not in ("cpu", ""):
                logger.warning("CUDA unavailable; falling back to CPU.")

        os.makedirs(self.save_dir, exist_ok=True)
        self._logger = setup_logger("FlashDet", self.save_dir)
        self.callbacks = CallbackList()

    def _apply_config_defaults(self, cfg) -> None:
        """Map YAML Config fields onto Trainer attributes so config-driven
        training works without the caller having to repeat every field."""
        tc = cfg.train
        self.epochs = tc.epochs
        self.batch_size = tc.batch_size
        self.lr = tc.learning_rate
        self.save_dir = tc.save_dir
        self.warmup_epochs = tc.warmup_epochs
        self.patience = getattr(tc, "patience", self.patience)
        self.pretrained_coco = getattr(tc, "pretrained_coco", self.pretrained_coco)
        self.amp = getattr(tc, "amp", self.amp)
        self.multi_gpu = getattr(tc, "multi_gpu", self.multi_gpu)
        self.grad_accum = max(1, getattr(tc, "grad_accum", self.grad_accum))

        if tc.resume:
            self.resume = tc.resume

        # LoRA
        self.lora = getattr(tc, "use_lora", self.lora)
        self.lora_variant = getattr(tc, "lora_variant", self.lora_variant)
        self.lora_rank = getattr(tc, "lora_rank", self.lora_rank)
        self.lora_alpha = getattr(tc, "lora_alpha", self.lora_alpha)
        self.lora_dropout = getattr(tc, "lora_dropout", self.lora_dropout)
        self.lora_targets = getattr(tc, "lora_target_modules", self.lora_targets)

        # QLoRA
        self.qlora = getattr(tc, "use_qlora", self.qlora)
        self.qlora_dtype = getattr(tc, "qlora_quant_dtype", self.qlora_dtype)

        # torchtune optimizations
        self.activation_checkpointing = getattr(tc, "enable_activation_checkpointing", self.activation_checkpointing)
        self.activation_offloading = getattr(tc, "enable_activation_offloading", self.activation_offloading)
        self.optimizer_in_bwd = getattr(tc, "optimizer_in_bwd", self.optimizer_in_bwd)
        self.use_8bit_optimizer = getattr(tc, "use_8bit_optimizer", self.use_8bit_optimizer)
        self.compile = getattr(tc, "compile_model", self.compile)
        self.chunked_loss = getattr(tc, "chunked_cross_entropy", self.chunked_loss)
        self.chunk_size = getattr(tc, "ce_chunk_size", self.chunk_size)

        # Model config
        mc = cfg.model
        self.architecture = getattr(mc, "architecture", self.architecture)
        if mc.backbone_size in ("0.5x", "1.0x", "1.5x"):
            size_map = {"0.5x": "m-0.5x", "1.0x": "m", "1.5x": "m-1.5x"}
            self.model_size = size_map[mc.backbone_size]
        self.input_size = mc.input_size if isinstance(mc.input_size, tuple) else (mc.input_size, mc.input_size)

        # Augmentations
        self.mosaic = getattr(tc, "mosaic", self.mosaic)
        self.mixup = getattr(tc, "mixup", self.mixup)
        self.copy_paste = getattr(tc, "copy_paste", self.copy_paste)

        # Data
        dc = cfg.data
        if dc.train_images:
            self.train_images = dc.train_images
        if dc.val_images:
            self.val_images = dc.val_images
        self.workers = dc.num_workers

    # ------------------------------------------------------------------
    # Callback API
    # ------------------------------------------------------------------

    def add_callback(self, callback: Callback) -> None:
        """Register a training callback."""
        self.callbacks.add(callback)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> Dict[str, float]:
        """Run the full training loop. Returns dict with best_map50 and best_loss."""
        cfg = self._config

        if self.train_images:
            cfg.data.train_images = self.train_images
            cfg.data.train_annotations = os.path.join(self.train_images, "_annotations.coco.json")
        if self.val_images:
            cfg.data.val_images = self.val_images
            cfg.data.val_annotations = os.path.join(self.val_images, "_annotations.coco.json")

        class_names = self._resolve_class_names(cfg)
        num_classes = len(class_names)

        self._logger.info("=" * 60)
        self._logger.info("FlashDet Training")
        self._logger.info("=" * 60)
        self._logger.info(f"Device: {self.device}")
        self._logger.info(f"Model: {self.model_size}, Input: {self.input_size}")
        self._logger.info(f"Epochs: {self.epochs}, Batch: {self.batch_size}, LR: {self.lr}")
        self._logger.info(f"Classes ({num_classes}): {class_names}")

        # Dataset verification
        data_root = os.path.dirname(os.path.normpath(cfg.data.train_images))
        if not verify_dataset(data_root):
            self._logger.error("Dataset not found! Please download it first.")
            raise FileNotFoundError(f"Dataset not found at {data_root}")

        # Data loaders
        train_loader = create_dataloader(
            img_dir=cfg.data.train_images,
            ann_file=cfg.data.train_annotations,
            batch_size=self.batch_size,
            input_size=self.input_size,
            num_workers=self.workers,
            is_train=True,
            mosaic=self.mosaic,
            mixup=self.mixup,
            copy_paste=self.copy_paste,
        )
        val_loader = create_dataloader(
            img_dir=cfg.data.val_images,
            ann_file=cfg.data.val_annotations,
            batch_size=self.batch_size,
            input_size=self.input_size,
            num_workers=self.workers,
            is_train=False,
        )

        # Build model
        arch = self.architecture.lower()
        if arch in ("flashdet", ""):
            size_key = resolve_flashdet_size(self.model_size)
            model = FlashDet(
                num_classes=num_classes,
                size=size_key,
                total_epochs=self.epochs,
                backbone_type=self.backbone_type,
            ).to(self.device)
        else:
            cfg.model.num_classes = num_classes
            model = build_model(cfg, architecture=arch).to(self.device)
            self._logger.info(f"Architecture: {arch}")

        # Apply LoRA / QLoRA
        model = self._apply_lora(model)

        # Fine-tune / pretrained COCO
        self._load_pretrained(model, cfg)

        # Chunked loss
        if self.chunked_loss:
            head = getattr(model, "head", None)
            if head is not None:
                head.use_chunked_loss = True
                head.chunk_size = self.chunk_size

        # AMP
        scaler = None
        if self.amp and self.device.type == "cuda":
            scaler = torch.amp.GradScaler("cuda", enabled=True)
            self._logger.info("AMP enabled")

        # Multi-GPU
        use_multi_gpu = self.multi_gpu and torch.cuda.device_count() > 1
        if use_multi_gpu:
            model = nn.DataParallel(model)

        raw_model = model.module if use_multi_gpu else model
        self._post_build_model(raw_model)

        # torchtune optimizations
        if self.activation_checkpointing:
            apply_activation_checkpointing(raw_model)
        offload_hook = None
        if self.activation_offloading:
            offload_hook = ActivationOffloadHook()
            offload_hook.register(raw_model)
        if self.compile:
            raw_model = torchtune_compile(raw_model)
            if not use_multi_gpu:
                model = raw_model

        # Optimizer
        optimizer = create_optimizer(
            model, lr=self.lr, weight_decay=cfg.train.weight_decay,
            use_8bit=self.use_8bit_optimizer, optimizer_in_bwd=self.optimizer_in_bwd,
            betas=(0.9, 0.999),
        )

        # LR schedule (Ultralytics style: cosine from lr to lr*lrf)
        lrf = getattr(self, 'lrf', 0.01)

        def _one_cycle(y1=0.0, y2=1.0, steps=100):
            return lambda x: max((1 - math.cos(x * math.pi / steps)) / 2, 0) * (y2 - y1) + y1

        lf = _one_cycle(1, lrf, self.epochs)
        self._lf = lf

        scheduler = None
        if not self.optimizer_in_bwd:
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

        # EMA
        ema = ModelEMA(raw_model, decay=0.9998, warmup=2000)

        # Resume
        start_epoch = 0
        best_loss = float("inf")
        best_map50 = 0.0

        if self.resume:
            ckpt = load_checkpoint(raw_model, self.resume, optimizer, scheduler, self.device)
            start_epoch = ckpt["epoch"] + 1
            best_loss = ckpt.get("loss", float("inf"))
            raw_ckpt = torch.load(self.resume, map_location=self.device, weights_only=False)
            if raw_ckpt and "ema_state_dict" in raw_ckpt:
                ema.load_state_dict(raw_ckpt["ema_state_dict"])
            else:
                ema = ModelEMA(raw_model, decay=0.9998, warmup=2000)
            self._logger.info(f"Resumed from epoch {start_epoch}")

        model_config = {
            "num_classes": num_classes,
            "input_size": self.input_size,
            "model_size": resolve_flashdet_size(self.model_size),
            "backbone_size": self._model_cfg["backbone"],
            "fpn_channels": self._model_cfg.get("fpn_channels"),
            "class_names": class_names,
            "architecture": self.architecture,
        }

        # Iteration-based warmup (Ultralytics style)
        nb = len(train_loader)
        nw = max(round(self.warmup_epochs * nb), 100) if self.warmup_epochs > 0 else -1
        warmup_momentum = 0.8
        base_momentum = 0.937

        self._logger.info(f"\nStarting training...")
        self._logger.info(f"LR schedule: {self.lr} -> {self.lr * lrf} (cosine)")
        self._logger.info(f"Warmup: {nw} iterations ({self.warmup_epochs} epochs)")
        epochs_without_improvement = 0
        self.callbacks.fire("on_train_start", self)

        for epoch in range(start_epoch, self.epochs):
            if scheduler is not None:
                scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]

            self._logger.info(f"\nEpoch {epoch + 1}/{self.epochs} (lr={current_lr:.6f})")
            self.callbacks.fire("on_epoch_start", self, epoch + 1)

            train_losses = self._train_one_epoch(
                model, train_loader, optimizer, epoch, ema, scaler,
                nb=nb, nw=nw, warmup_momentum=warmup_momentum,
                base_momentum=base_momentum, lf=lf,
            )

            epoch_metrics = {"train_loss": train_losses["loss"], "lr": current_lr}

            # Validate every epoch (Ultralytics style)
            self.callbacks.fire("on_val_start", self)
            val_loss, map50 = self._validate(
                raw_model, val_loader, ema, class_names,
            )
            epoch_metrics["val_loss"] = val_loss
            epoch_metrics["val_mAP"] = map50
            self.callbacks.fire("on_val_end", self, {"val_loss": val_loss, "val_mAP": map50})

            if val_loss < best_loss:
                best_loss = val_loss

            if map50 > best_map50:
                best_map50 = map50
                epochs_without_improvement = 0
                best_path = os.path.join(self.save_dir, "checkpoint_best.pth")
                save_checkpoint(
                    raw_model, optimizer, epoch, val_loss,
                    best_path,
                    scheduler=scheduler, config=model_config,
                    ema=ema,
                )
                save_inference_weights(
                    ema.ema,
                    os.path.join(self.save_dir, "model_best_inference.pth"),
                    config=model_config, half=False,
                )
                self._logger.info(f"  Best model saved (mAP@0.5: {best_map50:.4f})")
                self.callbacks.fire("on_checkpoint", self, best_path, True)
            else:
                epochs_without_improvement += 1

            if self.patience > 0 and epochs_without_improvement >= self.patience:
                self._logger.info(f"Early stopping at epoch {epoch + 1} (patience={self.patience})")
                break

            self.callbacks.fire("on_epoch_end", self, epoch + 1, epoch_metrics)

            # Check callback-driven early stopping
            for cb in self.callbacks.callbacks:
                if getattr(cb, "should_stop", False):
                    self._logger.info("Stopping training (callback request).")
                    break
            else:
                # Save latest
                last_path = os.path.join(self.save_dir, "checkpoint_last.pth")
                save_checkpoint(
                    raw_model, optimizer, epoch, train_losses["loss"],
                    last_path,
                    scheduler=scheduler, config=model_config, ema=ema,
                )
                save_inference_weights(
                    ema.ema,
                    os.path.join(self.save_dir, "model_last_inference.pth"),
                    config=model_config, half=False,
                )
                self.callbacks.fire("on_checkpoint", self, last_path, False)
                continue
            break

        # Final save
        if self.lora or self.qlora:
            lora_path = os.path.join(self.save_dir, "lora_adapters.pth")
            torch.save(get_lora_state_dict(ema.ema), lora_path)
            merge_lora_weights(ema.ema)

        save_inference_weights(
            ema.ema,
            os.path.join(self.save_dir, "model_final_inference.pth"),
            config=model_config, half=False,
        )
        save_inference_weights(
            ema.ema,
            os.path.join(self.save_dir, "model_final_fp16.pth"),
            config=model_config, half=True,
        )

        if offload_hook is not None:
            offload_hook.remove()

        final_metrics = {"best_map50": best_map50, "best_loss": best_loss}
        self.callbacks.fire("on_train_end", self, final_metrics)

        self._logger.info("=" * 60)
        self._logger.info("Training Complete!")
        self._logger.info(f"Best mAP@0.5: {best_map50:.4f}  |  Best Loss: {best_loss:.4f}")
        self._logger.info("=" * 60)

        return final_metrics

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _post_build_model(self, raw_model: nn.Module) -> None:
        """Hook for subclasses to attach extra trainable modules to the model."""

    def _resolve_class_names(self, cfg) -> List[str]:
        class_names = None
        if self.class_file:
            with open(self.class_file, encoding="utf-8") as f:
                class_names = [line.strip() for line in f if line.strip()]
        if not class_names:
            class_names = self._load_class_names_from_ann(cfg.data.train_annotations)
        if not class_names:
            class_names = cfg.class_names
        return class_names

    @staticmethod
    def _load_class_names_from_ann(ann_file: str) -> List[str]:
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

    def _apply_lora(self, model: nn.Module) -> nn.Module:
        if self.qlora:
            model = apply_qlora(
                model, rank=self.lora_rank, alpha=self.lora_alpha,
                dropout=self.lora_dropout, target_modules=self.lora_targets,
                quant_dtype=self.qlora_dtype, variant=self.lora_variant,
            )
            self._logger.info(f"QLoRA applied (rank={self.lora_rank})")
        elif self.lora:
            model = apply_lora(
                model, rank=self.lora_rank, alpha=self.lora_alpha,
                dropout=self.lora_dropout, target_modules=self.lora_targets,
                variant=self.lora_variant,
            )
            self._logger.info(f"LoRA applied (rank={self.lora_rank})")
        return model

    def _load_pretrained(self, model: nn.Module, cfg):
        if self.finetune and not self.resume:
            ckpt = torch.load(self.finetune, map_location=self.device, weights_only=False)
            src_sd = ckpt.get("model_state_dict", ckpt)
            src_sd = {k: v.float() if v.is_floating_point() else v for k, v in src_sd.items()}
            model.load_state_dict(src_sd, strict=False)
            self._logger.info(f"Fine-tune weights loaded from: {self.finetune}")
        elif self.pretrained_coco and not self.resume and not self.finetune:
            if self._model_cfg["backbone"] == "0.5x":
                self._logger.warning("COCO pretrained not available for 0.5x model.")
            else:
                try:
                    load_coco_pretrained(
                        model,
                        backbone_size=self._model_cfg["backbone"],
                        fpn_channels=self._model_cfg["fpn_channels"],
                        input_size=self.input_size[0],
                        checkpoint_path=self.pretrained_ckpt,
                    )
                    self._logger.info("COCO pretrained weights loaded.")
                except ValueError as e:
                    self._logger.warning(f"COCO pretrained unavailable: {e}")

    def _train_one_epoch(self, model, dataloader, optimizer, epoch, ema, scaler,
                         nb=None, nw=-1, warmup_momentum=0.8, base_momentum=0.937,
                         lf=None):
        model.train()
        use_amp = scaler is not None
        loss_meter = AverageMeter("Loss")
        sub_meters = {}
        raw_model = model.module if hasattr(model, "module") else model
        if nb is None:
            nb = len(dataloader)

        for batch_idx, (images, gt_meta) in enumerate(dataloader):
            # Iteration-based warmup (Ultralytics style)
            ni = batch_idx + nb * epoch
            if ni <= nw:
                xi = [0, nw]
                for pg in optimizer.param_groups:
                    pg['lr'] = np.interp(
                        ni, xi, [0.0, pg.get('initial_lr', self.lr) * (lf(epoch) if lf else 1.0)]
                    )
                    if 'momentum' in pg:
                        pg['momentum'] = np.interp(ni, xi, [warmup_momentum, base_momentum])

            self.callbacks.fire("on_batch_start", self, batch_idx, (images, gt_meta))
            images = images.to(self.device)

            with torch.amp.autocast(self.device.type, enabled=use_amp):
                output = model(images, gt_meta, epoch=epoch + 1)
                loss = output["loss"] / self.grad_accum

            if torch.isnan(loss):
                continue

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % self.grad_accum == 0 or (batch_idx + 1) == len(dataloader):
                if scaler:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(raw_model.parameters(), 10.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(raw_model.parameters(), 10.0)
                    optimizer.step()
                optimizer.zero_grad()

                if ema is not None:
                    ema.update(raw_model)

            loss_meter.update(output["loss"].item())
            loss_states = output.get("loss_states", {})
            for key, val in loss_states.items():
                if key.endswith("num_pos"):
                    continue
                if key not in sub_meters:
                    sub_meters[key] = AverageMeter(key)
                v = val.item() if hasattr(val, "item") else float(val)
                sub_meters[key].update(v)
            self.callbacks.fire("on_batch_end", self, batch_idx, output["loss"].item())

            if (batch_idx + 1) % 20 == 0:
                cur_lr = optimizer.param_groups[0]['lr']
                sub_str = ", ".join(f"{k}: {m.avg:.4f}" for k, m in sub_meters.items())
                self._logger.info(
                    f"  [{batch_idx+1}/{len(dataloader)}] Loss: {loss_meter.avg:.4f} LR: {cur_lr:.6f}"
                    + (f" ({sub_str})" if sub_str else "")
                )

        result = {"loss": loss_meter.avg}
        for key, meter in sub_meters.items():
            result[key] = meter.avg
        return result

    @torch.no_grad()
    def _validate(self, model, dataloader, ema, class_names):
        eval_model = ema.ema if ema is not None else model
        eval_model.eval()

        loss_meter = AverageMeter("Loss")
        sub_meters = {}
        all_preds, all_gts = [], []

        for images, gt_meta in dataloader:
            images = images.to(self.device)

            out = eval_model(images, gt_meta, epoch=0, compute_loss=True)

            if "loss" in out:
                loss_meter.update(out["loss"].item())
            loss_states = out.get("loss_states", {})
            for key, val in loss_states.items():
                if key.endswith("num_pos"):
                    continue
                if key not in sub_meters:
                    sub_meters[key] = AverageMeter(key)
                v = val.item() if hasattr(val, "item") else float(val)
                sub_meters[key].update(v)

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

        sub_str = ", ".join(f"{k}: {m.avg:.4f}" for k, m in sub_meters.items())
        self._logger.info(
            f"  Val Loss: {loss_meter.avg:.4f}"
            + (f" ({sub_str})" if sub_str else "")
            + f" | mAP@0.5: {map50:.4f}"
        )

        model.train()
        return loss_meter.avg, map50
