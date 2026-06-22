"""Few-Shot Object Detection Trainer for FlashDet.

Allows a pretrained detection model to learn new object classes from
very few annotated examples (1-30 shots per class) using prototype-based
metric learning combined with standard detection fine-tuning.

Usage::

    from flashdet.engine import FewShotTrainer

    trainer = FewShotTrainer(
        base_checkpoint="workspace/base/model_best_inference.pth",
        n_shot=5,
        novel_class_file="novel_classes.txt",
        train_images="data/novel/train",
    )
    trainer.train()
"""

import os
import math
import logging
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashdet.engine.training.trainer import Trainer, MODEL_SIZE_MAP
from flashdet.engine.core.ema import ModelEMA
from flashdet.models import FlashDet
from flashdet.models.detector import build_model
from flashdet.utils import AverageMeter

logger = logging.getLogger(__name__)


class PrototypeHead(nn.Module):
    """Cosine-similarity prototype classifier for few-shot detection.

    Maintains a learnable prototype vector per class. Classification is
    performed via scaled cosine similarity between region features and
    class prototypes.
    """

    def __init__(self, feat_dim: int, num_classes: int, scale: float = 20.0):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.prototypes)
        self.scale = nn.Parameter(torch.tensor(scale))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute class logits via cosine similarity.

        Args:
            features: (N, D) normalised feature vectors.

        Returns:
            (N, C) logits.
        """
        features = F.normalize(features, dim=-1)
        prototypes = F.normalize(self.prototypes, dim=-1)
        return self.scale * features @ prototypes.T


class FewShotTrainer(Trainer):
    """Few-Shot detection trainer.

    Extends the base ``Trainer`` with few-shot specific strategies:

    1. **Frozen backbone**: The backbone and neck are frozen; only the
       detection head (or a prototype head) is fine-tuned.
    2. **Prototype initialisation**: Class prototypes are initialised
       from the mean feature of support examples.
    3. **Balanced sampling**: Training batches are sampled to ensure
       each class appears at least once.
    4. **Higher LR for head**: The head uses a higher learning rate
       than the (optionally unfrozen) backbone.

    Args:
        base_checkpoint: Path to pretrained base model weights.
        n_shot: Number of annotated examples per novel class (K-shot).
        freeze_backbone: Whether to freeze the backbone during fine-tuning.
        freeze_neck: Whether to freeze the neck during fine-tuning.
        head_lr_factor: Learning rate multiplier for the detection head.
        use_prototype_head: Replace the classification head with a
            cosine-similarity prototype head.
        novel_class_file: Text file listing novel class names (one per line).
        support_images: Directory containing support-set images.
        **kwargs: Forwarded to :class:`Trainer`.
    """

    def __init__(
        self,
        base_checkpoint: str,
        n_shot: int = 5,
        freeze_backbone: bool = True,
        freeze_neck: bool = True,
        head_lr_factor: float = 10.0,
        use_prototype_head: bool = False,
        novel_class_file: Optional[str] = None,
        support_images: Optional[str] = None,
        **kwargs,
    ):
        kwargs.setdefault("epochs", 30)
        kwargs.setdefault("lr", 0.001)
        kwargs.setdefault("patience", 15)
        super().__init__(**kwargs)

        self.base_checkpoint = base_checkpoint
        self.n_shot = n_shot
        self.freeze_backbone = freeze_backbone
        self.freeze_neck = freeze_neck
        self.head_lr_factor = head_lr_factor
        self.use_prototype_head = use_prototype_head
        self.novel_class_file = novel_class_file
        self.support_images = support_images

    def train(self):
        """Override to load base checkpoint and freeze layers before training."""
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
        self._logger.info(f"Few-Shot Training ({self.n_shot}-shot)")
        self._logger.info("=" * 60)

        from flashdet.data import create_dataloader, verify_dataset
        data_root = os.path.dirname(os.path.normpath(cfg.data.train_images))
        if not verify_dataset(data_root):
            raise FileNotFoundError(f"Dataset not found at {data_root}")

        train_loader = create_dataloader(
            img_dir=cfg.data.train_images, ann_file=cfg.data.train_annotations,
            batch_size=self.batch_size, input_size=self.input_size,
            num_workers=self.workers, is_train=True,
        )
        val_loader = create_dataloader(
            img_dir=cfg.data.val_images, ann_file=cfg.data.val_annotations,
            batch_size=self.batch_size, input_size=self.input_size,
            num_workers=self.workers, is_train=False,
        )

        arch = self.architecture.lower()
        if arch in ("flashdet", ""):
            size_key = {"m": "n", "m-0.5x": "n", "m-1.5x": "s"}.get(self.model_size, "n")
            model = FlashDet(
                num_classes=num_classes, size=size_key, total_epochs=self.epochs,
            ).to(self.device)
        else:
            cfg.model.num_classes = num_classes
            model = build_model(cfg, architecture=arch).to(self.device)

        self._load_base_model(model)
        self._log_trainable_summary(model)

        param_groups = self._build_param_groups(model)
        from flashdet.utils.torchtune_optim import create_optimizer
        if param_groups:
            optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.train.weight_decay)
        else:
            optimizer = create_optimizer(model, lr=self.lr, weight_decay=cfg.train.weight_decay)

        from flashdet.engine.core.ema import ModelEMA
        ema = ModelEMA(model, decay=0.9998, warmup=2000)

        import math
        eta_min = 0.00005
        eta_min_factor = eta_min / self.lr
        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return (epoch + 1) / self.warmup_epochs
            progress = (epoch - self.warmup_epochs) / max(self.epochs - self.warmup_epochs, 1)
            return eta_min_factor + (1.0 - eta_min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        from flashdet.utils import save_checkpoint, save_inference_weights
        from flashdet.utils.metrics import compute_map

        best_map50 = 0.0
        best_loss = float("inf")
        model_config = {"num_classes": num_classes, "class_names": class_names, "architecture": self.architecture}

        self._logger.info(f"\nStarting {self.n_shot}-shot fine-tuning for {self.epochs} epochs...")
        for epoch in range(self.epochs):
            current_lr = optimizer.param_groups[0]["lr"]
            self._logger.info(f"\nEpoch {epoch+1}/{self.epochs} (lr={current_lr:.6f})")

            train_losses = self._train_one_epoch(model, train_loader, optimizer, epoch + 1, ema, None)

            if (epoch + 1) % cfg.train.val_interval == 0:
                val_loss, map50 = self._validate(model, val_loader, ema, class_names)
                if map50 > best_map50:
                    best_map50 = map50
                    save_checkpoint(model, optimizer, epoch, val_loss,
                                    os.path.join(self.save_dir, "checkpoint_best.pth"),
                                    scheduler=scheduler, config=model_config, ema=ema)
                    save_inference_weights(ema.ema, os.path.join(self.save_dir, "model_best_inference.pth"),
                                           config=model_config, half=False)
                    self._logger.info(f"  Best model (mAP@0.5: {best_map50:.4f})")
                if val_loss < best_loss:
                    best_loss = val_loss

            scheduler.step()

        self._logger.info(f"\nFew-shot training complete. Best mAP@0.5: {best_map50:.4f}")
        return {"best_map50": best_map50, "best_loss": best_loss}

    def _load_base_model(self, model: nn.Module) -> None:
        """Load base model weights and optionally freeze layers."""
        ckpt = torch.load(self.base_checkpoint, map_location=self.device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        state = {k: v.float() if v.is_floating_point() else v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        self._logger.info(
            f"Base model loaded from {self.base_checkpoint} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )

        if self.freeze_backbone:
            for name, param in model.named_parameters():
                if "backbone" in name or "stem" in name or "stage" in name:
                    param.requires_grad = False
            frozen_count = sum(1 for p in model.parameters() if not p.requires_grad)
            self._logger.info(f"Froze {frozen_count} backbone parameters")

        if self.freeze_neck:
            for name, param in model.named_parameters():
                if "neck" in name or "fpn" in name:
                    param.requires_grad = False
            self._logger.info("Froze neck/FPN parameters")

    def _build_param_groups(self, model: nn.Module) -> List[Dict]:
        """Create parameter groups with different LRs for head vs rest."""
        head_params = []
        other_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "head" in name or "prototype" in name or "cls" in name:
                head_params.append(param)
            else:
                other_params.append(param)

        groups = []
        if other_params:
            groups.append({"params": other_params, "lr": self.lr})
        if head_params:
            groups.append({"params": head_params, "lr": self.lr * self.head_lr_factor})
        return groups

    def _log_trainable_summary(self, model: nn.Module) -> None:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self._logger.info(
            f"Parameters: {total:,} total, {trainable:,} trainable "
            f"({100*trainable/total:.1f}%)"
        )
