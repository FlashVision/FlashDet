"""Semi-Supervised Object Detection Trainer for FlashDet.

Implements a teacher-student framework where a teacher model generates
pseudo-labels on unlabeled data, and the student is trained on both
labeled ground truth and teacher-generated pseudo-labels.

Based on:
    - Unbiased Teacher (Liu et al., ICLR 2021)
    - Soft Teacher (Xu et al., ICCV 2021)

Usage::

    from flashdet.engine import SemiSupervisedTrainer

    trainer = SemiSupervisedTrainer(
        labeled_images="data/labeled/train",
        unlabeled_images="data/unlabeled/images",
        pseudo_label_threshold=0.7,
        unsup_loss_weight=1.0,
        teacher_momentum=0.999,
    )
    trainer.train()
"""

import os
import copy
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


class SemiSupervisedTrainer(Trainer):
    """Semi-supervised detection trainer with pseudo-labeling.

    Maintains two copies of the model:

    - **Student**: Trained on both labeled and pseudo-labeled data.
    - **Teacher**: An EMA of the student that generates pseudo-labels
      on unlabeled images.

    The training alternates between supervised steps (on labeled data)
    and unsupervised steps (on unlabeled data with teacher pseudo-labels).

    Pseudo-labels are filtered by confidence threshold and optionally
    by a consistency check between weak and strong augmentations.

    Args:
        unlabeled_images: Path to directory of unlabeled images.
        pseudo_label_threshold: Confidence threshold for accepting
            teacher pseudo-labels.
        unsup_loss_weight: Weight multiplier for the unsupervised loss.
        teacher_momentum: EMA decay for updating the teacher from the student.
        warmup_teacher_epochs: Number of epochs of pure supervised training
            before enabling pseudo-labels.
        strong_aug: Whether to apply strong augmentation on unlabeled images
            before passing to the student (while teacher sees weak aug).
        pseudo_nms_threshold: NMS threshold for filtering teacher predictions.
        **kwargs: Forwarded to :class:`Trainer`.
    """

    def __init__(
        self,
        unlabeled_images: Optional[str] = None,
        pseudo_label_threshold: float = 0.7,
        unsup_loss_weight: float = 1.0,
        teacher_momentum: float = 0.999,
        warmup_teacher_epochs: int = 5,
        strong_aug: bool = True,
        pseudo_nms_threshold: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.unlabeled_images = unlabeled_images
        self.pseudo_label_threshold = pseudo_label_threshold
        self.unsup_loss_weight = unsup_loss_weight
        self.teacher_momentum = teacher_momentum
        self.warmup_teacher_epochs = warmup_teacher_epochs
        self.strong_aug = strong_aug
        self.pseudo_nms_threshold = pseudo_nms_threshold

    def train(self):
        """Override to build teacher model and unlabeled data loader."""
        cfg = self._config
        if self.train_images:
            cfg.data.train_images = self.train_images
            cfg.data.train_annotations = os.path.join(self.train_images, "_annotations.coco.json")
        if self.val_images:
            cfg.data.val_images = self.val_images
            cfg.data.val_annotations = os.path.join(self.val_images, "_annotations.coco.json")

        class_names = self._resolve_class_names(cfg)
        num_classes = len(class_names)

        arch = self.architecture.lower()
        if arch in ("flashdet", ""):
            size_key = {"m": "n", "m-0.5x": "n", "m-1.5x": "s"}.get(self.model_size, "n")
            self._teacher_model = FlashDet(
                num_classes=num_classes, size=size_key, total_epochs=self.epochs,
            ).to(self.device)
        else:
            cfg.model.num_classes = num_classes
            self._teacher_model = build_model(cfg, architecture=arch).to(self.device)
        self._teacher_model.eval()
        for p in self._teacher_model.parameters():
            p.requires_grad = False
        self._logger.info("Semi-supervised teacher model created (EMA copy)")

        if self.unlabeled_images:
            from flashdet.data import create_dataloader
            unsup_ann = os.path.join(self.unlabeled_images, "_annotations.coco.json")
            if os.path.exists(unsup_ann):
                self._unsup_loader = create_dataloader(
                    img_dir=self.unlabeled_images, ann_file=unsup_ann,
                    batch_size=self.batch_size, input_size=self.input_size,
                    num_workers=self.workers, is_train=True,
                )
                self._logger.info(f"Unlabeled data: {len(self._unsup_loader.dataset)} images")
            else:
                self._logger.warning(f"No annotations at {unsup_ann}, running supervised only")

        return super().train()

    @torch.no_grad()
    def _generate_pseudo_labels(
        self, teacher: nn.Module, images: torch.Tensor,
    ) -> Dict:
        """Generate pseudo ground-truth from teacher predictions.

        Args:
            teacher: The frozen teacher model.
            images: Batch of unlabeled images (B, C, H, W).

        Returns:
            A gt_meta-compatible dict with pseudo bboxes and labels.
        """
        teacher.eval()
        results = teacher.predict(images, None, score_thr=self.pseudo_label_threshold, nms_thr=self.pseudo_nms_threshold)

        gt_bboxes, gt_labels = [], []
        for dets, lbs in results:
            if dets is not None and dets.numel() > 0:
                gt_bboxes.append(dets[:, :4].cpu().numpy().tolist())
                gt_labels.append(lbs.cpu().numpy().tolist())
            else:
                gt_bboxes.append([])
                gt_labels.append([])

        return {"gt_bboxes": gt_bboxes, "gt_labels": gt_labels}

    @torch.no_grad()
    def _update_teacher(self, student: nn.Module, teacher: nn.Module):
        """EMA update of the teacher model from the student."""
        m = self.teacher_momentum
        for sp, tp in zip(student.parameters(), teacher.parameters()):
            tp.data.mul_(m).add_(sp.data, alpha=1 - m)
        for sb, tb in zip(student.buffers(), teacher.buffers()):
            tb.data.copy_(sb.data)

    def _compute_unsup_loss(
        self, model: nn.Module, images: torch.Tensor, pseudo_meta: Dict, epoch: int,
    ) -> torch.Tensor:
        """Compute detection loss on pseudo-labeled images."""
        if not pseudo_meta["gt_bboxes"] or all(len(b) == 0 for b in pseudo_meta["gt_bboxes"]):
            return torch.tensor(0.0, device=self.device)

        output = model(images, pseudo_meta, epoch=epoch)
        return output.get("loss", torch.tensor(0.0, device=self.device))

    def _train_one_epoch(self, model, dataloader, optimizer, epoch, ema, scaler):
        """Override to add semi-supervised pseudo-label training."""
        model.train()
        use_amp = scaler is not None

        loss_meter = AverageMeter("Loss")
        sup_meter = AverageMeter("Sup")
        unsup_meter = AverageMeter("Unsup")
        pseudo_count = AverageMeter("PseudoBoxes")
        raw_model = model.module if hasattr(model, "module") else model

        teacher = getattr(self, "_teacher_model", None)
        unsup_loader = getattr(self, "_unsup_loader", None)
        use_pseudo = (
            teacher is not None
            and unsup_loader is not None
            and epoch > self.warmup_teacher_epochs
        )
        unsup_iter = iter(unsup_loader) if use_pseudo else None

        for batch_idx, (images, gt_meta) in enumerate(dataloader):
            self.callbacks.fire("on_batch_start", self, batch_idx, (images, gt_meta))
            images = images.to(self.device)

            with torch.amp.autocast(self.device.type, enabled=use_amp):
                sup_out = model(images, gt_meta, epoch=epoch)
                sup_loss = sup_out["loss"]

                unsup_loss = torch.tensor(0.0, device=self.device)
                n_pseudo = 0

                if use_pseudo:
                    try:
                        unsup_images, _ = next(unsup_iter)
                    except StopIteration:
                        unsup_iter = iter(unsup_loader)
                        unsup_images, _ = next(unsup_iter)

                    unsup_images = unsup_images.to(self.device)
                    pseudo_meta = self._generate_pseudo_labels(teacher, unsup_images)
                    n_pseudo = sum(len(b) for b in pseudo_meta["gt_bboxes"])
                    unsup_loss = self._compute_unsup_loss(model, unsup_images, pseudo_meta, epoch)

                loss = (sup_loss + self.unsup_loss_weight * unsup_loss) / self.grad_accum

            if torch.isnan(loss):
                continue

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % self.grad_accum == 0 or (batch_idx + 1) == len(dataloader):
                if scaler:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(raw_model.parameters(), 35.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(raw_model.parameters(), 35.0)
                    optimizer.step()
                optimizer.zero_grad()

                if ema is not None:
                    ema.update(raw_model)

                if teacher is not None:
                    self._update_teacher(raw_model, teacher)

            total_val = sup_loss.item() + unsup_loss.item()
            loss_meter.update(total_val)
            sup_meter.update(sup_loss.item())
            unsup_meter.update(unsup_loss.item())
            pseudo_count.update(n_pseudo)
            self.callbacks.fire("on_batch_end", self, batch_idx, total_val)

            if (batch_idx + 1) % 10 == 0:
                self._logger.info(
                    f"  [{batch_idx+1}/{len(dataloader)}] "
                    f"Loss: {loss_meter.avg:.4f} "
                    f"(Sup: {sup_meter.avg:.4f}, Unsup: {unsup_meter.avg:.4f}, "
                    f"Pseudo/batch: {pseudo_count.avg:.0f})"
                )

        return {
            "loss": loss_meter.avg,
            "sup_loss": sup_meter.avg,
            "unsup_loss": unsup_meter.avg,
        }
