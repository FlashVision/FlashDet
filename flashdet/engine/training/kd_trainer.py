"""Knowledge Distillation Trainer for FlashDet.

Trains a smaller student model by distilling knowledge from a larger
teacher model. Supports both logit-level and feature-level distillation.

Usage::

    from flashdet.engine import KDTrainer

    trainer = KDTrainer(
        teacher_checkpoint="workspace/teacher/model_best_inference.pth",
        teacher_size="m-1.5x",
        model_size="m-0.5x",
        kd_temperature=4.0,
        kd_alpha=0.5,
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


class KDTrainer(Trainer):
    """Knowledge Distillation trainer.

    Extends the base ``Trainer`` with teacher-student distillation logic.
    The teacher model is frozen and used to generate soft targets that the
    student is trained to match alongside the standard detection loss.

    Args:
        teacher_checkpoint: Path to the teacher model checkpoint.
        teacher_size: Teacher model size key (e.g. "m-1.5x").
        teacher_architecture: Architecture of the teacher model.
        kd_temperature: Softmax temperature for logit distillation.
        kd_alpha: Weight blending student loss vs KD loss (0=pure student, 1=pure KD).
        kd_feature_weight: Weight for intermediate feature distillation loss.
        kd_feature_layers: Which backbone stages to distill features from.
        **kwargs: Forwarded to :class:`Trainer`.
    """

    def __init__(
        self,
        teacher_checkpoint: str,
        teacher_size: str = "m-1.5x",
        teacher_architecture: str = "flashdet",
        kd_temperature: float = 4.0,
        kd_alpha: float = 0.5,
        kd_feature_weight: float = 1.0,
        kd_feature_layers: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.teacher_checkpoint = teacher_checkpoint
        self.teacher_size = teacher_size
        self.teacher_architecture = teacher_architecture
        self.kd_temperature = kd_temperature
        self.kd_alpha = kd_alpha
        self.kd_feature_weight = kd_feature_weight
        self.kd_feature_layers = kd_feature_layers or ["stage2", "stage3", "stage4"]
        self._teacher_model = None

    def train(self):
        """Override to build teacher before starting the training loop."""
        cfg = self._config
        if self.train_images:
            cfg.data.train_images = self.train_images
            cfg.data.train_annotations = os.path.join(self.train_images, "_annotations.coco.json")
        class_names = self._resolve_class_names(cfg)
        num_classes = len(class_names)
        self._teacher_model = self._build_teacher(num_classes)
        return super().train()

    def _build_teacher(self, num_classes: int) -> nn.Module:
        """Build and freeze the teacher model."""
        arch = self.teacher_architecture.lower()
        if arch in ("flashdet", ""):
            size_key = {"m": "n", "m-0.5x": "n", "m-1.5x": "s"}.get(self.teacher_size, "n")
            teacher = FlashDet(
                num_classes=num_classes,
                size=size_key,
                total_epochs=self.epochs,
            )
        else:
            from flashdet.cfg import get_config
            cfg = get_config()
            cfg.model.num_classes = num_classes
            teacher = build_model(cfg, architecture=arch)

        ckpt = torch.load(self.teacher_checkpoint, map_location=self.device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        teacher.load_state_dict(state, strict=False)
        teacher = teacher.to(self.device).eval()

        for p in teacher.parameters():
            p.requires_grad = False

        self._logger.info(f"Teacher loaded from {self.teacher_checkpoint} ({self.teacher_size})")
        return teacher

    def _kd_logit_loss(
        self, student_logits: torch.Tensor, teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """KL-divergence loss between student/teacher logit distributions."""
        T = self.kd_temperature
        s_log = F.log_softmax(student_logits / T, dim=-1)
        t_soft = F.softmax(teacher_logits / T, dim=-1)
        return F.kl_div(s_log, t_soft, reduction="batchmean") * (T * T)

    def _kd_feature_loss(
        self, student_feats: List[torch.Tensor], teacher_feats: List[torch.Tensor],
    ) -> torch.Tensor:
        """L2 feature distillation loss (with spatial alignment)."""
        loss = torch.tensor(0.0, device=self.device)
        for sf, tf in zip(student_feats, teacher_feats):
            if sf.shape != tf.shape:
                tf = F.interpolate(tf, size=sf.shape[2:], mode="bilinear", align_corners=False)
                if tf.shape[1] != sf.shape[1]:
                    continue
            loss = loss + F.mse_loss(sf, tf)
        return loss / max(len(student_feats), 1)

    def _train_one_epoch(self, model, dataloader, optimizer, epoch, ema, scaler):
        """Override to inject KD loss alongside detection loss."""
        model.train()
        teacher = self._teacher_model
        use_amp = scaler is not None

        loss_meter = AverageMeter("Loss")
        kd_meter = AverageMeter("KD")
        det_meter = AverageMeter("Det")
        raw_model = model.module if hasattr(model, "module") else model

        for batch_idx, (images, gt_meta) in enumerate(dataloader):
            self.callbacks.fire("on_batch_start", self, batch_idx, (images, gt_meta))
            images = images.to(self.device)

            with torch.amp.autocast(self.device.type, enabled=use_amp):
                student_out = model(images, gt_meta, epoch=epoch, return_features=True)
                det_loss = student_out["loss"]

                with torch.no_grad():
                    teacher_out = teacher(images, gt_meta, epoch=epoch, return_features=True)

                kd_loss = torch.tensor(0.0, device=self.device)
                s_logits = student_out.get("o2o_cls", student_out.get("preds"))
                t_logits = teacher_out.get("o2o_cls", teacher_out.get("preds"))
                if s_logits is not None and t_logits is not None:
                    kd_loss = kd_loss + self._kd_logit_loss(s_logits, t_logits)

                s_feats = student_out.get("fpn_features", student_out.get("backbone_features", []))
                t_feats = teacher_out.get("fpn_features", teacher_out.get("backbone_features", []))
                if s_feats and t_feats and self.kd_feature_weight > 0:
                    kd_loss = kd_loss + self.kd_feature_weight * self._kd_feature_loss(s_feats, t_feats)

                loss = (1 - self.kd_alpha) * det_loss + self.kd_alpha * kd_loss
                loss = loss / self.grad_accum

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

            total_loss_val = det_loss.item() + kd_loss.item()
            loss_meter.update(total_loss_val)
            det_meter.update(det_loss.item())
            kd_meter.update(kd_loss.item())
            self.callbacks.fire("on_batch_end", self, batch_idx, total_loss_val)

            if (batch_idx + 1) % 10 == 0:
                self._logger.info(
                    f"  [{batch_idx+1}/{len(dataloader)}] "
                    f"Loss: {loss_meter.avg:.4f} (Det: {det_meter.avg:.4f}, KD: {kd_meter.avg:.4f})"
                )

        return {"loss": loss_meter.avg, "det_loss": det_meter.avg, "kd_loss": kd_meter.avg}
