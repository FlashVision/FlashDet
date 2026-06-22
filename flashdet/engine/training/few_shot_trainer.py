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
