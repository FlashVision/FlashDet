"""
Knowledge Distillation losses for FlashDet.

Implements distillation losses inspired by torchtune's KD recipe, adapted
for object detection (feature-level and logit-level distillation).

Supported distillation modes:
  - **Logit KD**: KL-divergence between teacher and student classification
    logits (soft targets) + optional regression distillation.
  - **Feature KD**: L2 alignment between teacher and student FPN features
    after a lightweight adapter projection.
  - **Combined**: Both logit and feature KD with configurable weighting.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class LogitDistillationLoss(nn.Module):
    """Classification logit distillation via KL-divergence.

    Matches the teacher's soft class probability distribution using
    temperature-scaled KL divergence — the same formulation used in
    Hinton et al. (2015) and torchtune's KD recipe.

    For the regression branch, we use smooth-L1 between teacher and
    student distribution logits (DFL outputs).

    Args:
        temperature: Softmax temperature for KL divergence.
        cls_weight: Weight for the classification KD loss.
        reg_weight: Weight for the regression KD loss.
    """

    def __init__(
        self,
        temperature: float = 4.0,
        cls_weight: float = 1.0,
        reg_weight: float = 0.5,
    ):
        super().__init__()
        self.temperature = temperature
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight

    def forward(
        self,
        student_preds: torch.Tensor,
        teacher_preds: torch.Tensor,
        num_classes: int,
        reg_max: int = 7,
    ) -> dict:
        """Compute logit-level KD loss.

        Args:
            student_preds: [B, N, C] student head output.
            teacher_preds: [B, N, C] teacher head output (detached).
            num_classes: Number of object classes.
            reg_max: reg_max from the detection head.

        Returns:
            Dict with ``kd_cls_loss``, ``kd_reg_loss``, and ``kd_logit_loss``.
        """
        reg_channels = 4 * (reg_max + 1)
        s_cls, s_reg = student_preds.split([num_classes, reg_channels], dim=-1)
        t_cls, t_reg = teacher_preds.split([num_classes, reg_channels], dim=-1)

        T = self.temperature

        # Classification KD: KL(softmax(t/T) || softmax(s/T))
        s_log_probs = F.log_softmax(s_cls / T, dim=-1)
        t_probs = F.softmax(t_cls / T, dim=-1)
        kd_cls = F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (T * T)

        # Regression KD: smooth-L1 on the DFL distribution logits
        kd_reg = F.smooth_l1_loss(s_reg, t_reg, reduction="mean")

        kd_logit_loss = self.cls_weight * kd_cls + self.reg_weight * kd_reg

        return {
            "kd_cls_loss": kd_cls.detach(),
            "kd_reg_loss": kd_reg.detach(),
            "kd_logit_loss": kd_logit_loss,
        }


class FeatureDistillationLoss(nn.Module):
    """FPN feature distillation via L2 / cosine alignment.

    Aligns student FPN feature maps to teacher FPN features using an
    optional 1x1 conv adapter (when channel counts differ) and normalised
    L2 distance.  This is the channel-wise feature mimicking approach from
    "Mimicking Very Efficient Network for Object Detection" (Li et al.).

    Args:
        student_channels: Student FPN output channels.
        teacher_channels: Teacher FPN output channels.
        num_levels: Number of FPN levels to distill.
        loss_weight: Overall weighting for the feature KD loss.
    """

    def __init__(
        self,
        student_channels: int = 96,
        teacher_channels: int = 128,
        num_levels: int = 4,
        loss_weight: float = 0.5,
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.num_levels = num_levels

        self.adapters = nn.ModuleList()
        for _ in range(num_levels):
            if student_channels != teacher_channels:
                self.adapters.append(
                    nn.Conv2d(student_channels, teacher_channels, 1, bias=False)
                )
            else:
                self.adapters.append(nn.Identity())

    def forward(
        self,
        student_feats: list,
        teacher_feats: list,
    ) -> torch.Tensor:
        """Compute feature-level KD loss.

        Args:
            student_feats: List of student FPN feature maps.
            teacher_feats: List of teacher FPN feature maps (detached).

        Returns:
            Scalar feature distillation loss.
        """
        total_loss = 0.0
        n = min(len(student_feats), len(teacher_feats), self.num_levels)

        for i in range(n):
            s_feat = self.adapters[i](student_feats[i])
            t_feat = teacher_feats[i]

            if s_feat.shape[2:] != t_feat.shape[2:]:
                s_feat = F.adaptive_avg_pool2d(s_feat, t_feat.shape[2:])

            # Normalised L2 distance
            s_norm = F.normalize(s_feat, dim=1)
            t_norm = F.normalize(t_feat, dim=1)
            total_loss = total_loss + F.mse_loss(s_norm, t_norm)

        return self.loss_weight * total_loss / max(n, 1)


class KnowledgeDistillationLoss(nn.Module):
    """Combined knowledge distillation loss for object detection.

    Combines logit-level and feature-level distillation into a single
    module, inspired by torchtune's KD training recipe adapted for
    FlashDet.

    Args:
        temperature: KL divergence temperature.
        logit_weight: Weight for the logit KD component.
        feature_weight: Weight for the feature KD component.
        student_channels: Student FPN channels.
        teacher_channels: Teacher FPN channels.
        num_levels: Number of FPN feature levels.
    """

    def __init__(
        self,
        temperature: float = 4.0,
        logit_weight: float = 1.0,
        feature_weight: float = 0.5,
        student_channels: int = 96,
        teacher_channels: int = 128,
        num_levels: int = 4,
    ):
        super().__init__()
        self.logit_loss = LogitDistillationLoss(
            temperature=temperature,
            cls_weight=logit_weight,
            reg_weight=logit_weight * 0.5,
        )
        self.feature_loss = FeatureDistillationLoss(
            student_channels=student_channels,
            teacher_channels=teacher_channels,
            num_levels=num_levels,
            loss_weight=feature_weight,
        )
        self.logit_weight = logit_weight
        self.feature_weight = feature_weight

    def forward(
        self,
        student_preds: torch.Tensor,
        teacher_preds: torch.Tensor,
        student_fpn_feats: list,
        teacher_fpn_feats: list,
        num_classes: int,
        reg_max: int = 7,
    ) -> dict:
        """Compute the combined KD loss.

        Returns:
            Dict with all loss components and the combined ``kd_loss``.
        """
        result = {}

        if self.logit_weight > 0:
            logit_res = self.logit_loss(
                student_preds, teacher_preds, num_classes, reg_max
            )
            result.update(logit_res)
        else:
            result["kd_logit_loss"] = torch.tensor(0.0, device=student_preds.device)
            result["kd_cls_loss"] = torch.tensor(0.0, device=student_preds.device)
            result["kd_reg_loss"] = torch.tensor(0.0, device=student_preds.device)

        if self.feature_weight > 0 and student_fpn_feats and teacher_fpn_feats:
            feat_loss = self.feature_loss(student_fpn_feats, teacher_fpn_feats)
            result["kd_feature_loss"] = feat_loss
        else:
            feat_loss = torch.tensor(0.0, device=student_preds.device)
            result["kd_feature_loss"] = feat_loss

        result["kd_loss"] = result["kd_logit_loss"] + feat_loss

        return result
