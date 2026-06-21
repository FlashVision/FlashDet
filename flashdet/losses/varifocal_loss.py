"""
Varifocal Loss and Sigmoid Focal Loss.

Varifocal Loss treats positive and negative samples asymmetrically:
positives are supervised by their IoU-aware classification score (IACS)
while negatives use standard focal loss weighting.

Sigmoid Focal Loss is the standard focal loss from RetinaNet applied
with sigmoid activation.

References:
  - Zhang et al., "VarifocalNet: An IoU-aware Dense Object Detector", CVPR 2021.
  - Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def varifocal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
    weight: torch.Tensor = None,
    avg_factor: float = None,
) -> torch.Tensor:
    """Varifocal Loss functional.

    Positives: weighted by target quality score (soft label).
    Negatives: down-weighted by ``alpha * sigmoid^gamma`` (standard focal).

    Args:
        pred: [N, C] classification logits (before sigmoid).
        target: [N, C] soft targets (IoU quality for positives, 0 for negatives).
        alpha: Balancing factor for negatives.
        gamma: Focusing parameter.
        weight: Optional per-sample weight [N].
        avg_factor: Optional normaliser.

    Returns:
        Scalar loss tensor.
    """
    pred_sigmoid = pred.sigmoid()
    focal_weight = target * (target > 0).float() + \
        alpha * pred_sigmoid.pow(gamma) * (target == 0).float()

    bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    loss = (focal_weight * bce).sum(dim=-1)

    if weight is not None:
        loss = loss * weight

    if avg_factor is not None:
        return loss.sum() / max(avg_factor, 1.0)
    return loss.mean()


class VarifocalLoss(nn.Module):
    """Varifocal Loss module.

    Args:
        alpha: Balancing factor for negatives.
        gamma: Focusing parameter.
        loss_weight: Scalar multiplier applied to the final loss.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, loss_weight: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.loss_weight = loss_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor = None,
        avg_factor: float = None,
    ) -> torch.Tensor:
        return self.loss_weight * varifocal_loss(
            pred, target, alpha=self.alpha, gamma=self.gamma,
            weight=weight, avg_factor=avg_factor,
        )


def sigmoid_focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    weight: torch.Tensor = None,
    avg_factor: float = None,
) -> torch.Tensor:
    """Sigmoid Focal Loss functional (RetinaNet-style).

    Args:
        pred: [N, C] classification logits (before sigmoid).
        target: [N, C] one-hot or soft targets.
        alpha: Balancing factor. Use -1 to disable.
        gamma: Focusing parameter.
        weight: Optional per-sample weight [N].
        avg_factor: Optional normaliser.

    Returns:
        Scalar loss tensor.
    """
    pred_sigmoid = pred.sigmoid()
    ce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    p_t = pred_sigmoid * target + (1 - pred_sigmoid) * (1 - target)
    focal_weight = (1 - p_t) ** gamma

    if alpha >= 0:
        alpha_t = alpha * target + (1 - alpha) * (1 - target)
        focal_weight = alpha_t * focal_weight

    loss = (focal_weight * ce).sum(dim=-1)

    if weight is not None:
        loss = loss * weight

    if avg_factor is not None:
        return loss.sum() / max(avg_factor, 1.0)
    return loss.mean()


class SigmoidFocalLoss(nn.Module):
    """Sigmoid Focal Loss module (RetinaNet-style).

    Args:
        alpha: Balancing factor between positive and negative.
        gamma: Focusing parameter.
        loss_weight: Scalar multiplier applied to the final loss.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, loss_weight: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.loss_weight = loss_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor = None,
        avg_factor: float = None,
    ) -> torch.Tensor:
        return self.loss_weight * sigmoid_focal_loss(
            pred, target, alpha=self.alpha, gamma=self.gamma,
            weight=weight, avg_factor=avg_factor,
        )
