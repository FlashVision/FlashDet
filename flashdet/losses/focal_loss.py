"""
Focal Loss variants matching the official FlashDet implementation exactly.

References:
  - Official QFL/DFL: flashdet/model/loss/gfocal_loss.py
  - "Generalized Focal Loss" (Li et al., 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def quality_focal_loss(pred, target, weight=None, beta=2.0, avg_factor=None):
    """Quality Focal Loss (QFL) functional.

    Negatives are supervised by a quality score of 0, with scale factor
    ``pred_sigmoid^beta``.  Positives are supervised by the IoU score, with
    scale factor ``|score - pred_sigmoid[pos, label]|^beta``.

    Args:
        pred (Tensor): [N, C] logits (before sigmoid).
        target (tuple): ``(labels [N,], scores [N,])`` where ``labels`` are
            category ids (background = num_classes) and ``scores`` are
            per-positive IoU quality values.
        weight (Tensor, optional): Per-sample weight [N].
        beta (float): Focusing exponent. Default: 2.0.
        avg_factor (float, optional): Normaliser for the loss sum.

    Returns:
        Tensor: Scalar loss.
    """
    assert len(target) == 2, (
        "target for QFL must be a tuple (labels, scores)"
    )
    label, score = target

    pred_sigmoid = pred.sigmoid()
    zerolabel = pred_sigmoid.new_zeros(pred.shape)

    # All samples initialised as background: BCE(pred, 0) * sigmoid^beta
    loss = F.binary_cross_entropy_with_logits(
        pred, zerolabel, reduction="none"
    ) * pred_sigmoid.pow(beta)

    # Override positive positions: BCE(pred[pos, label], score) * |score - sigmoid|^beta
    bg_class_ind = pred.size(1)
    pos = torch.nonzero(
        (label >= 0) & (label < bg_class_ind), as_tuple=False
    ).squeeze(1)

    if pos.numel() > 0:
        pos_label = label[pos].long()
        scale_factor = score[pos] - pred_sigmoid[pos, pos_label]
        loss[pos, pos_label] = F.binary_cross_entropy_with_logits(
            pred[pos, pos_label], score[pos], reduction="none"
        ) * scale_factor.abs().pow(beta)

    # Sum over class dimension → [N]
    loss = loss.sum(dim=1)

    if weight is not None:
        loss = loss * weight

    if avg_factor is None:
        return loss.mean()
    return loss.sum() / avg_factor


def distribution_focal_loss(pred, label, weight=None, avg_factor=None):
    """Distribution Focal Loss (DFL) functional.

    Soft two-hot cross-entropy over the discrete distribution bins.

    Args:
        pred (Tensor): [N, reg_max+1] distribution logits (before softmax).
        label (Tensor): [N] continuous target values (expected to be in
            ``[0, reg_max - 0.1]`` — the assigner is responsible for clamping).
        weight (Tensor, optional): Per-sample weight [N].
        avg_factor (float, optional): Normaliser for the loss sum.

    Returns:
        Tensor: Scalar loss.
    """
    dis_left = label.long()
    dis_right = dis_left + 1
    weight_left = dis_right.float() - label
    weight_right = label - dis_left.float()

    loss = (
        F.cross_entropy(pred, dis_left, reduction="none") * weight_left
        + F.cross_entropy(pred, dis_right, reduction="none") * weight_right
    )

    if weight is not None:
        loss = loss * weight

    if avg_factor is None:
        return loss.mean()
    return loss.sum() / avg_factor


class QualityFocalLoss(nn.Module):
    """Quality Focal Loss module.

    Args:
        beta (float): Focusing exponent. Default: 2.0.
        loss_weight (float): Scalar multiplier applied to the final loss.
    """

    def __init__(self, beta: float = 2.0, loss_weight: float = 1.0):
        super().__init__()
        self.beta = beta
        self.loss_weight = loss_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: tuple,
        weight: torch.Tensor = None,
        avg_factor: float = None,
    ) -> torch.Tensor:
        """
        Args:
            pred: [N, C] logits.
            target: ``(labels [N,], scores [N,])``.
            weight: Per-sample weight [N].
            avg_factor: Normaliser (number of positive samples).
        """
        return self.loss_weight * quality_focal_loss(
            pred, target, weight=weight, beta=self.beta, avg_factor=avg_factor
        )


class DistributionFocalLoss(nn.Module):
    """Distribution Focal Loss module.

    Args:
        loss_weight (float): Scalar multiplier applied to the final loss.
    """

    def __init__(self, loss_weight: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor = None,
        avg_factor: float = None,
    ) -> torch.Tensor:
        """
        Args:
            pred: [N, reg_max+1] distribution logits.
            target: [N] continuous target values.
            weight: Per-sample weight [N].
            avg_factor: Normaliser.
        """
        return self.loss_weight * distribution_focal_loss(
            pred, target, weight=weight, avg_factor=avg_factor
        )
