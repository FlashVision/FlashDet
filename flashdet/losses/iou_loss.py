"""
IoU-based bounding box regression losses matching the official FlashDet
implementation exactly.

References:
  - Official GIoU: flashdet/model/loss/iou_loss.py + bbox_overlaps
"""

import torch
import torch.nn as nn

from ..utils.box_utils import bbox_overlaps


def giou_loss(pred, target, weight=None, eps=1e-7, avg_factor=None):
    """GIoU Loss functional.

    Args:
        pred (Tensor): Predicted boxes [N, 4] in ``(x1, y1, x2, y2)`` format.
        target (Tensor): Target boxes [N, 4] in ``(x1, y1, x2, y2)`` format.
        weight (Tensor, optional): Per-sample weight [N].
        eps (float): Epsilon for numerical stability. Default: 1e-7.
        avg_factor (float, optional): Normaliser for the loss sum.

    Returns:
        Tensor: Scalar loss.
    """
    gious = bbox_overlaps(pred, target, mode="giou", is_aligned=True, eps=eps)
    loss = 1 - gious

    if weight is not None:
        loss = loss * weight

    if avg_factor is None:
        return loss.mean()
    return loss.sum() / avg_factor


class GIoULoss(nn.Module):
    """Generalized IoU Loss module.

    Args:
        eps (float): Epsilon for numerical stability. Default: 1e-7.
        loss_weight (float): Scalar multiplier applied to the final loss.
    """

    def __init__(self, eps: float = 1e-7, loss_weight: float = 1.0):
        super().__init__()
        self.eps = eps
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
            pred: [N, 4] predicted boxes.
            target: [N, 4] target boxes.
            weight: Per-sample weight [N].
            avg_factor: Normaliser.
        """
        return self.loss_weight * giou_loss(
            pred, target, weight=weight, eps=self.eps, avg_factor=avg_factor
        )


def iou_loss(pred, target, weight=None, eps=1e-6, avg_factor=None):
    """IoU Loss functional using the ``-log(IoU)`` formulation.

    Args:
        pred (Tensor): Predicted boxes [N, 4].
        target (Tensor): Target boxes [N, 4].
        weight (Tensor, optional): Per-sample weight [N].
        eps (float): Epsilon for numerical stability.
        avg_factor (float, optional): Normaliser.

    Returns:
        Tensor: Scalar loss.
    """
    ious = bbox_overlaps(pred, target, mode="iou", is_aligned=True, eps=eps)
    loss = -ious.clamp(min=eps).log()

    if weight is not None:
        loss = loss * weight

    if avg_factor is None:
        return loss.mean()
    return loss.sum() / avg_factor


class IoULoss(nn.Module):
    """IoU Loss module (``-log(IoU)`` formulation).

    Args:
        eps (float): Epsilon for numerical stability.
        loss_weight (float): Scalar multiplier applied to the final loss.
    """

    def __init__(self, eps: float = 1e-6, loss_weight: float = 1.0):
        super().__init__()
        self.eps = eps
        self.loss_weight = loss_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor = None,
        avg_factor: float = None,
    ) -> torch.Tensor:
        return self.loss_weight * iou_loss(
            pred, target, weight=weight, eps=self.eps, avg_factor=avg_factor
        )
