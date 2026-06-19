"""
Chunked loss computation inspired by torchtune's CEWithChunkedOutputLoss.

For object detection, the "chunked" approach splits the spatial predictions
into smaller chunks before computing the focal/CE losses.  This avoids
materializing the full (N_anchors × num_classes) tensor at once, reducing
peak memory during the loss backward pass.

This is most beneficial when:
  - Training with many anchors (high-resolution inputs / small strides)
  - Using large batch sizes
  - Running on memory-constrained GPUs
"""

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def chunked_quality_focal_loss(
    pred: torch.Tensor,
    target: tuple,
    beta: float = 2.0,
    weight: torch.Tensor = None,
    avg_factor: float = None,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Compute QFL in chunks along the anchor dimension.

    Same semantics as ``quality_focal_loss`` but processes ``chunk_size``
    anchors at a time to limit peak memory.
    """
    label, score = target
    N = pred.size(0)

    if N <= chunk_size:
        from .focal_loss import quality_focal_loss
        return quality_focal_loss(pred, target, weight=weight, beta=beta, avg_factor=avg_factor)

    total_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk_pred = pred[start:end]
        chunk_label = label[start:end]
        chunk_score = score[start:end]
        chunk_weight = weight[start:end] if weight is not None else None

        pred_sigmoid = chunk_pred.sigmoid()
        zerolabel = pred_sigmoid.new_zeros(chunk_pred.shape)
        loss = F.binary_cross_entropy_with_logits(
            chunk_pred, zerolabel, reduction="none"
        ) * pred_sigmoid.pow(beta)

        bg_class_ind = chunk_pred.size(1)
        pos = torch.nonzero(
            (chunk_label >= 0) & (chunk_label < bg_class_ind), as_tuple=False
        ).squeeze(1)

        if pos.numel() > 0:
            pos_label = chunk_label[pos].long()
            scale_factor = chunk_score[pos] - pred_sigmoid[pos, pos_label]
            loss[pos, pos_label] = F.binary_cross_entropy_with_logits(
                chunk_pred[pos, pos_label], chunk_score[pos], reduction="none"
            ) * scale_factor.abs().pow(beta)

        loss = loss.sum(dim=1)
        if chunk_weight is not None:
            loss = loss * chunk_weight
        total_loss = total_loss + loss.sum()

    if avg_factor is None:
        return total_loss / N
    return total_loss / avg_factor


def chunked_distribution_focal_loss(
    pred: torch.Tensor,
    label: torch.Tensor,
    weight: torch.Tensor = None,
    avg_factor: float = None,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Compute DFL in chunks along the sample dimension."""
    N = pred.size(0)

    if N <= chunk_size:
        from .focal_loss import distribution_focal_loss
        return distribution_focal_loss(pred, label, weight=weight, avg_factor=avg_factor)

    total_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk_pred = pred[start:end]
        chunk_label = label[start:end]
        chunk_weight = weight[start:end] if weight is not None else None

        dis_left = chunk_label.long()
        dis_right = dis_left + 1
        weight_left = dis_right.float() - chunk_label
        weight_right = chunk_label - dis_left.float()

        loss = (
            F.cross_entropy(chunk_pred, dis_left, reduction="none") * weight_left
            + F.cross_entropy(chunk_pred, dis_right, reduction="none") * weight_right
        )

        if chunk_weight is not None:
            loss = loss * chunk_weight
        total_loss = total_loss + loss.sum()

    if avg_factor is None:
        return total_loss / N
    return total_loss / avg_factor
