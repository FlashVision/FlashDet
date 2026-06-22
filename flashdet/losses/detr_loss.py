"""DETR set-based loss with Hungarian matching."""

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from flashdet.models.assignment.hungarian_matcher import (
    HungarianMatcher,
    cxcywh_to_xyxy,
    generalized_iou,
)


def compute_detr_loss(
    pred_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    gt_meta: Dict,
    num_classes: int,
    matcher: HungarianMatcher,
    num_queries: int,
    img_shape: Tuple[int, int],
    loss_ce_weight: float = 1.0,
    loss_bbox_weight: float = 5.0,
    loss_giou_weight: float = 2.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute DETR set-based loss.

    Args:
        pred_logits: [B, num_queries, num_classes+1].
        pred_boxes: [B, num_queries, 4] in cxcywh normalized format.
        gt_meta: Dict with gt_bboxes (xyxy pixel) and gt_labels.
        num_classes: Number of object classes (excluding background).
        matcher: HungarianMatcher instance.
        num_queries: Number of object queries.
        img_shape: (H, W) of the input image.
        loss_ce_weight: Classification loss weight.
        loss_bbox_weight: L1 bbox regression loss weight.
        loss_giou_weight: GIoU loss weight.

    Returns:
        (total_loss, loss_states dict).
    """
    device = pred_logits.device
    B = pred_logits.shape[0]

    gt_labels_list = []
    gt_boxes_list = []
    for i in range(B):
        gt_b = torch.as_tensor(gt_meta["gt_bboxes"][i], dtype=torch.float32, device=device)
        gt_l = torch.as_tensor(gt_meta["gt_labels"][i], dtype=torch.long, device=device)
        if gt_b.numel() > 0:
            gt_boxes_list.append(_xyxy_to_cxcywh_norm(gt_b, img_shape))
        else:
            gt_boxes_list.append(gt_b.reshape(0, 4))
        gt_labels_list.append(gt_l)

    indices = matcher(pred_logits, pred_boxes, gt_labels_list, gt_boxes_list)

    target_classes = torch.full(
        (B, num_queries), num_classes, dtype=torch.long, device=device,
    )
    for b, (src_idx, tgt_idx) in enumerate(indices):
        if src_idx.numel() > 0:
            target_classes[b, src_idx] = gt_labels_list[b][tgt_idx]

    eos_weight = torch.ones(num_classes + 1, device=device)
    eos_weight[-1] = 0.1
    loss_ce = F.cross_entropy(
        pred_logits.flatten(0, 1), target_classes.flatten(), weight=eos_weight,
    )

    src_boxes_all = []
    tgt_boxes_all = []
    for b, (src_idx, tgt_idx) in enumerate(indices):
        if src_idx.numel() > 0:
            src_boxes_all.append(pred_boxes[b, src_idx])
            tgt_boxes_all.append(gt_boxes_list[b][tgt_idx])

    if src_boxes_all:
        src_cat = torch.cat(src_boxes_all)
        tgt_cat = torch.cat(tgt_boxes_all)
        loss_bbox = F.l1_loss(src_cat, tgt_cat, reduction="mean")
        loss_giou = (1 - torch.diag(
            generalized_iou(cxcywh_to_xyxy(src_cat), cxcywh_to_xyxy(tgt_cat))
        )).mean()
    else:
        loss_bbox = pred_boxes.sum() * 0
        loss_giou = pred_boxes.sum() * 0

    total = (
        loss_ce_weight * loss_ce
        + loss_bbox_weight * loss_bbox
        + loss_giou_weight * loss_giou
    )

    return total, {
        "loss_ce": loss_ce.detach(),
        "loss_bbox": loss_bbox.detach(),
        "loss_giou": loss_giou.detach(),
    }


def _xyxy_to_cxcywh_norm(boxes: torch.Tensor, img_shape: Tuple[int, int]) -> torch.Tensor:
    h, w = img_shape
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2 / w
    cy = (y1 + y2) / 2 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return torch.stack([cx, cy, bw, bh], dim=-1)
