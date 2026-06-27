"""Bounding box utilities shared across training, inference, and loss modules."""

import math
from typing import List, Tuple

import torch
import torch.nn.functional as F


def make_anchor_grid(
    feat_sizes: List[Tuple[int, int]],
    strides: List[int],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build anchor center grid and stride tensor for all FPN levels.

    Returns:
        anchor_centers: [total_anchors, 2] (x, y) in pixel space.
        anchor_strides: [total_anchors, 1] stride per anchor.
    """
    centers_list = []
    strides_list = []
    for (h, w), stride in zip(feat_sizes, strides):
        shift_x = (torch.arange(w, device=device) + 0.5) * stride
        shift_y = (torch.arange(h, device=device) + 0.5) * stride
        yy, xx = torch.meshgrid(shift_y, shift_x, indexing="ij")
        centers_list.append(torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1))
        strides_list.append(torch.full((h * w, 1), stride, device=device, dtype=torch.float32))
    return torch.cat(centers_list, dim=0), torch.cat(strides_list, dim=0)


def decode_ltrb(
    anchor_centers: torch.Tensor,
    anchor_strides: torch.Tensor,
    reg_pred: torch.Tensor,
) -> torch.Tensor:
    """Decode LTRB distances to xyxy boxes.

    Args:
        anchor_centers: [N, 2]
        anchor_strides: [N, 1]
        reg_pred: [N, 4] raw regression output (softplus * stride).

    Returns:
        decoded_bboxes: [N, 4] in xyxy format.
    """
    ltrb = F.softplus(reg_pred, beta=1.0) * anchor_strides
    x1 = anchor_centers[:, 0] - ltrb[:, 0]
    y1 = anchor_centers[:, 1] - ltrb[:, 1]
    x2 = anchor_centers[:, 0] + ltrb[:, 2]
    y2 = anchor_centers[:, 1] + ltrb[:, 3]
    return torch.stack([x1, y1, x2, y2], dim=-1)


def bbox_iou_aligned(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """CIoU between aligned pairs [N,4] in xyxy format."""
    inter_x1 = torch.max(box1[:, 0], box2[:, 0])
    inter_y1 = torch.max(box1[:, 1], box2[:, 1])
    inter_x2 = torch.min(box1[:, 2], box2[:, 2])
    inter_y2 = torch.min(box1[:, 3], box2[:, 3])
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1 + area2 - inter + eps
    iou = inter / union

    enclose_x1 = torch.min(box1[:, 0], box2[:, 0])
    enclose_y1 = torch.min(box1[:, 1], box2[:, 1])
    enclose_x2 = torch.max(box1[:, 2], box2[:, 2])
    enclose_y2 = torch.max(box1[:, 3], box2[:, 3])
    c2 = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + eps
    cx1, cy1 = (box1[:, 0] + box1[:, 2]) / 2, (box1[:, 1] + box1[:, 3]) / 2
    cx2, cy2 = (box2[:, 0] + box2[:, 2]) / 2, (box2[:, 1] + box2[:, 3]) / 2
    rho2 = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

    w1, h1 = box1[:, 2] - box1[:, 0], box1[:, 3] - box1[:, 1]
    w2, h2 = box2[:, 2] - box2[:, 0], box2[:, 3] - box2[:, 1]
    v = (4 / (math.pi ** 2)) * (torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps))) ** 2
    with torch.no_grad():
        alpha_ciou = v / (1 - iou + v + eps)

    ciou = iou - rho2 / c2 - alpha_ciou * v
    return ciou


def decode_batch_nms_free(
    cls_logits: torch.Tensor,
    reg_preds: torch.Tensor,
    anchor_centers: torch.Tensor,
    anchor_strides: torch.Tensor,
    img_hw: Tuple[int, int],
    score_thr: float = 0.25,
    max_det: int = 300,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Pure NMS-free batch decode — score threshold + top-k only.

    The o2o head produces one prediction per anchor with 1:1 assignment,
    so duplicates do not exist — no NMS needed.

    Args:
        cls_logits: [B, N, num_classes] raw logits from o2o head.
        reg_preds:  [B, N, 4] raw LTRB regression from o2o head.
        anchor_centers: [N, 2] precomputed anchor grid centers.
        anchor_strides: [N, 1] stride per anchor.
        img_hw: (H, W) of the input image tensor.
        score_thr: Minimum confidence to keep.
        max_det: Maximum detections per image.

    Returns:
        List of (det_bboxes [M, 5], det_labels [M]) per image.
    """
    H, W = img_hw
    B = cls_logits.shape[0]
    device = cls_logits.device
    results: List[Tuple[torch.Tensor, torch.Tensor]] = []

    for b in range(B):
        scores = cls_logits[b].sigmoid()
        max_scores, labels = scores.max(dim=1)

        keep = max_scores > score_thr
        if keep.sum() == 0:
            results.append((
                torch.zeros((0, 5), device=device),
                torch.zeros((0,), dtype=torch.long, device=device),
            ))
            continue

        scores_k = max_scores[keep]
        labels_k = labels[keep]
        reg_k = reg_preds[b][keep]
        centers_k = anchor_centers[keep]
        strides_k = anchor_strides[keep]

        boxes_k = decode_ltrb(centers_k, strides_k, reg_k)
        boxes_k[:, 0].clamp_(min=0, max=W)
        boxes_k[:, 1].clamp_(min=0, max=H)
        boxes_k[:, 2].clamp_(min=0, max=W)
        boxes_k[:, 3].clamp_(min=0, max=H)

        if scores_k.shape[0] > max_det:
            topk_idx = scores_k.topk(max_det).indices
            scores_k = scores_k[topk_idx]
            boxes_k = boxes_k[topk_idx]
            labels_k = labels_k[topk_idx]

        det_bboxes = torch.cat([boxes_k, scores_k[:, None]], dim=1)
        results.append((det_bboxes, labels_k))

    return results
