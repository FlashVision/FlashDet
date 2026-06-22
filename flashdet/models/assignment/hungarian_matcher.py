"""Bipartite (Hungarian) matching for DETR-family detectors."""

from typing import List, Tuple

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment


class HungarianMatcher(nn.Module):
    """Bipartite matching between predictions and ground-truth using the
    Hungarian algorithm, as described in the DETR paper."""

    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 5.0, cost_giou: float = 2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(
        self,
        pred_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        gt_labels_list: List[torch.Tensor],
        gt_boxes_list: List[torch.Tensor],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        B, N, C = pred_logits.shape
        indices = []

        for b in range(B):
            if gt_labels_list[b].numel() == 0:
                indices.append((
                    torch.tensor([], dtype=torch.long, device=pred_logits.device),
                    torch.tensor([], dtype=torch.long, device=pred_logits.device),
                ))
                continue

            prob = pred_logits[b].softmax(-1)
            cost_cls = -prob[:, gt_labels_list[b]]

            cost_box = torch.cdist(pred_boxes[b], gt_boxes_list[b], p=1)

            cost_giou = -generalized_iou(
                cxcywh_to_xyxy(pred_boxes[b]),
                cxcywh_to_xyxy(gt_boxes_list[b]),
            )

            cost = self.cost_class * cost_cls + self.cost_bbox * cost_box + self.cost_giou * cost_giou
            row, col = linear_sum_assignment(cost.cpu().numpy())
            indices.append((
                torch.as_tensor(row, dtype=torch.long, device=pred_logits.device),
                torch.as_tensor(col, dtype=torch.long, device=pred_logits.device),
            ))
        return indices


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2) format."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def generalized_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Pairwise GIoU between two sets of boxes (xyxy format)."""
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union.clamp(min=1e-6)

    enc_lt = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    enc_rb = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    enc_wh = (enc_rb - enc_lt).clamp(min=0)
    enc_area = enc_wh[..., 0] * enc_wh[..., 1]

    return iou - (enc_area - union) / enc_area.clamp(min=1e-6)
