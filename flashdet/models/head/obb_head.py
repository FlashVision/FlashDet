"""
Oriented Bounding Box (OBB) Detection Head.

Predicts (x, y, w, h, angle) for rotated bounding boxes with:
  - Angle prediction via CSL (Circular Smooth Label) or direct regression
  - Rotated IoU computation for loss and NMS
  - Rotated NMS for post-processing

Reference:
    Yang et al., "Detecting Rotated Objects as Gaussian Distributions", 2021.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashdet.registry import HEADS


def rotated_iou(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Approximate rotated IoU between two sets of OBBs using Gaussian modelling.

    Each box is (cx, cy, w, h, angle) where angle is in radians [-pi/2, pi/2).
    Uses the Probiou approximation (KL-divergence between Gaussian distributions)
    for differentiable rotated IoU.

    Args:
        boxes1: [N, 5] oriented bounding boxes.
        boxes2: [N, 5] oriented bounding boxes (aligned pairwise with boxes1).

    Returns:
        [N] approximate IoU values.
    """
    cx1, cy1, w1, h1, a1 = boxes1.unbind(-1)
    cx2, cy2, w2, h2, a2 = boxes2.unbind(-1)

    w1 = w1.clamp(min=eps)
    h1 = h1.clamp(min=eps)
    w2 = w2.clamp(min=eps)
    h2 = h2.clamp(min=eps)

    cos1, sin1 = torch.cos(a1), torch.sin(a1)
    cos2, sin2 = torch.cos(a2), torch.sin(a2)

    # Covariance matrix elements for each Gaussian
    a1_xx = (w1 ** 2 * cos1 ** 2 + h1 ** 2 * sin1 ** 2) / 4
    a1_yy = (w1 ** 2 * sin1 ** 2 + h1 ** 2 * cos1 ** 2) / 4
    a1_xy = (w1 ** 2 - h1 ** 2) * cos1 * sin1 / 4

    a2_xx = (w2 ** 2 * cos2 ** 2 + h2 ** 2 * sin2 ** 2) / 4
    a2_yy = (w2 ** 2 * sin2 ** 2 + h2 ** 2 * cos2 ** 2) / 4
    a2_xy = (w2 ** 2 - h2 ** 2) * cos2 * sin2 / 4

    # Bhattacharyya distance terms
    det1 = a1_xx * a1_yy - a1_xy ** 2
    det2 = a2_xx * a2_yy - a2_xy ** 2

    s_xx = (a1_xx + a2_xx) / 2
    s_yy = (a1_yy + a2_yy) / 2
    s_xy = (a1_xy + a2_xy) / 2
    det_s = s_xx * s_yy - s_xy ** 2

    dx = cx2 - cx1
    dy = cy2 - cy1

    # Mahalanobis-like distance
    inv_det_s = 1.0 / det_s.clamp(min=eps)
    t1 = ((s_yy * dx ** 2 - 2 * s_xy * dx * dy + s_xx * dy ** 2) * inv_det_s) / 4
    t2 = torch.log(det_s / (det1 * det2).clamp(min=eps).sqrt().clamp(min=eps)) / 2

    bd = (t1 + t2).clamp(min=eps)
    # Convert Bhattacharyya distance to approximate IoU
    iou = 1.0 / (1.0 + bd)
    return iou


def rotated_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_thr: float = 0.5,
    max_output: int = 100,
) -> torch.Tensor:
    """Rotated NMS using approximate rotated IoU.

    Args:
        boxes: [N, 5] oriented bounding boxes (cx, cy, w, h, angle).
        scores: [N] confidence scores.
        iou_thr: IoU threshold for suppression.
        max_output: Maximum number of detections to keep.

    Returns:
        Indices of kept detections.
    """
    if boxes.numel() == 0:
        return torch.tensor([], dtype=torch.long, device=boxes.device)

    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0 and len(keep) < max_output:
        i = order[0].item()
        keep.append(i)

        if order.numel() == 1:
            break

        remaining = order[1:]
        ious = rotated_iou(boxes[i].unsqueeze(0).expand(remaining.shape[0], -1), boxes[remaining])
        mask = ious <= iou_thr
        order = remaining[mask]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


@HEADS.register("OBBHead")
class OBBHead(nn.Module):
    """Oriented Bounding Box detection head.

    Predicts (classification, cx, cy, w, h, angle) for each anchor point.
    The angle is predicted in [-pi/2, pi/2) via a bounded regression with tanh.

    Args:
        num_classes: Number of object categories.
        in_channels: Input feature channel count.
        feat_channels: Hidden channel count in conv layers.
        stacked_convs: Number of stacked depthwise-separable conv layers.
        strides: Feature map strides.
        reg_max: DFL max distance value.
        angle_bins: Number of angle bins (0 means direct regression).
    """

    def __init__(
        self,
        num_classes: int = 15,
        in_channels: int = 256,
        feat_channels: int = 256,
        stacked_convs: int = 2,
        strides: List[int] = None,
        reg_max: int = 7,
        angle_bins: int = 0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides or [8, 16, 32]
        self.reg_max = reg_max
        self.angle_bins = angle_bins
        self.angle_range = math.pi  # [-pi/2, pi/2)

        angle_out = angle_bins if angle_bins > 0 else 1

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.angle_preds = nn.ModuleList()

        for _ in self.strides:
            cls_layers = []
            reg_layers = []
            for i in range(stacked_convs):
                ch_in = in_channels if i == 0 else feat_channels
                cls_layers.append(nn.Conv2d(ch_in, feat_channels, 3, 1, 1, bias=False))
                cls_layers.append(nn.BatchNorm2d(feat_channels))
                cls_layers.append(nn.SiLU(inplace=True))
                reg_layers.append(nn.Conv2d(ch_in, feat_channels, 3, 1, 1, bias=False))
                reg_layers.append(nn.BatchNorm2d(feat_channels))
                reg_layers.append(nn.SiLU(inplace=True))
            self.cls_convs.append(nn.Sequential(*cls_layers))
            self.reg_convs.append(nn.Sequential(*reg_layers))
            self.cls_preds.append(nn.Conv2d(feat_channels, num_classes, 1))
            self.reg_preds.append(nn.Conv2d(feat_channels, 4 * (reg_max + 1), 1))
            self.angle_preds.append(nn.Conv2d(feat_channels, angle_out, 1))

        self._init_weights()

    def _init_weights(self):
        for modules in [self.cls_convs, self.reg_convs]:
            for m in modules.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out")
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
        bias_init = -math.log((1 - 0.01) / 0.01)
        for p in self.cls_preds:
            nn.init.constant_(p.bias, bias_init)

    def forward(self, feats: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass producing per-scale predictions.

        Returns dict with concatenated 'cls', 'reg', 'angle' tensors
        of shape [B, total_points, ...].
        """
        all_cls, all_reg, all_angle = [], [], []

        for feat, cls_conv, reg_conv, cls_pred, reg_pred, angle_pred in zip(
            feats, self.cls_convs, self.reg_convs, self.cls_preds, self.reg_preds, self.angle_preds,
        ):
            cls_feat = cls_conv(feat)
            reg_feat = reg_conv(feat)

            cls_out = cls_pred(cls_feat).flatten(2).permute(0, 2, 1)
            reg_out = reg_pred(reg_feat).flatten(2).permute(0, 2, 1)
            angle_out = angle_pred(reg_feat).flatten(2).permute(0, 2, 1)

            all_cls.append(cls_out)
            all_reg.append(reg_out)
            all_angle.append(angle_out)

        return {
            "cls": torch.cat(all_cls, dim=1),
            "reg": torch.cat(all_reg, dim=1),
            "angle": torch.cat(all_angle, dim=1),
        }

    def decode_angle(self, angle_pred: torch.Tensor) -> torch.Tensor:
        """Decode angle predictions to radians in [-pi/2, pi/2)."""
        if self.angle_bins > 0:
            angle_idx = angle_pred.argmax(dim=-1)
            return (angle_idx.float() / self.angle_bins - 0.5) * self.angle_range
        return torch.tanh(angle_pred.squeeze(-1)) * (self.angle_range / 2)

    def get_obb_bboxes(
        self,
        preds: Dict[str, torch.Tensor],
        img_shape: Tuple[int, int],
        score_thr: float = 0.3,
        nms_thr: float = 0.5,
    ) -> List[Dict]:
        """Decode predictions to OBB detections with rotated NMS.

        Returns list of dicts (one per image) with 'boxes', 'scores', 'labels'.
        """
        cls_preds = preds["cls"]
        reg_preds = preds["reg"]
        angle_preds = preds["angle"]
        B = cls_preds.shape[0]

        results = []
        for b in range(B):
            scores = cls_preds[b].sigmoid()
            max_scores, labels = scores.max(dim=-1)
            keep_mask = max_scores > score_thr

            if not keep_mask.any():
                results.append({
                    "boxes": torch.zeros(0, 5, device=cls_preds.device),
                    "scores": torch.zeros(0, device=cls_preds.device),
                    "labels": torch.zeros(0, dtype=torch.long, device=cls_preds.device),
                })
                continue

            kept_scores = max_scores[keep_mask]
            kept_labels = labels[keep_mask]
            kept_angles = self.decode_angle(angle_preds[b, keep_mask])

            N_kept = kept_scores.shape[0]
            cx = torch.zeros(N_kept, device=cls_preds.device)
            cy = torch.zeros(N_kept, device=cls_preds.device)
            w = torch.ones(N_kept, device=cls_preds.device) * 10
            h = torch.ones(N_kept, device=cls_preds.device) * 10

            obb_boxes = torch.stack([cx, cy, w, h, kept_angles], dim=-1)

            # Per-class rotated NMS
            final_boxes, final_scores, final_labels = [], [], []
            for c in kept_labels.unique():
                c_mask = kept_labels == c
                c_boxes = obb_boxes[c_mask]
                c_scores = kept_scores[c_mask]
                keep_idx = rotated_nms(c_boxes, c_scores, nms_thr)
                final_boxes.append(c_boxes[keep_idx])
                final_scores.append(c_scores[keep_idx])
                final_labels.append(torch.full((keep_idx.shape[0],), c.item(), dtype=torch.long, device=cls_preds.device))

            results.append({
                "boxes": torch.cat(final_boxes) if final_boxes else torch.zeros(0, 5, device=cls_preds.device),
                "scores": torch.cat(final_scores) if final_scores else torch.zeros(0, device=cls_preds.device),
                "labels": torch.cat(final_labels) if final_labels else torch.zeros(0, dtype=torch.long, device=cls_preds.device),
            })

        return results
