"""
YOLOX — Anchor-Free YOLO with Decoupled Head and SimOTA.

Independent implementation based on:
    Ge et al., "YOLOX: Exceeding YOLO Series in 2021",
    arXiv:2107.08430, 2021.
    Original code: https://github.com/Megvii-BaseDetection/YOLOX (Apache-2.0)

This is a clean-room implementation based on the paper's architectural
description. The original YOLOX code is Apache-2.0 licensed.

License: MIT (same as FlashDet)
"""

import logging
from typing import Dict, List

import torch
import torch.nn as nn

from flashdet.models.backbone.yolox_backbone import YOLOXBackbone
from flashdet.models.neck.yolox_neck import YOLOXNeck
from flashdet.models.head.yolox_head import YOLOXHead
from flashdet.registry import DETECTORS

logger = logging.getLogger(__name__)


@DETECTORS.register("YOLOX")
class YOLOX(nn.Module):
    """YOLOX anchor-free detector with decoupled head.

    Architecture: CSPDarknet(Focus) → PAFPN(CSP) → Decoupled Head (cls+reg+obj)

    Args:
        num_classes: Number of object classes.
        width_mult: Channel multiplier (0.33=nano, 0.50=tiny, 0.75=s, 1.0=m).
        depth_mult: Depth multiplier (0.33=nano/tiny/s, 0.67=m, 1.0=l).
    """

    def __init__(self, num_classes: int = 80, width_mult: float = 1.0,
                 depth_mult: float = 1.0, **kwargs):
        super().__init__()
        self.num_classes = num_classes

        self.backbone = YOLOXBackbone(width_mult, depth_mult)
        self.neck = YOLOXNeck(self.backbone.out_channels, depth_mult)
        self.head = YOLOXHead(num_classes, self.neck.out_channels)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        features = self.backbone(x)
        features = self.neck(features)
        outputs = self.head(features)
        return outputs

    @torch.no_grad()
    def predict(self, x: torch.Tensor, img_metas=None,
                score_thr: float = 0.05, nms_thr: float = 0.6) -> list:
        """Run inference with NMS. Returns [(dets, labels), ...] per image."""
        self.eval()
        out = self.forward(x)
        return self._decode_yolox(out, x.shape[2:], score_thr, nms_thr)

    def _decode_yolox(self, out: Dict, input_hw, score_thr: float, nms_thr: float) -> list:
        """Decode YOLOX multi-scale outputs into detections."""
        import torchvision
        cls_preds = out["cls_preds"]
        reg_preds = out["reg_preds"]
        obj_preds = out["obj_preds"]

        h, w = input_hw
        strides = [h // cls_preds[i].shape[2] for i in range(len(cls_preds))]

        all_boxes, all_scores, all_labels = [], [], []
        for i in range(len(cls_preds)):
            stride = strides[i]
            fh, fw = cls_preds[i].shape[2], cls_preds[i].shape[3]

            # Build grid
            yv, xv = torch.meshgrid(torch.arange(fh), torch.arange(fw), indexing="ij")
            grid = torch.stack([xv, yv], dim=2).reshape(1, -1, 2).to(cls_preds[i].device).float()

            # Decode
            cls_out = cls_preds[i].sigmoid().flatten(2).permute(0, 2, 1)  # B, HW, nc
            obj_out = obj_preds[i].sigmoid().flatten(2).permute(0, 2, 1)  # B, HW, 1
            reg_out = reg_preds[i].flatten(2).permute(0, 2, 1)            # B, HW, 4

            # Box decode: center + offset, then convert to xyxy
            xy = (grid + 0.5 + reg_out[..., :2]) * stride
            wh = reg_out[..., 2:4].exp() * stride
            x1y1 = xy - wh / 2
            x2y2 = xy + wh / 2
            boxes = torch.cat([x1y1, x2y2], dim=-1)  # B, HW, 4

            scores = (cls_out * obj_out)  # B, HW, nc
            max_scores, labels = scores.max(dim=-1)  # B, HW

            # Filter by score
            mask = max_scores[0] > score_thr
            if mask.sum() == 0:
                continue
            all_boxes.append(boxes[0][mask])
            all_scores.append(max_scores[0][mask])
            all_labels.append(labels[0][mask])

        if not all_boxes:
            empty = torch.zeros(0, 5, device=cls_preds[0].device)
            return [(empty, torch.zeros(0, dtype=torch.long, device=cls_preds[0].device))]

        boxes_cat = torch.cat(all_boxes, dim=0)
        scores_cat = torch.cat(all_scores, dim=0)
        labels_cat = torch.cat(all_labels, dim=0)

        keep = torchvision.ops.nms(boxes_cat, scores_cat, nms_thr)
        boxes_cat = boxes_cat[keep]
        scores_cat = scores_cat[keep]
        labels_cat = labels_cat[keep]

        dets = torch.cat([boxes_cat, scores_cat.unsqueeze(1)], dim=1)
        return [(dets, labels_cat)]

    def info(self) -> str:
        n_params = sum(p.numel() for p in self.parameters()) / 1e6
        return f"YOLOX (params={n_params:.2f}M, nc={self.num_classes})"
