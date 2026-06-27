"""
YOLOv8 — Anchor-Free Object Detection with C2f and Decoupled Head.

Independent implementation based on:
    Jocher et al., "YOLOv8", Ultralytics, 2023.
    Architecture publicly documented: C2f backbone, PANet neck, decoupled DFL head.

This is a clean-room implementation based on the publicly documented
architectural design. No code was copied from AGPL-licensed repositories.

License: MIT (same as FlashDet)
"""

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from flashdet.models.backbone.yolov8_backbone import YOLOv8Backbone
from flashdet.models.neck.yolov8_neck import YOLOv8Neck
from flashdet.models.head.yolo_head import YOLODetectionHead
from flashdet.registry import DETECTORS

logger = logging.getLogger(__name__)


@DETECTORS.register("YOLOv8")
class YOLOv8(nn.Module):
    """YOLOv8 anchor-free detector with C2f blocks and DFL head.

    Architecture: CSPDarknet(C2f) → PANet(C2f) → Decoupled Head (DFL + BCE)

    Args:
        num_classes: Number of object classes.
        width_mult: Channel multiplier (0.25=n, 0.5=s, 0.75=m, 1.0=l).
        depth_mult: Depth multiplier (0.33=n, 0.33=s, 0.67=m, 1.0=l).
        reg_max: DFL regression range.
    """

    def __init__(self, num_classes: int = 80, width_mult: float = 1.0,
                 depth_mult: float = 1.0, reg_max: int = 16, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max

        self.backbone = YOLOv8Backbone(width_mult, depth_mult)
        self.neck = YOLOv8Neck(self.backbone.out_channels, depth_mult)

        # Per-scale detection heads
        self.heads = nn.ModuleList([
            YOLODetectionHead(num_classes=num_classes, in_channels=ch, reg_max=reg_max)
            for ch in self.neck.out_channels
        ])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)
        features = self.neck(features)
        preds = [self.heads[i](feat) for i, feat in enumerate(features)]
        return {"preds": preds}

    @torch.no_grad()
    def predict(self, x: torch.Tensor, img_metas=None,
                score_thr: float = 0.05, nms_thr: float = 0.6) -> list:
        """Run inference with NMS. Returns [(dets, labels), ...] per image."""
        self.eval()
        out = self.forward(x)
        from flashdet.engine.inference.postprocess import decode_yolo_predictions
        return decode_yolo_predictions(
            out["preds"], self.num_classes, x.shape[2:],
            reg_max=self.reg_max, score_thr=score_thr, nms_thr=nms_thr,
        )

    def info(self) -> str:
        n_params = sum(p.numel() for p in self.parameters()) / 1e6
        return f"YOLOv8 (params={n_params:.2f}M, nc={self.num_classes})"
