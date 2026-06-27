"""
YOLOv9 — Programmable Gradient Information (PGI) with GELAN architecture.

Independent implementation based on:
    Wang et al., "YOLOv9: Learning What You Want to Learn Using
    Programmable Gradient Information", arXiv:2402.13616, 2024.

This is a clean-room implementation based on the paper's architectural
description. No code was copied from GPL/AGPL-licensed repositories.

License: MIT (same as FlashDet)
"""

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from flashdet.registry import DETECTORS
from flashdet.models.backbone.yolov9_backbone import YOLOv9Backbone
from flashdet.models.neck.yolov9_neck import YOLOv9Neck
from flashdet.models.head.yolo_head import YOLODetectionHead, PGIAuxBranch

logger = logging.getLogger(__name__)


@DETECTORS.register("YOLOv9")
class YOLOv9(nn.Module):
    """YOLOv9 object detector with GELAN backbone and PGI.

    Args:
        num_classes: Number of detection classes.
        width_mult: Channel width multiplier.
        depth_mult: Block depth multiplier.
        in_channels: Input image channels.
        reg_max: DFL regression max value.
        use_pgi: Enable PGI auxiliary branch during training.
    """

    def __init__(
        self,
        num_classes: int = 80,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        in_channels: int = 3,
        reg_max: int = 16,
        use_pgi: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.use_pgi = use_pgi

        self.backbone = YOLOv9Backbone(in_channels, width_mult, depth_mult)

        neck_out = self.backbone.out_channels[1]  # channels[2]
        self.neck = YOLOv9Neck(self.backbone.out_channels, neck_out)

        self.heads = nn.ModuleList([
            YOLODetectionHead(num_classes, neck_out, reg_max=reg_max)
            for _ in range(3)
        ])

        if use_pgi:
            self.pgi_aux = PGIAuxBranch(self.backbone.out_channels, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, gt_meta: Optional[Dict] = None, compute_loss: bool = False, **kwargs) -> Dict:
        backbone_feats = self.backbone(x)
        neck_feats = self.neck(backbone_feats)

        preds = [head(f) for head, f in zip(self.heads, neck_feats)]

        result: Dict = {"preds": preds}

        if self.training and self.use_pgi:
            result["aux_preds"] = self.pgi_aux(backbone_feats)

        if (self.training or compute_loss) and gt_meta is not None:
            from flashdet.losses.yolo_loss import compute_yolo_loss
            gt_meta["input_h"] = x.shape[2]
            loss, loss_states = compute_yolo_loss(
                preds, gt_meta, self.num_classes, reg_max=self.reg_max,
            )
            result["loss"] = loss
            result["loss_states"] = loss_states

        return result

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        img_metas=None,
        score_thr: float = 0.05,
        nms_thr: float = 0.6,
    ) -> list:
        """Run inference and return ``[(dets, labels), ...]`` per image."""
        self.eval()
        out = self.forward(x)
        from flashdet.engine.inference.postprocess import decode_yolo_predictions
        return decode_yolo_predictions(
            out["preds"], self.num_classes, x.shape[2:],
            reg_max=self.reg_max, score_thr=score_thr, nms_thr=nms_thr,
        )

    def get_model_info(self) -> Dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "name": "YOLOv9",
            "num_classes": self.num_classes,
            "total_params": total,
            "trainable_params": trainable,
            "params_mb": total * 4 / (1024 ** 2),
        }
