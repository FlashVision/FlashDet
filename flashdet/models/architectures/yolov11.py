"""
YOLOv11 — Next-generation YOLO with C3k2 blocks and enhanced SPPF.

Independent implementation based on the publicly described architecture:
    Jocher et al., YOLO11, Ultralytics, 2024.

This is a clean-room implementation based on the publicly documented
architectural design (C3k2, C2PSA, SPPF blocks). No code was copied
from AGPL-licensed repositories.

License: MIT (same as FlashDet)
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn

from flashdet.registry import DETECTORS
from flashdet.models.backbone.yolov11_backbone import YOLOv11Backbone
from flashdet.models.neck.yolov11_neck import YOLOv11Neck
from flashdet.models.head.yolo_head import YOLODetectionHead

logger = logging.getLogger(__name__)


@DETECTORS.register("YOLOv11")
class YOLOv11(nn.Module):
    """YOLOv11 object detector with C3k2 blocks and SPPF.

    Args:
        num_classes: Number of detection classes.
        width_mult: Channel width multiplier (0.25=n, 0.5=s, 1.0=m, 1.25=l, 1.5=x).
        depth_mult: Block depth multiplier.
        reg_max: DFL regression max value.
        use_c2psa: Use C2PSA attention in the last backbone stage.
    """

    def __init__(
        self,
        num_classes: int = 80,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        reg_max: int = 16,
        use_c2psa: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max

        self.backbone = YOLOv11Backbone(width_mult, depth_mult, use_c2psa)

        neck_out = self.backbone.out_channels[1]  # channels[2]
        self.neck = YOLOv11Neck(self.backbone.out_channels, neck_out)

        self.heads = nn.ModuleList([
            YOLODetectionHead(num_classes, neck_out, reg_max) for _ in range(3)
        ])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        for head in self.heads:
            nn.init.constant_(head.cls_pred.bias, -4.595)

    def forward(self, x: torch.Tensor, gt_meta: Optional[Dict] = None, compute_loss: bool = False, **kwargs) -> Dict:
        features = self.backbone(x)
        neck_feats = self.neck(features)
        preds = [head(f) for head, f in zip(self.heads, neck_feats)]

        result: Dict = {"preds": preds}

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
            "name": "YOLOv11",
            "num_classes": self.num_classes,
            "total_params": total,
            "trainable_params": trainable,
            "params_mb": total * 4 / (1024 ** 2),
        }
