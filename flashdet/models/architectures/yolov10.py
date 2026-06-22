"""
YOLOv10 — NMS-Free Real-Time End-to-End Object Detection.

Key innovations:
  - Dual label assignment: one-to-many for training, one-to-one for inference
  - Consistent dual assignment for NMS-free end-to-end detection
  - Efficiency-driven model design with spatial-channel decoupled downsampling

Reference:
    Wang et al., "YOLOv10: Real-Time End-to-End Object Detection", 2024.
"""

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from flashdet.registry import DETECTORS
from flashdet.models.backbone.yolov10_backbone import YOLOv10Backbone
from flashdet.models.neck.yolov10_neck import YOLOv10Neck
from flashdet.models.head.yolo_head import DualHeadOne2One, DualHeadOne2Many

logger = logging.getLogger(__name__)


@DETECTORS.register("YOLOv10")
class YOLOv10(nn.Module):
    """YOLOv10 detector with NMS-free end-to-end design.

    Uses consistent dual assignment: one-to-many head for training supervision
    and one-to-one head for NMS-free inference.

    Args:
        num_classes: Number of detection classes.
        width_mult: Channel width multiplier.
        depth_mult: Depth multiplier for bottleneck blocks.
        reg_max: DFL regression max value.
        use_psa: Use Partial Self-Attention in backbone.
    """

    def __init__(
        self,
        num_classes: int = 80,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        reg_max: int = 16,
        use_psa: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max

        self.backbone = YOLOv10Backbone(width_mult, depth_mult, use_psa)

        neck_out = self.backbone.out_channels[1]  # channels[2]
        self.neck = YOLOv10Neck(self.backbone.out_channels, neck_out)

        self.o2o_heads = nn.ModuleList([
            DualHeadOne2One(num_classes, neck_out, reg_max) for _ in range(3)
        ])
        self.o2m_heads = nn.ModuleList([
            DualHeadOne2Many(num_classes, neck_out, reg_max) for _ in range(3)
        ])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, gt_meta: Optional[Dict] = None, compute_loss: bool = False, **kwargs) -> Dict:
        features = self.backbone(x)
        neck_feats = self.neck(features)

        o2m_preds = [h(f) for h, f in zip(self.o2m_heads, neck_feats)]
        result: Dict = {"preds": o2m_preds}

        if (self.training or compute_loss) and gt_meta is not None:
                from flashdet.losses.yolo_loss import compute_yolo_loss
                gt_meta["input_h"] = x.shape[2]
                loss, loss_states = compute_yolo_loss(
                    o2m_preds, gt_meta, self.num_classes, reg_max=self.reg_max,
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
            "name": "YOLOv10",
            "num_classes": self.num_classes,
            "total_params": total,
            "trainable_params": trainable,
            "params_mb": total * 4 / (1024 ** 2),
        }
