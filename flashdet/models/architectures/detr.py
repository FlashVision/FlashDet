"""
DETR — DEtection TRansformer.

End-to-end object detection with a transformer encoder-decoder architecture
and Hungarian (bipartite) matching for set-based loss.

Architecture:
    ResNet backbone → positional encoding → Transformer encoder-decoder
    → FFN prediction heads (class + bbox per query)

Reference:
    Carion et al., "End-to-End Object Detection with Transformers", ECCV 2020.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from flashdet.registry import DETECTORS
from flashdet.models.backbone.resnet import ResNetBackbone
from flashdet.models.transformer import PositionalEncoding2D, DETRTransformer
from flashdet.models.head.detr_head import DETRHead
from flashdet.models.assignment.hungarian_matcher import HungarianMatcher

logger = logging.getLogger(__name__)


@DETECTORS.register("DETR")
class DETR(nn.Module):
    """DETR: End-to-End Object Detection with Transformers.

    Args:
        num_classes: Number of detection classes (excluding background).
        num_queries: Number of object queries.
        d_model: Transformer hidden dimension.
        nhead: Number of attention heads.
        num_encoder_layers: Transformer encoder depth.
        num_decoder_layers: Transformer decoder depth.
        dim_feedforward: FFN hidden size.
        dropout: Dropout rate.
        backbone: ResNet variant for the backbone.
        pretrained_backbone: Load ImageNet-pretrained backbone.
        cost_class: Hungarian matching class cost weight.
        cost_bbox: Hungarian matching L1 bbox cost weight.
        cost_giou: Hungarian matching GIoU cost weight.
        loss_ce_weight: Classification loss weight.
        loss_bbox_weight: L1 bbox regression loss weight.
        loss_giou_weight: GIoU loss weight.
    """

    def __init__(
        self,
        num_classes: int = 80,
        num_queries: int = 100,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        backbone: str = "resnet50",
        pretrained_backbone: bool = True,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        loss_ce_weight: float = 1.0,
        loss_bbox_weight: float = 5.0,
        loss_giou_weight: float = 2.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.d_model = d_model
        self.loss_ce_weight = loss_ce_weight
        self.loss_bbox_weight = loss_bbox_weight
        self.loss_giou_weight = loss_giou_weight

        self.backbone = ResNetBackbone(backbone, d_model, pretrained_backbone)
        self.pos_encoder = PositionalEncoding2D(d_model)
        self.transformer = DETRTransformer(
            d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout,
        )

        self.query_embed = nn.Embedding(num_queries, d_model)
        self.head = DETRHead(d_model, num_classes)
        self.matcher = HungarianMatcher(cost_class, cost_bbox, cost_giou)

        nn.init.uniform_(self.query_embed.weight)

    def forward(
        self,
        x: torch.Tensor,
        gt_meta: Optional[Dict] = None,
        compute_loss: bool = False,
        **kwargs,
    ) -> Dict:
        features = self.backbone(x)
        pos = self.pos_encoder(features)

        hs = self.transformer(features, self.query_embed.weight, pos)

        pred_logits, pred_boxes = self.head(hs)

        if (self.training or compute_loss) and gt_meta is not None:
            gt_meta["img"] = x
            from flashdet.losses.detr_loss import compute_detr_loss
            loss, loss_states = compute_detr_loss(
                pred_logits, pred_boxes, gt_meta,
                self.num_classes, self.matcher, self.num_queries,
                x.shape[2:],
                self.loss_ce_weight, self.loss_bbox_weight, self.loss_giou_weight,
            )
            return {"loss": loss, "loss_states": loss_states, "preds": {"logits": pred_logits, "boxes": pred_boxes}}

        return {"preds": {"logits": pred_logits, "boxes": pred_boxes}}

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        img_metas=None,
        score_thr: float = 0.05,
        nms_thr: float = 0.6,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Run inference. Returns ``[(dets, labels), ...]`` per image."""
        self.eval()
        out = self.forward(x)
        from flashdet.engine.inference.postprocess import decode_detr_predictions
        return decode_detr_predictions(
            out["preds"]["logits"], out["preds"]["boxes"],
            x.shape[2:], score_thr=score_thr, use_softmax=True,
        )

    def get_model_info(self) -> Dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "name": "DETR",
            "num_classes": self.num_classes,
            "num_queries": self.num_queries,
            "total_params": total,
            "trainable_params": trainable,
            "params_mb": total * 4 / (1024 ** 2),
        }
