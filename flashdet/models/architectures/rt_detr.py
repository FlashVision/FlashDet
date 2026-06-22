"""
RT-DETR — Real-Time DEtection TRansformer.

A real-time end-to-end detector that eliminates NMS post-processing.
Uses a hybrid encoder (CNN backbone + intra-scale and cross-scale
transformer feature fusion) with IoU-aware query selection.

Architecture:
    ResNet/HGNetv2 backbone → Hybrid Encoder (AIFI + CCFM) →
    Transformer Decoder with IoU-aware query selection → prediction heads

Reference:
    Lv et al., "DETRs Beat YOLOs on Real-time Object Detection", 2023.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from flashdet.registry import DETECTORS
from flashdet.models.backbone.resnet import ResNetMultiScaleBackbone
from flashdet.models.neck.hybrid_encoder import HybridEncoder
from flashdet.models.head.rt_detr_decoder import RTDETRDecoder

logger = logging.getLogger(__name__)


@DETECTORS.register("RTDETR")
class RTDETR(nn.Module):
    """RT-DETR: Real-Time DEtection TRansformer.

    Args:
        num_classes: Number of detection classes.
        backbone: ResNet variant for backbone.
        hidden_dim: Transformer/encoder hidden dimension.
        nhead: Number of attention heads.
        num_encoder_layers: AIFI encoder depth.
        num_decoder_layers: Transformer decoder depth.
        dim_feedforward: FFN hidden size.
        num_queries: Number of selected object queries.
        num_csp_blocks: Number of RepVGG blocks in each CSP layer.
        pretrained_backbone: Whether to load pretrained backbone.
    """

    def __init__(
        self,
        num_classes: int = 80,
        backbone: str = "resnet50",
        hidden_dim: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 1,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 1024,
        num_queries: int = 300,
        num_csp_blocks: int = 3,
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries

        self.backbone = ResNetMultiScaleBackbone(backbone, pretrained_backbone)

        self.encoder = HybridEncoder(
            in_channels=self.backbone.out_channels, hidden_dim=hidden_dim,
            nhead=nhead, dim_feedforward=dim_feedforward,
            num_encoder_layers=num_encoder_layers,
            num_csp_blocks=num_csp_blocks,
        )

        self.decoder = RTDETRDecoder(
            d_model=hidden_dim, nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            num_queries=num_queries,
            num_classes=num_classes,
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.encoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.decoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        gt_meta: Optional[Dict] = None,
        compute_loss: bool = False,
        **kwargs,
    ) -> Dict:
        input_tensor = x
        backbone_feats = self.backbone(x)

        encoder_outs = self.encoder(backbone_feats)
        dec_out = self.decoder(encoder_outs)

        preds = {"logits": dec_out["pred_logits"], "boxes": dec_out["pred_boxes"]}

        if (self.training or compute_loss) and gt_meta is not None:
            gt_meta["img"] = input_tensor
            from flashdet.losses.rt_detr_loss import compute_rt_detr_loss
            loss, loss_states = compute_rt_detr_loss(
                dec_out, gt_meta, self.num_classes,
                self.num_queries, input_tensor.shape[2:],
            )
            return {"loss": loss, "loss_states": loss_states, "preds": preds}

        return {"preds": preds}

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
            x.shape[2:], score_thr=score_thr, use_softmax=False,
        )

    def get_model_info(self) -> Dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "name": "RT-DETR",
            "num_classes": self.num_classes,
            "num_queries": self.num_queries,
            "total_params": total,
            "trainable_params": trainable,
            "params_mb": total * 4 / (1024 ** 2),
        }
