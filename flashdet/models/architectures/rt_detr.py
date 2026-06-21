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

import math
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from flashdet.registry import BACKBONES

logger = logging.getLogger(__name__)


class RepVGGBlock(nn.Module):
    """RepVGG-style block with re-parameterisable 3x3+1x1+identity branches."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.use_identity = (stride == 1 and in_channels == out_channels)
        if self.use_identity:
            self.bn_identity = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv3x3(x) + self.conv1x1(x)
        if self.use_identity:
            out = out + self.bn_identity(x)
        return self.act(out)


class CSPRepLayer(nn.Module):
    """CSP bottleneck layer with RepVGG blocks used in the CCFM."""

    def __init__(self, in_channels: int, out_channels: int, num_blocks: int = 3, expansion: float = 1.0):
        super().__init__()
        hidden = int(out_channels * expansion)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[RepVGGBlock(hidden, hidden) for _ in range(num_blocks)]
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden * 2, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv3(torch.cat([self.blocks(self.conv1(x)), self.conv2(x)], dim=1))


class AIFI(nn.Module):
    """Attention-based Intra-scale Feature Interaction.

    Applies self-attention within a single scale feature map using a standard
    transformer encoder layer with 2-D positional encoding.
    """

    def __init__(self, d_model: int = 256, nhead: int = 8, dim_feedforward: int = 1024, num_layers: int = 1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=0.0, batch_first=True, norm_first=True,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def _build_2d_sincos_pos(self, H: int, W: int, d_model: int, device: torch.device) -> torch.Tensor:
        half = d_model // 2
        dim = torch.arange(half // 2, dtype=torch.float32, device=device)
        dim = 10000.0 ** (2 * (dim // 2) / (half // 2))

        pos_y = torch.arange(H, dtype=torch.float32, device=device).unsqueeze(1) / dim
        pos_x = torch.arange(W, dtype=torch.float32, device=device).unsqueeze(1) / dim

        pe_y = torch.cat([pos_y.sin(), pos_y.cos()], dim=-1).unsqueeze(1).expand(-1, W, -1)
        pe_x = torch.cat([pos_x.sin(), pos_x.cos()], dim=-1).unsqueeze(0).expand(H, -1, -1)

        return torch.cat([pe_y, pe_x], dim=-1).reshape(H * W, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        pos = self._build_2d_sincos_pos(H, W, C, x.device)

        tokens = x.flatten(2).permute(0, 2, 1)
        tokens = tokens + pos.unsqueeze(0)

        for layer in self.layers:
            tokens = layer(tokens)
        tokens = self.norm(tokens)

        return tokens.permute(0, 2, 1).reshape(B, C, H, W)


class HybridEncoder(nn.Module):
    """RT-DETR Hybrid Encoder: AIFI (intra-scale) + CCFM (cross-scale fusion).

    Takes multi-scale features from the backbone, applies AIFI to the
    highest-resolution feature, then fuses features top-down and
    bottom-up through CCFM layers.
    """

    def __init__(
        self,
        in_channels: List[int],
        hidden_dim: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        num_encoder_layers: int = 1,
        num_csp_blocks: int = 3,
        expansion: float = 1.0,
    ):
        super().__init__()
        self.num_scales = len(in_channels)
        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
            )
            for ch in in_channels
        ])

        self.aifi = AIFI(hidden_dim, nhead, dim_feedforward, num_encoder_layers)

        self.top_down_layers = nn.ModuleList()
        self.top_down_csp = nn.ModuleList()
        self.bottom_up_layers = nn.ModuleList()
        self.bottom_up_csp = nn.ModuleList()

        for _ in range(self.num_scales - 1):
            self.top_down_layers.append(nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True),
            ))
            self.top_down_csp.append(CSPRepLayer(hidden_dim * 2, hidden_dim, num_csp_blocks, expansion))

        for _ in range(self.num_scales - 1):
            self.bottom_up_layers.append(nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, 2, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True),
            ))
            self.bottom_up_csp.append(CSPRepLayer(hidden_dim * 2, hidden_dim, num_csp_blocks, expansion))

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        projected = [proj(f) for proj, f in zip(self.input_proj, features)]

        projected[-1] = self.aifi(projected[-1])

        # Top-down path
        td_outs = [None] * self.num_scales
        td_outs[-1] = projected[-1]
        for i in range(self.num_scales - 2, -1, -1):
            up = F.interpolate(self.top_down_layers[i](td_outs[i + 1]),
                               size=projected[i].shape[2:], mode="nearest")
            td_outs[i] = self.top_down_csp[i](torch.cat([up, projected[i]], dim=1))

        # Bottom-up path
        bu_outs = [None] * self.num_scales
        bu_outs[0] = td_outs[0]
        for i in range(self.num_scales - 1):
            down = self.bottom_up_layers[i](bu_outs[i])
            bu_outs[i + 1] = self.bottom_up_csp[i](torch.cat([down, td_outs[i + 1]], dim=1))

        return bu_outs


class RTDETRDecoder(nn.Module):
    """RT-DETR Transformer decoder with IoU-aware query selection."""

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        num_queries: int = 300,
        num_classes: int = 80,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.d_model = d_model
        self.num_classes = num_classes

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.decoder_norm = nn.LayerNorm(d_model)

        self.enc_cls_head = nn.Linear(d_model, num_classes)
        self.enc_bbox_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
            nn.Linear(d_model, 4),
        )
        self.enc_iou_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
            nn.Linear(d_model, 1),
        )

        self.dec_cls_head = nn.Linear(d_model, num_classes)
        self.dec_bbox_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
            nn.Linear(d_model, 4),
        )

        self.query_pos_head = nn.Sequential(
            nn.Linear(4, d_model * 2), nn.ReLU(inplace=True),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, encoder_features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Flatten and concatenate multi-scale encoder features
        feat_list = []
        for feat in encoder_features:
            feat_list.append(feat.flatten(2).permute(0, 2, 1))
        memory = torch.cat(feat_list, dim=1)
        B, N, C = memory.shape

        enc_cls = self.enc_cls_head(memory)
        enc_bbox = self.enc_bbox_head(memory).sigmoid()
        enc_iou = self.enc_iou_head(memory).squeeze(-1).sigmoid()

        # IoU-aware query selection: score = cls_confidence * iou_score
        cls_scores = enc_cls.sigmoid().max(dim=-1).values
        selection_scores = cls_scores * enc_iou
        _, topk_idx = selection_scores.topk(self.num_queries, dim=1)

        query_embed = torch.gather(memory, 1, topk_idx.unsqueeze(-1).expand(-1, -1, C))
        ref_boxes = torch.gather(enc_bbox, 1, topk_idx.unsqueeze(-1).expand(-1, -1, 4))

        query_pos = self.query_pos_head(ref_boxes)

        hs = self.decoder(query_embed + query_pos, memory)
        hs = self.decoder_norm(hs)

        dec_cls = self.dec_cls_head(hs)
        dec_bbox = (self.dec_bbox_head(hs) + self._inverse_sigmoid(ref_boxes)).sigmoid()

        return {
            "pred_logits": dec_cls,
            "pred_boxes": dec_bbox,
            "enc_logits": enc_cls,
            "enc_boxes": enc_bbox,
            "topk_idx": topk_idx,
        }

    @staticmethod
    def _inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        x = x.clamp(min=eps, max=1 - eps)
        return torch.log(x / (1 - x))


@BACKBONES.register("RTDETR")
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

        import torchvision.models as tv
        backbone_factory = {
            "resnet18": (tv.resnet18, [128, 256, 512]),
            "resnet34": (tv.resnet34, [128, 256, 512]),
            "resnet50": (tv.resnet50, [512, 1024, 2048]),
            "resnet101": (tv.resnet101, [512, 1024, 2048]),
        }
        builder, in_channels = backbone_factory[backbone]
        weights = "DEFAULT" if pretrained_backbone else None
        resnet = builder(weights=weights)

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.encoder = HybridEncoder(
            in_channels=in_channels, hidden_dim=hidden_dim,
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
        **kwargs,
    ) -> Dict:
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        encoder_outs = self.encoder([c3, c4, c5])
        dec_out = self.decoder(encoder_outs)

        if self.training and gt_meta is not None:
            loss, loss_states = self._compute_loss(dec_out, gt_meta)
            return {"loss": loss, "loss_states": loss_states}

        return {"preds": {"logits": dec_out["pred_logits"], "boxes": dec_out["pred_boxes"]}}

    def _compute_loss(self, dec_out: Dict, gt_meta: Dict) -> Tuple[torch.Tensor, Dict]:
        pred_logits = dec_out["pred_logits"]
        pred_boxes = dec_out["pred_boxes"]
        device = pred_logits.device
        B = pred_logits.shape[0]
        img_shape = gt_meta["img"].shape[2:]

        gt_labels_list, gt_boxes_list = [], []
        for i in range(B):
            gt_b = torch.as_tensor(gt_meta["gt_bboxes"][i], dtype=torch.float32, device=device)
            gt_l = torch.as_tensor(gt_meta["gt_labels"][i], dtype=torch.long, device=device)
            if gt_b.numel() > 0:
                cx = (gt_b[:, 0] + gt_b[:, 2]) / 2 / img_shape[1]
                cy = (gt_b[:, 1] + gt_b[:, 3]) / 2 / img_shape[0]
                w = (gt_b[:, 2] - gt_b[:, 0]) / img_shape[1]
                h = (gt_b[:, 3] - gt_b[:, 1]) / img_shape[0]
                gt_boxes_list.append(torch.stack([cx, cy, w, h], dim=-1))
            else:
                gt_boxes_list.append(gt_b.reshape(0, 4))
            gt_labels_list.append(gt_l)

        # Hungarian matching
        indices = self._match(pred_logits, pred_boxes, gt_labels_list, gt_boxes_list)

        # Classification loss (focal-style via BCE with logits)
        target_classes = torch.full(
            (B, self.num_queries), self.num_classes, dtype=torch.long, device=device,
        )
        for b, (si, ti) in enumerate(indices):
            if si.numel() > 0:
                target_classes[b, si] = gt_labels_list[b][ti]

        # Use focal loss for classification
        target_onehot = F.one_hot(target_classes, self.num_classes + 1)[..., :-1].float()
        loss_cls = self._sigmoid_focal_loss(pred_logits, target_onehot, alpha=0.25, gamma=2.0)

        # Bbox losses
        src_boxes, tgt_boxes = [], []
        for b, (si, ti) in enumerate(indices):
            if si.numel() > 0:
                src_boxes.append(pred_boxes[b, si])
                tgt_boxes.append(gt_boxes_list[b][ti])

        if src_boxes:
            src_cat = torch.cat(src_boxes)
            tgt_cat = torch.cat(tgt_boxes)
            loss_l1 = F.l1_loss(src_cat, tgt_cat, reduction="mean")
            loss_giou = self._giou_loss(src_cat, tgt_cat)
        else:
            loss_l1 = pred_boxes.sum() * 0
            loss_giou = pred_boxes.sum() * 0

        total = loss_cls + 5.0 * loss_l1 + 2.0 * loss_giou

        return total, {
            "loss_cls": loss_cls.detach(),
            "loss_l1": loss_l1.detach(),
            "loss_giou": loss_giou.detach(),
        }

    @staticmethod
    def _sigmoid_focal_loss(
        inputs: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0,
    ) -> torch.Tensor:
        prob = inputs.sigmoid()
        ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce * ((1 - p_t) ** gamma)
        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss
        return loss.mean()

    @staticmethod
    def _giou_loss(src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        def to_xyxy(b):
            cx, cy, w, h = b.unbind(-1)
            return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)

        s, t = to_xyxy(src), to_xyxy(tgt)
        lt = torch.max(s[:, :2], t[:, :2])
        rb = torch.min(s[:, 2:], t[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]
        a1 = (s[:, 2] - s[:, 0]) * (s[:, 3] - s[:, 1])
        a2 = (t[:, 2] - t[:, 0]) * (t[:, 3] - t[:, 1])
        union = a1 + a2 - inter
        iou = inter / union.clamp(min=1e-6)
        enc_lt = torch.min(s[:, :2], t[:, :2])
        enc_rb = torch.max(s[:, 2:], t[:, 2:])
        enc_area = (enc_rb[:, 0] - enc_lt[:, 0]).clamp(min=0) * (enc_rb[:, 1] - enc_lt[:, 1]).clamp(min=0)
        giou = iou - (enc_area - union) / enc_area.clamp(min=1e-6)
        return (1 - giou).mean()

    @torch.no_grad()
    def _match(self, pred_logits, pred_boxes, gt_labels_list, gt_boxes_list):
        B = pred_logits.shape[0]
        indices = []
        for b in range(B):
            if gt_labels_list[b].numel() == 0:
                dev = pred_logits.device
                indices.append((torch.tensor([], dtype=torch.long, device=dev),
                                torch.tensor([], dtype=torch.long, device=dev)))
                continue
            prob = pred_logits[b].sigmoid()
            cost_cls = -prob[:, gt_labels_list[b]]
            cost_l1 = torch.cdist(pred_boxes[b], gt_boxes_list[b], p=1)

            def _to_xyxy(bx):
                cx, cy, w, h = bx.unbind(-1)
                return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)

            s_xy = _to_xyxy(pred_boxes[b])
            t_xy = _to_xyxy(gt_boxes_list[b])
            lt = torch.max(s_xy[:, None, :2], t_xy[None, :, :2])
            rb = torch.min(s_xy[:, None, 2:], t_xy[None, :, 2:])
            wh = (rb - lt).clamp(min=0)
            inter = wh[..., 0] * wh[..., 1]
            a1 = (s_xy[:, 2] - s_xy[:, 0]) * (s_xy[:, 3] - s_xy[:, 1])
            a2 = (t_xy[:, 2] - t_xy[:, 0]) * (t_xy[:, 3] - t_xy[:, 1])
            union = a1[:, None] + a2[None, :] - inter
            iou = inter / union.clamp(min=1e-6)
            enc_lt = torch.min(s_xy[:, None, :2], t_xy[None, :, :2])
            enc_rb = torch.max(s_xy[:, None, 2:], t_xy[None, :, 2:])
            enc_area = (enc_rb[..., 0] - enc_lt[..., 0]).clamp(0) * (enc_rb[..., 1] - enc_lt[..., 1]).clamp(0)
            giou = iou - (enc_area - union) / enc_area.clamp(min=1e-6)
            cost_giou = -giou

            cost = cost_cls + 5.0 * cost_l1 + 2.0 * cost_giou
            row, col = linear_sum_assignment(cost.cpu().numpy())
            dev = pred_logits.device
            indices.append((torch.as_tensor(row, dtype=torch.long, device=dev),
                            torch.as_tensor(col, dtype=torch.long, device=dev)))
        return indices

    @torch.no_grad()
    def predict(self, x: torch.Tensor, score_thr: float = 0.5) -> List[Dict]:
        self.eval()
        out = self.forward(x)
        pred_logits = out["preds"]["logits"]
        pred_boxes = out["preds"]["boxes"]
        B = pred_logits.shape[0]
        h, w = x.shape[2:]
        results = []
        for b in range(B):
            scores = pred_logits[b].sigmoid().max(-1)
            keep = scores.values > score_thr
            boxes = pred_boxes[b, keep]
            boxes_xyxy = torch.stack([
                (boxes[:, 0] - boxes[:, 2] / 2) * w,
                (boxes[:, 1] - boxes[:, 3] / 2) * h,
                (boxes[:, 0] + boxes[:, 2] / 2) * w,
                (boxes[:, 1] + boxes[:, 3] / 2) * h,
            ], dim=-1)
            results.append({
                "boxes": boxes_xyxy,
                "scores": scores.values[keep],
                "labels": scores.indices[keep],
            })
        return results

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
