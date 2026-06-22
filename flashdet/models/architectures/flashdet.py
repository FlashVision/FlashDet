"""
FlashDet — YOLO26-Based NMS-Free Object Detector.

Architecture built on YOLO26 principles:
  - YOLO11 Backbone (C3k2 + SPPF + C2PSA)
  - YOLO11 PAN-FPN Neck
  - DFL-free Dual Detection Head (One-to-One + One-to-Many)
  - STAL (Small-Target-Aware Label Assignment)
  - ProgLoss (Progressive Loss shifting from o2m → o2o)
  - MuSGD optimizer (Muon + SGD hybrid)

NMS-Free Inference:
    The one-to-one (o2o) head is trained with 1:1 label assignment —
    each ground truth matched to exactly one anchor. At inference only
    the o2o head runs, producing at most one prediction per object.
    No NMS is needed — just score threshold + top-k.

Model Sizes:
    - FlashDet-P  (ShuffleNetV2-0.5x + GhostPAN):  ~298K inf params
    - FlashDet-N  (width=0.25, depth=0.33): ~1.06M inf params
    - FlashDet-S  (width=0.50, depth=0.33): ~4.2M inf params
    - FlashDet-M  (width=1.00, depth=0.67): ~18M inf params

Reference:
    Ultralytics YOLO26 (2026).
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashdet.registry import DETECTORS
from flashdet.models.backbone.yolov11_backbone import YOLOv11Backbone
from flashdet.models.neck.yolov11_neck import YOLOv11Neck
from flashdet.models.head.e2e_head import E2EDualHead
from flashdet.losses.e2e_loss import E2EDetectionLoss, _make_anchor_grid, _decode_ltrb

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NMS-Free decode utilities (shared by FlashDet and FlashDetPico)
# ---------------------------------------------------------------------------

def _decode_batch_nms_free(
    cls_logits: torch.Tensor,
    reg_preds: torch.Tensor,
    anchor_centers: torch.Tensor,
    anchor_strides: torch.Tensor,
    img_hw: Tuple[int, int],
    score_thr: float = 0.25,
    max_det: int = 300,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Pure NMS-free batch decode — score threshold + top-k only.

    The o2o head produces one prediction per anchor with 1:1 assignment,
    so duplicates do not exist. This is the YOLO26 inference paradigm.

    Args:
        cls_logits: [B, N, num_classes] raw logits from o2o head.
        reg_preds:  [B, N, 4] raw LTRB regression from o2o head.
        anchor_centers: [N, 2] precomputed anchor grid centers.
        anchor_strides: [N, 1] stride per anchor.
        img_hw: (H, W) of the input image tensor.
        score_thr: Minimum confidence to keep.
        max_det: Maximum detections per image.

    Returns:
        List of (det_bboxes [M, 5], det_labels [M]) per image.
    """
    H, W = img_hw
    B = cls_logits.shape[0]
    device = cls_logits.device
    results: List[Tuple[torch.Tensor, torch.Tensor]] = []

    for b in range(B):
        scores = cls_logits[b].sigmoid()                   # [N, C]
        max_scores, labels = scores.max(dim=1)             # [N], [N]

        keep = max_scores > score_thr
        if keep.sum() == 0:
            results.append((
                torch.zeros((0, 5), device=device),
                torch.zeros((0,), dtype=torch.long, device=device),
            ))
            continue

        scores_k = max_scores[keep]
        labels_k = labels[keep]
        reg_k = reg_preds[b][keep]
        centers_k = anchor_centers[keep]
        strides_k = anchor_strides[keep]

        boxes_k = _decode_ltrb(centers_k, strides_k, reg_k)
        boxes_k[:, 0].clamp_(min=0, max=W)
        boxes_k[:, 1].clamp_(min=0, max=H)
        boxes_k[:, 2].clamp_(min=0, max=W)
        boxes_k[:, 3].clamp_(min=0, max=H)

        # Top-k by score (no NMS — o2o head is already duplicate-free)
        if scores_k.shape[0] > max_det:
            topk_idx = scores_k.topk(max_det).indices
            scores_k = scores_k[topk_idx]
            boxes_k = boxes_k[topk_idx]
            labels_k = labels_k[topk_idx]

        det_bboxes = torch.cat([boxes_k, scores_k[:, None]], dim=1)
        results.append((det_bboxes, labels_k))

    return results

SIZE_CONFIGS = {
    "p": {"type": "pico"},
    "n": {"width_mult": 0.25, "depth_mult": 0.33, "use_c2psa": True},
    "s": {"width_mult": 0.50, "depth_mult": 0.33, "use_c2psa": True},
    "m": {"width_mult": 1.00, "depth_mult": 0.67, "use_c2psa": True},
    "l": {"width_mult": 1.25, "depth_mult": 1.00, "use_c2psa": True},
    "x": {"width_mult": 1.50, "depth_mult": 1.00, "use_c2psa": True},
}


class _PicoDetHead(nn.Module):
    """Ultra-lightweight depthwise-separable detection head for Pico.

    Uses depthwise separable convolutions instead of full convolutions
    to minimize parameters while maintaining receptive field.
    """

    def __init__(self, num_classes: int, in_channels: int):
        super().__init__()
        self.num_classes = num_classes

        self.cls_convs = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )
        self.reg_convs = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )
        self.cls_pred = nn.Conv2d(in_channels, num_classes, 1)
        self.reg_pred = nn.Conv2d(in_channels, 4, 1)

    def forward(self, x):
        return self.cls_pred(self.cls_convs(x)), self.reg_pred(self.reg_convs(x))


class _PicoDualHead(nn.Module):
    """Pico dual detection head with shared depthwise stems."""

    def __init__(self, num_classes: int, in_channels: int, num_levels: int = 3):
        super().__init__()
        self.o2o_heads = nn.ModuleList([
            _PicoDetHead(num_classes, in_channels) for _ in range(num_levels)
        ])
        self.o2m_heads = nn.ModuleList([
            _PicoDetHead(num_classes, in_channels) for _ in range(num_levels)
        ])

    def forward(self, features, training=True):
        o2o_cls_list, o2o_reg_list, feat_sizes = [], [], []
        for head, feat in zip(self.o2o_heads, features):
            cls, reg = head(feat)
            B, _, H, W = cls.shape
            feat_sizes.append((H, W))
            o2o_cls_list.append(cls.permute(0, 2, 3, 1).reshape(B, H * W, -1))
            o2o_reg_list.append(reg.permute(0, 2, 3, 1).reshape(B, H * W, 4))

        result = {
            "o2o_cls": torch.cat(o2o_cls_list, dim=1),
            "o2o_reg": torch.cat(o2o_reg_list, dim=1),
            "feat_sizes": feat_sizes,
        }
        if training:
            o2m_cls_list, o2m_reg_list = [], []
            for head, feat in zip(self.o2m_heads, features):
                cls, reg = head(feat)
                B, _, H, W = cls.shape
                o2m_cls_list.append(cls.permute(0, 2, 3, 1).reshape(B, H * W, -1))
                o2m_reg_list.append(reg.permute(0, 2, 3, 1).reshape(B, H * W, 4))
            result["o2m_cls"] = torch.cat(o2m_cls_list, dim=1)
            result["o2m_reg"] = torch.cat(o2m_reg_list, dim=1)
        return result


@DETECTORS.register("FlashDetPico")
class FlashDetPico(nn.Module):
    """FlashDet-Pico — Sub-1MB object detector.

    Architecture optimized for extreme efficiency:
      - ShuffleNetV2-0.5x backbone (channel shuffle + depthwise, ImageNet pretrained)
      - GhostPAN neck with 64-ch output (Ghost modules for cheap features)
      - Depthwise-separable E2E dual head
      - STAL + ProgLoss (same training recipe as larger FlashDet)

    Target: < 500K inference params = < 1MB FP16 weight file.

    Model Stats:
      - Inference params: ~397K  (~0.76 MB FP16)
      - Training params:  ~479K  (incl. o2m head)
    """

    def __init__(
        self,
        num_classes: int = 80,
        strides: Tuple[int, ...] = (8, 16, 32),
        total_epochs: int = 100,
        neck_channels: int = 64,
        pretrained_backbone: bool = True,
        **kwargs,
    ):
        super().__init__()
        from flashdet.models.backbone.shufflenet import ShuffleNetV2
        from flashdet.models.neck.ghost_pan import GhostPAN

        self.num_classes = num_classes
        self.size = "p"
        self.strides = strides
        self.total_epochs = total_epochs

        self.backbone = ShuffleNetV2(
            model_size="0.5x",
            out_stages=(2, 3, 4),
            pretrained=pretrained_backbone,
            activation="LeakyReLU",
        )

        self.neck = GhostPAN(
            in_channels=self.backbone.out_channels,
            out_channels=neck_channels,
            kernel_size=5,
            num_blocks=1,
            use_depthwise=True,
            activation="LeakyReLU",
        )

        self.head = _PicoDualHead(num_classes, neck_channels, num_levels=3)

        self.loss_fn = E2EDetectionLoss(
            num_classes=num_classes,
            strides=strides,
            alpha_init=kwargs.get("prog_alpha_init", 1.0),
            alpha_final=kwargs.get("prog_alpha_final", 0.0),
            o2m_topk=kwargs.get("o2m_topk", 10),
            o2o_topk=kwargs.get("o2o_topk", 7),
            box_weight=kwargs.get("box_weight", 7.5),
            cls_weight=kwargs.get("cls_weight", 1.0),
            l1_weight=kwargs.get("l1_weight", 0.5),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        for heads in [self.head.o2o_heads, self.head.o2m_heads]:
            for h in heads:
                nn.init.constant_(h.cls_pred.bias, -4.595)

    def forward(self, x, gt_meta=None, epoch=0, compute_loss=False, return_features=False, **kwargs):
        features = self.backbone(x)
        neck_feats = self.neck(features)
        head_out = self.head(neck_feats, training=self.training or compute_loss)

        result = {}
        if return_features:
            result["backbone_features"] = features
            result["fpn_features"] = neck_feats

        if (self.training or compute_loss) and gt_meta is not None:
            device = x.device
            gt_bboxes_list = [torch.as_tensor(b, dtype=torch.float32, device=device).reshape(-1, 4)
                              for b in gt_meta["gt_bboxes"]]
            gt_labels_list = [torch.as_tensor(l, dtype=torch.long, device=device).reshape(-1)
                              for l in gt_meta["gt_labels"]]

            loss, loss_states = self.loss_fn(
                o2o_cls=head_out["o2o_cls"], o2o_reg=head_out["o2o_reg"],
                o2m_cls=head_out["o2m_cls"], o2m_reg=head_out["o2m_reg"],
                gt_bboxes_list=gt_bboxes_list, gt_labels_list=gt_labels_list,
                feat_sizes=head_out["feat_sizes"],
                epoch=epoch, total_epochs=self.total_epochs,
            )
            result["loss"] = loss
            result["loss_states"] = loss_states

        result["preds"] = head_out["o2o_cls"]
        result["o2o_cls"] = head_out["o2o_cls"]
        result["o2o_reg"] = head_out["o2o_reg"]
        result["feat_sizes"] = head_out["feat_sizes"]
        return result

    # ------ Anchor grid cache (avoids recompute per call) ------
    _cached_anchors: Optional[Tuple[Tuple[int, ...], torch.Tensor, torch.Tensor]] = None

    def _get_anchors(self, feat_sizes, device):
        key = tuple((h, w) for h, w in feat_sizes)
        if self._cached_anchors is not None and self._cached_anchors[0] == key:
            c, s = self._cached_anchors[1], self._cached_anchors[2]
            if c.device == device:
                return c, s
        centers, strides = _make_anchor_grid(feat_sizes, list(self.strides), device)
        self._cached_anchors = (key, centers, strides)
        return centers, strides

    @torch.no_grad()
    def predict(self, x, img_metas=None, score_thr=0.25, max_det=300, **kwargs):
        """NMS-free inference — score threshold + top-k only.

        The o2o head produces 1:1 predictions (one per object), so NMS
        is unnecessary. This is significantly faster on CPU.
        """
        self.eval()
        out = self.forward(x)
        anchor_centers, anchor_strides = self._get_anchors(
            out["feat_sizes"], x.device
        )
        return _decode_batch_nms_free(
            out["o2o_cls"], out["o2o_reg"],
            anchor_centers, anchor_strides,
            img_hw=(x.shape[2], x.shape[3]),
            score_thr=score_thr, max_det=max_det,
        )

    def strip_o2m(self):
        """Remove one-to-many heads and loss for lean CPU/edge deployment.

        After calling this the model cannot be trained, but inference is
        faster and the checkpoint is smaller.
        """
        del self.head.o2m_heads
        del self.loss_fn
        self.head.o2m_heads = None  # type: ignore[assignment]
        self.loss_fn = None  # type: ignore[assignment]
        logger.info("Stripped o2m heads + loss for NMS-free inference-only mode")
        return self

    def get_model_info(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        o2m_params = sum(p.numel() for p in self.head.o2m_heads.parameters())
        inference_params = total - o2m_params
        return {
            "name": "FlashDet-P",
            "num_classes": self.num_classes,
            "size": "p",
            "total_params": total,
            "trainable_params": trainable,
            "inference_params": inference_params,
            "params_mb": total * 4 / (1024 ** 2),
            "inference_params_mb": inference_params * 4 / (1024 ** 2),
            "inference_fp16_mb": inference_params * 2 / (1024 ** 2),
        }


@DETECTORS.register("FlashDet")
class FlashDet(nn.Module):
    """FlashDet — YOLO26-based lightweight object detector.

    Args:
        num_classes: Number of detection classes.
        size: Model size variant ("p", "n", "s", "m", "l", "x").
            "p" (Pico) returns a :class:`FlashDetPico` — a sub-1MB model
            using ShuffleNetV2 + GhostPAN + depthwise heads.
        width_mult: Override width multiplier (ignored if size is set).
        depth_mult: Override depth multiplier (ignored if size is set).
        use_c2psa: Use C2PSA attention in backbone.
        strides: Feature pyramid strides.
        total_epochs: Total training epochs (for ProgLoss scheduling).
        prog_alpha_init: ProgLoss initial o2m weight.
        prog_alpha_final: ProgLoss final o2m weight.
        o2m_topk: One-to-many assigner top-k.
        o2o_topk: One-to-one assigner top-k.
        box_weight: CIoU loss weight.
        cls_weight: Classification loss weight.
        l1_weight: L1 distance loss weight.
    """

    def __new__(cls, num_classes=80, size="n", **kwargs):
        if size == "p":
            return FlashDetPico(num_classes=num_classes, **kwargs)
        return super().__new__(cls)

    def __init__(
        self,
        num_classes: int = 80,
        size: str = "n",
        width_mult: Optional[float] = None,
        depth_mult: Optional[float] = None,
        use_c2psa: Optional[bool] = None,
        strides: Tuple[int, ...] = (8, 16, 32),
        total_epochs: int = 100,
        prog_alpha_init: float = 1.0,
        prog_alpha_final: float = 0.0,
        o2m_topk: int = 10,
        o2o_topk: int = 7,
        box_weight: float = 7.5,
        cls_weight: float = 1.0,
        l1_weight: float = 0.5,
    ):
        if size == "p":
            return
        super().__init__()

        cfg = SIZE_CONFIGS.get(size, SIZE_CONFIGS["n"])
        wm = width_mult if width_mult is not None else cfg["width_mult"]
        dm = depth_mult if depth_mult is not None else cfg["depth_mult"]
        c2psa = use_c2psa if use_c2psa is not None else cfg["use_c2psa"]

        self.num_classes = num_classes
        self.size = size
        self.strides = strides
        self.total_epochs = total_epochs

        # Backbone: YOLO11 C3k2 + SPPF + C2PSA
        self.backbone = YOLOv11Backbone(wm, dm, c2psa)

        # Neck: YOLO11 PAN-FPN
        neck_out = self.backbone.out_channels[1]
        self.neck = YOLOv11Neck(self.backbone.out_channels, neck_out)

        # Head: DFL-free dual detection head (o2o + o2m)
        self.head = E2EDualHead(num_classes, neck_out, num_levels=3)

        # Loss: E2E with ProgLoss + STAL
        self.loss_fn = E2EDetectionLoss(
            num_classes=num_classes,
            strides=strides,
            alpha_init=prog_alpha_init,
            alpha_final=prog_alpha_final,
            o2m_topk=o2m_topk,
            o2o_topk=o2o_topk,
            box_weight=box_weight,
            cls_weight=cls_weight,
            l1_weight=l1_weight,
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Initialize cls bias for low initial false-positive rate
        for heads in [self.head.o2o_heads, self.head.o2m_heads]:
            for h in heads:
                nn.init.constant_(h.cls_pred.bias, -4.595)

    def forward(
        self,
        x: torch.Tensor,
        gt_meta: Optional[Dict] = None,
        epoch: int = 0,
        compute_loss: bool = False,
        return_features: bool = False,
        **kwargs,
    ) -> Dict:
        """Forward pass.

        Args:
            x: [B, 3, H, W] input tensor.
            gt_meta: Ground truth with "gt_bboxes" and "gt_labels".
            epoch: Current training epoch (for ProgLoss schedule).
            compute_loss: Force loss computation even in eval mode.
            return_features: Include backbone/neck features in output.

        Returns:
            Training: {"loss", "loss_states"} and optionally features.
            Inference: {"preds"} — o2o head outputs for NMS-free decode.
        """
        features = self.backbone(x)
        neck_feats = self.neck(features)
        head_out = self.head(neck_feats, training=self.training or compute_loss)

        result: Dict = {}

        if return_features:
            result["backbone_features"] = features
            result["fpn_features"] = neck_feats

        if (self.training or compute_loss) and gt_meta is not None:
            device = x.device
            gt_bboxes_list = []
            gt_labels_list = []
            for b_bboxes, b_labels in zip(gt_meta["gt_bboxes"], gt_meta["gt_labels"]):
                gt_bboxes_list.append(torch.as_tensor(b_bboxes, dtype=torch.float32, device=device).reshape(-1, 4))
                gt_labels_list.append(torch.as_tensor(b_labels, dtype=torch.long, device=device).reshape(-1))

            loss, loss_states = self.loss_fn(
                o2o_cls=head_out["o2o_cls"],
                o2o_reg=head_out["o2o_reg"],
                o2m_cls=head_out["o2m_cls"],
                o2m_reg=head_out["o2m_reg"],
                gt_bboxes_list=gt_bboxes_list,
                gt_labels_list=gt_labels_list,
                feat_sizes=head_out["feat_sizes"],
                epoch=epoch,
                total_epochs=self.total_epochs,
            )

            result["loss"] = loss
            result["loss_states"] = loss_states
            result["preds"] = head_out["o2o_cls"]  # for compatibility
            result["o2o_cls"] = head_out["o2o_cls"]
            result["o2o_reg"] = head_out["o2o_reg"]
            result["feat_sizes"] = head_out["feat_sizes"]
        else:
            result["preds"] = head_out["o2o_cls"]
            result["o2o_cls"] = head_out["o2o_cls"]
            result["o2o_reg"] = head_out["o2o_reg"]
            result["feat_sizes"] = head_out["feat_sizes"]

        return result

    # ------ Anchor grid cache (avoids recompute per call) ------
    _cached_anchors: Optional[Tuple[Tuple[int, ...], torch.Tensor, torch.Tensor]] = None

    def _get_anchors(self, feat_sizes, device):
        key = tuple((h, w) for h, w in feat_sizes)
        if self._cached_anchors is not None and self._cached_anchors[0] == key:
            c, s = self._cached_anchors[1], self._cached_anchors[2]
            if c.device == device:
                return c, s
        centers, strides = _make_anchor_grid(feat_sizes, list(self.strides), device)
        self._cached_anchors = (key, centers, strides)
        return centers, strides

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        img_metas: Optional[Dict] = None,
        score_thr: float = 0.25,
        max_det: int = 300,
        **kwargs,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """NMS-free inference — YOLO26 one-to-one head, no post-processing.

        The o2o head is trained with 1:1 label assignment so each object
        produces exactly one prediction. No NMS is needed — just score
        threshold + top-k, making this extremely fast on CPU.

        Returns:
            List of (det_bboxes [N,5], det_labels [N]) per image.
        """
        self.eval()
        out = self.forward(x)
        anchor_centers, anchor_strides = self._get_anchors(
            out["feat_sizes"], x.device
        )
        return _decode_batch_nms_free(
            out["o2o_cls"], out["o2o_reg"],
            anchor_centers, anchor_strides,
            img_hw=(x.shape[2], x.shape[3]),
            score_thr=score_thr, max_det=max_det,
        )

    def strip_o2m(self):
        """Remove one-to-many heads and loss for lean CPU/edge deployment.

        After calling this the model cannot be trained, but inference is
        faster and the checkpoint is smaller.
        """
        del self.head.o2m_heads
        del self.loss_fn
        self.head.o2m_heads = None  # type: ignore[assignment]
        self.loss_fn = None  # type: ignore[assignment]
        logger.info("Stripped o2m heads + loss for NMS-free inference-only mode")
        return self

    def get_model_info(self) -> Dict:
        """Get model information."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Inference uses only o2o heads (o2m is training-only)
        o2m_params = sum(p.numel() for p in self.head.o2m_heads.parameters())
        inference_params = total - o2m_params

        return {
            "name": f"FlashDet-{self.size.upper()}",
            "num_classes": self.num_classes,
            "size": self.size,
            "total_params": total,
            "trainable_params": trainable,
            "inference_params": inference_params,
            "params_mb": total * 4 / (1024 ** 2),
            "inference_params_mb": inference_params * 4 / (1024 ** 2),
            "inference_fp16_mb": inference_params * 2 / (1024 ** 2),
        }
