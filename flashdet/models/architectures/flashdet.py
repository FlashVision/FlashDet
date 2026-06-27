"""
FlashDet — NMS-Free Object Detector.

Architecture (unified RepNeXt-based design across all sizes):
  - Backbone: PicoBlock + StrideDown (reparameterizable multi-scale DW)
  - PicoNeck PAN-FPN (LiteModule-based, depthwise)
  - DFL-free Depthwise Dual Detection Head (One-to-One + One-to-Many)
  - STAL (Small-Target-Aware Label Assignment)
  - ProgLoss (Progressive Loss shifting from o2m → o2o)
  - MuSGD optimizer (Muon + SGD hybrid)

NMS-Free Inference:
    The one-to-one (o2o) head is trained with 1:1 label assignment —
    each ground truth matched to exactly one anchor. At inference only
    the o2o head runs, producing at most one prediction per object.
    No NMS is needed — just score threshold + top-k.

Model Sizes (RepNeXt backbone, strides 8/16/32):
    - FlashDet-P  (PicoBackbone stem=24 + PicoNeck): ~298K inf params
    - FlashDet-N  (FlashBackbone stem=32 + PicoNeck):  ~790K inf params
    - FlashDet-S  (FlashBackbone stem=48 + PicoNeck):  ~1.8M inf params
    - FlashDet-M  (FlashBackbone stem=64 + PicoNeck):  ~3.6M inf params
    - FlashDet-L  (FlashBackbone stem=80 + PicoNeck):  ~5.8M inf params
    - FlashDet-X  (FlashBackbone stem=96 + PicoNeck):  ~9.0M inf params
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from flashdet.registry import DETECTORS
from flashdet.models.backbone.flash_backbone import FlashBackbone
from flashdet.models.neck.pico_neck import PicoNeck
from flashdet.models.head.e2e_head import E2EDualHead
from flashdet.losses.e2e_loss import E2EDetectionLoss
from flashdet.utils.bbox import make_anchor_grid, decode_batch_nms_free

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-size architecture configs
# ---------------------------------------------------------------------------

SIZE_CONFIGS = {
    "p": {"type": "pico"},
    "n": {"stem": 32,  "depths": (2, 4, 2),   "neck_ch": 96,  "neck_blocks": 1},
    "s": {"stem": 48,  "depths": (3, 6, 3),   "neck_ch": 128, "neck_blocks": 1},
    "m": {"stem": 64,  "depths": (4, 8, 4),   "neck_ch": 192, "neck_blocks": 2},
    "l": {"stem": 80,  "depths": (4, 12, 4),  "neck_ch": 256, "neck_blocks": 2},
    "x": {"stem": 96,  "depths": (4, 16, 4),  "neck_ch": 320, "neck_blocks": 2},
}


# ---------------------------------------------------------------------------
# FlashDet-Pico  (sub-1MB detector)
# ---------------------------------------------------------------------------

@DETECTORS.register("FlashDetPico")
class FlashDetPico(nn.Module):
    """FlashDet-Pico — Sub-1MB object detector.

    Architecture optimized for extreme efficiency:
      - LiteBackbone-0.5x (channel mixing + depthwise, ImageNet pretrained)
      - PicoNeck with 64-ch output (lightweight modules for cheap features)
      - Depthwise-separable E2E dual head
      - STAL + ProgLoss (same training recipe as larger FlashDet)

    Target: < 500K inference params = < 1MB FP16 weight file.

    Model Stats:
      - Inference params: ~297K  (~0.57 MB FP16)
      - Training params:  ~344K  (incl. o2m head)
    """

    def __init__(
        self,
        num_classes: int = 80,
        strides: Tuple[int, ...] = (8, 16, 32),
        total_epochs: int = 100,
        neck_channels: int = 64,
        pretrained_backbone: bool = True,
        backbone_type: str = "lite",
        **kwargs,
    ):
        super().__init__()
        from flashdet.models.backbone.lite_backbone import LiteBackbone
        from flashdet.models.backbone.pico_backbone import PicoBackbone

        self.num_classes = num_classes
        self.size = "p"
        self.strides = strides
        self.total_epochs = total_epochs
        self.backbone_type = backbone_type

        if backbone_type in ("pico_v2", "repnext"):
            self.backbone = PicoBackbone(
                stem_channels=24,
                stage_channels=(48, 96, 192),
                stage_depths=(2, 3, 1),
                out_stages=(0, 1, 2),
                activation="LeakyReLU",
            )
        else:
            self.backbone = LiteBackbone(
                model_size="0.5x",
                out_stages=(2, 3, 4),
                pretrained=pretrained_backbone,
                activation="LeakyReLU",
            )

        self.neck = PicoNeck(
            in_channels=self.backbone.out_channels,
            out_channels=neck_channels,
            kernel_size=5,
            num_blocks=1,
            use_depthwise=True,
            activation="LeakyReLU",
        )

        self.head = E2EDualHead(num_classes, neck_channels, num_levels=3)

        self.loss_fn = E2EDetectionLoss(
            num_classes=num_classes,
            strides=strides,
            alpha_init=kwargs.get("prog_alpha_init", 0.8),
            alpha_final=kwargs.get("prog_alpha_final", 0.3),
            o2m_topk=kwargs.get("o2m_topk", 10),
            o2o_topk=kwargs.get("o2o_topk", 7),
            box_weight=kwargs.get("box_weight", 7.5),
            cls_weight=kwargs.get("cls_weight", 0.5),
            l1_weight=kwargs.get("l1_weight", 1.5),
        )

        self._init_weights()

        bb_loaded = getattr(self.backbone, "pretrained_loaded", False)
        if backbone_type in ("pico_v2", "repnext"):
            logger.info("FlashDetPico backbone: PicoBackbone (trained from scratch)")
        else:
            logger.info(
                "FlashDetPico backbone: LiteBackbone-0.5x %s",
                "(ImageNet pretrained)" if bb_loaded else "(RANDOM init)",
            )

    def _init_weights(self):
        """Kaiming-normal init for neck + head; backbone keeps pretrained weights."""
        for module in (self.neck, self.head):
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
        for heads in [self.head.o2o_heads, self.head.o2m_heads]:
            for h in heads:
                nn.init.constant_(h.cls_pred.bias, -4.595)
                nn.init.normal_(h.reg_pred.weight, std=0.001)
                nn.init.zeros_(h.reg_pred.bias)

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
            self._last_loss_states = loss_states

        if return_features or not self.training:
            result["o2o_cls"] = head_out["o2o_cls"]
            result["o2o_reg"] = head_out["o2o_reg"]
            result["feat_sizes"] = head_out["feat_sizes"]

        if not self.training:
            result["preds"] = head_out["o2o_cls"]
        return result

    _cached_anchors: Optional[Tuple[Tuple[int, ...], torch.Tensor, torch.Tensor]] = None

    def _get_anchors(self, feat_sizes, device):
        key = tuple((h, w) for h, w in feat_sizes)
        if self._cached_anchors is not None and self._cached_anchors[0] == key:
            c, s = self._cached_anchors[1], self._cached_anchors[2]
            if c.device == device:
                return c, s
        centers, strides = make_anchor_grid(feat_sizes, list(self.strides), device)
        self._cached_anchors = (key, centers, strides)
        return centers, strides

    @torch.no_grad()
    def predict(self, x, img_metas=None, score_thr=0.25, max_det=300, **kwargs):
        """NMS-free inference — score threshold + top-k only."""
        self.eval()
        out = self.forward(x)
        anchor_centers, anchor_strides = self._get_anchors(
            out["feat_sizes"], x.device
        )
        return decode_batch_nms_free(
            out["o2o_cls"], out["o2o_reg"],
            anchor_centers, anchor_strides,
            img_hw=(x.shape[2], x.shape[3]),
            score_thr=score_thr, max_det=max_det,
        )

    def strip_o2m(self):
        """Remove one-to-many heads and loss for lean CPU/edge deployment."""
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


# ---------------------------------------------------------------------------
# FlashDet N/S/M/L/X  (scaled RepNeXt architecture)
# ---------------------------------------------------------------------------

@DETECTORS.register("FlashDet")
class FlashDet(nn.Module):
    """FlashDet — NMS-free object detector (RepNeXt-based).

    Unified architecture family based on PicoBackbone's reparameterizable
    multi-scale depthwise blocks, scaled via explicit channel/depth configs.

    Components (consistent across all sizes):
      - FlashBackbone: PicoBlock + StrideDown (RepNeXt reparameterization)
      - PicoNeck: LiteModule-based PAN-FPN (depthwise separable)
      - DW dual head: Depthwise-separable o2o + o2m detection heads

    Args:
        num_classes: Number of detection classes.
        size: Model size variant ("p", "n", "s", "m", "l", "x").
            "p" (Pico) returns a :class:`FlashDetPico`.
        stem_channels: Override backbone stem width.
        stage_depths: Override backbone per-stage block counts.
        neck_channels: Override PicoNeck output width.
        strides: Feature pyramid strides.
        total_epochs: Total training epochs (for ProgLoss scheduling).
    """

    def __new__(cls, num_classes=80, size="n", **kwargs):
        if size == "p":
            return FlashDetPico(num_classes=num_classes, **kwargs)
        return super().__new__(cls)

    def __init__(
        self,
        num_classes: int = 80,
        size: str = "n",
        stem_channels: Optional[int] = None,
        stage_depths: Optional[Tuple[int, ...]] = None,
        neck_channels: Optional[int] = None,
        strides: Tuple[int, ...] = (8, 16, 32),
        total_epochs: int = 100,
        prog_alpha_init: float = 1.0,
        prog_alpha_final: float = 0.0,
        o2m_topk: int = 10,
        o2o_topk: int = 7,
        box_weight: float = 7.5,
        cls_weight: float = 0.5,
        l1_weight: float = 1.5,
        **kwargs,
    ):
        if size == "p":
            return
        super().__init__()

        cfg = SIZE_CONFIGS.get(size, SIZE_CONFIGS["n"])
        _stem = stem_channels if stem_channels is not None else cfg["stem"]
        _depths = stage_depths if stage_depths is not None else cfg["depths"]
        _neck_ch = neck_channels if neck_channels is not None else cfg["neck_ch"]
        _neck_blocks = cfg.get("neck_blocks", 1)

        self.num_classes = num_classes
        self.size = size
        self.strides = strides
        self.total_epochs = total_epochs

        # Backbone: RepNeXt-style (PicoBlock + StrideDown + SpatialPool)
        self.backbone = FlashBackbone(
            stem_channels=_stem,
            stage_depths=_depths,
            use_sppf=True,
        )

        # Neck: PicoNeck PAN-FPN (LiteModule-based, depthwise)
        self.neck = PicoNeck(
            in_channels=self.backbone.out_channels,
            out_channels=_neck_ch,
            kernel_size=5,
            num_blocks=_neck_blocks,
            use_depthwise=True,
            activation="LeakyReLU",
        )

        # Head: Depthwise-separable dual detection head (o2o + o2m)
        self.head = E2EDualHead(num_classes, _neck_ch, num_levels=3)

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
        """Init neck + head only; backbone handles its own init."""
        for module in (self.neck, self.head):
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
        for heads in [self.head.o2o_heads, self.head.o2m_heads]:
            for h in heads:
                nn.init.constant_(h.cls_pred.bias, -4.595)
                nn.init.normal_(h.reg_pred.weight, std=0.001)
                nn.init.zeros_(h.reg_pred.bias)

    def forward(
        self,
        x: torch.Tensor,
        gt_meta: Optional[Dict] = None,
        epoch: int = 0,
        compute_loss: bool = False,
        return_features: bool = False,
        **kwargs,
    ) -> Dict:
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
            self._last_loss_states = loss_states

        if return_features or not self.training:
            result["o2o_cls"] = head_out["o2o_cls"]
            result["o2o_reg"] = head_out["o2o_reg"]
            result["feat_sizes"] = head_out["feat_sizes"]

        if not self.training:
            result["preds"] = head_out["o2o_cls"]

        return result

    _cached_anchors: Optional[Tuple[Tuple[int, ...], torch.Tensor, torch.Tensor]] = None

    def _get_anchors(self, feat_sizes, device):
        key = tuple((h, w) for h, w in feat_sizes)
        if self._cached_anchors is not None and self._cached_anchors[0] == key:
            c, s = self._cached_anchors[1], self._cached_anchors[2]
            if c.device == device:
                return c, s
        centers, strides = make_anchor_grid(feat_sizes, list(self.strides), device)
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
        """NMS-free inference — one-to-one head, no post-processing."""
        self.eval()
        out = self.forward(x)
        anchor_centers, anchor_strides = self._get_anchors(
            out["feat_sizes"], x.device
        )
        return decode_batch_nms_free(
            out["o2o_cls"], out["o2o_reg"],
            anchor_centers, anchor_strides,
            img_hw=(x.shape[2], x.shape[3]),
            score_thr=score_thr, max_det=max_det,
        )

    def strip_o2m(self):
        """Remove one-to-many heads and loss for lean CPU/edge deployment."""
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
