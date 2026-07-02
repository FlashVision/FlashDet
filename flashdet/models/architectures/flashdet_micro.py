"""FlashDet-Micro — Sub-0.5MB Object Detector with Unified MicroBlock.

Architecture using the same MicroBlock across backbone, neck, and head.
Target: < 262K inference params = < 0.5MB FP16 weight file.

Key innovations:
  - MicroBlock: Reparameterizable DW conv + ECA attention + residual
  - Same block used in backbone feature extraction, neck fusion, and head
  - Weight-shared head across all FPN levels (3x param savings in head)
  - ECA channel attention for accuracy boost with negligible cost

Architecture:
  - MicroBackbone: Stem → 3 stages of MicroDown + MicroBlock stacks
  - MicroNeck: PAN-FPN with MicroBlock fusion blocks
  - MicroDualHead: Weight-shared dual head using MicroBlock

Model Stats (default config):
  - Inference params: ~234K (~0.45 MB FP16)
  - Deployed params:  ~224K (~0.43 MB FP16, after reparameterization)
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashdet.registry import DETECTORS
from flashdet.models.layers.micro_block import MicroBlock, MicroDown
from flashdet.models.layers.conv_module import ConvModule
from flashdet.losses.e2e_loss import E2EDetectionLoss
from flashdet.utils.bbox import make_anchor_grid, decode_batch_nms_free

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MicroBackbone — backbone built entirely from MicroBlock + MicroDown
# ---------------------------------------------------------------------------

class MicroBackbone(nn.Module):
    """Lightweight backbone using unified MicroBlock at every stage.

    +----------+
    |  Stem    |  Conv3x3 stride-2 → BN → Act
    | MaxPool  |  stride-2
    +----------+  stride 4, stem_ch
    | Stage 0  |  MicroDown(stem→48) + N× MicroBlock(48)
    +----------+  stride 8,  48 ch
    | Stage 1  |  MicroDown(48→96)  + N× MicroBlock(96)
    +----------+  stride 16, 96 ch
    | Stage 2  |  MicroDown(96→160) + N× MicroBlock(160)
    +----------+  stride 32, 160 ch
    """

    def __init__(
        self,
        stem_channels: int = 24,
        stage_channels: Tuple[int, ...] = (48, 96, 160),
        stage_depths: Tuple[int, ...] = (2, 3, 2),
        activation: str = "LeakyReLU",
    ):
        super().__init__()
        self.out_channels = list(stage_channels)

        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.pool = nn.MaxPool2d(3, 2, 1)

        in_ch = stem_channels
        self.stages = nn.ModuleList()
        for ch, depth in zip(stage_channels, stage_depths):
            layers: List[nn.Module] = [MicroDown(in_ch, ch, activation)]
            for _ in range(depth):
                layers.append(MicroBlock(ch, activation))
            self.stages.append(nn.Sequential(*layers))
            in_ch = ch

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.pool(self.stem(x))
        outputs: List[torch.Tensor] = []
        for stage in self.stages:
            x = stage(x)
            outputs.append(x)
        return outputs

    @torch.no_grad()
    def fuse(self):
        """Fuse all MicroBlock reparameterizable branches."""
        for m in self.modules():
            if isinstance(m, MicroBlock):
                m.fuse()
        logger.info("MicroBackbone: fused all reparameterizable branches")
        return self


# ---------------------------------------------------------------------------
# MicroNeck — PAN-FPN using MicroBlock for fusion
# ---------------------------------------------------------------------------

class MicroNeck(nn.Module):
    """Lightweight PAN-FPN neck using MicroBlock for feature fusion.

    Same top-down + bottom-up structure as PicoNeck, but replaces
    LiteBlock/LiteModule with the unified MicroBlock.
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int = 64,
        activation: str = "LeakyReLU",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Lateral 1x1 convs to unify channel widths
        self.reduce_layers = nn.ModuleList([
            ConvModule(ch, out_channels, 1, activation=activation)
            for ch in in_channels
        ])

        n_levels = len(in_channels)

        # Top-down: upsample + cat → project 1x1 → MicroBlock
        self.td_projects = nn.ModuleList()
        self.td_blocks = nn.ModuleList()
        for _ in range(n_levels - 1):
            self.td_projects.append(
                ConvModule(out_channels * 2, out_channels, 1, activation=activation),
            )
            self.td_blocks.append(MicroBlock(out_channels, activation))

        # Bottom-up: DW stride-2 downsample + cat → project 1x1 → MicroBlock
        self.bu_downs = nn.ModuleList()
        self.bu_projects = nn.ModuleList()
        self.bu_blocks = nn.ModuleList()
        for _ in range(n_levels - 1):
            act_mod: nn.Module
            if activation == "LeakyReLU":
                act_mod = nn.LeakyReLU(0.1, inplace=True)
            else:
                act_mod = nn.SiLU(inplace=True)
            self.bu_downs.append(nn.Sequential(
                nn.Conv2d(
                    out_channels, out_channels, 3, 2, 1,
                    groups=out_channels, bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                act_mod,
            ))
            self.bu_projects.append(
                ConvModule(out_channels * 2, out_channels, 1, activation=activation),
            )
            self.bu_blocks.append(MicroBlock(out_channels, activation))

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        assert len(inputs) == len(self.in_channels)

        # Reduce to uniform channels
        inputs = [red(x) for red, x in zip(self.reduce_layers, inputs)]

        # Top-down path
        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = inputs[idx - 1]
            up = F.interpolate(
                feat_high, size=feat_low.shape[2:],
                mode="bilinear", align_corners=False,
            )
            td_idx = len(self.in_channels) - 1 - idx
            fused = self.td_projects[td_idx](torch.cat([up, feat_low], dim=1))
            fused = self.td_blocks[td_idx](fused)
            inner_outs.insert(0, fused)

        # Bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            down = self.bu_downs[idx](outs[-1])
            fused = self.bu_projects[idx](
                torch.cat([down, inner_outs[idx + 1]], dim=1),
            )
            fused = self.bu_blocks[idx](fused)
            outs.append(fused)

        return outs

    @torch.no_grad()
    def fuse(self):
        for m in self.modules():
            if isinstance(m, MicroBlock):
                m.fuse()
        return self


# ---------------------------------------------------------------------------
# MicroDetHead / MicroDualHead — weight-shared head using MicroBlock
# ---------------------------------------------------------------------------

class MicroDetHead(nn.Module):
    """Detection head using MicroBlock — shared across all FPN levels.

    cls: MicroBlock → Conv 1x1 → [B, num_classes, H, W]
    reg: MicroBlock → Conv 1x1 → [B, 4, H, W]
    """

    def __init__(
        self, num_classes: int, in_channels: int, activation: str = "LeakyReLU",
    ):
        super().__init__()
        self.cls_block = MicroBlock(in_channels, activation)
        self.reg_block = MicroBlock(in_channels, activation)
        self.cls_pred = nn.Conv2d(in_channels, num_classes, 1)
        self.reg_pred = nn.Conv2d(in_channels, 4, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cls_pred(self.cls_block(x)), self.reg_pred(self.reg_block(x))


class MicroDualHead(nn.Module):
    """Dual detection head with weight sharing across FPN levels.

    One-to-One head (inference) + One-to-Many head (training only).
    A single MicroDetHead instance is applied to all FPN levels,
    saving ~2/3 of head parameters compared to per-level heads.
    """

    def __init__(
        self, num_classes: int, in_channels: int, activation: str = "LeakyReLU",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.o2o_head = MicroDetHead(num_classes, in_channels, activation)
        self.o2m_head = MicroDetHead(num_classes, in_channels, activation)

    def forward(
        self, features: List[torch.Tensor], training: bool = True,
    ) -> dict:
        o2o_cls_list: List[torch.Tensor] = []
        o2o_reg_list: List[torch.Tensor] = []
        feat_sizes: List[Tuple[int, int]] = []

        for feat in features:
            cls, reg = self.o2o_head(feat)
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
            o2m_cls_list: List[torch.Tensor] = []
            o2m_reg_list: List[torch.Tensor] = []
            for feat in features:
                cls, reg = self.o2m_head(feat)
                B, _, H, W = cls.shape
                o2m_cls_list.append(cls.permute(0, 2, 3, 1).reshape(B, H * W, -1))
                o2m_reg_list.append(reg.permute(0, 2, 3, 1).reshape(B, H * W, 4))
            result["o2m_cls"] = torch.cat(o2m_cls_list, dim=1)
            result["o2m_reg"] = torch.cat(o2m_reg_list, dim=1)

        return result


# ---------------------------------------------------------------------------
# FlashDetMicro — full detector
# ---------------------------------------------------------------------------

@DETECTORS.register("FlashDetMicro")
class FlashDetMicro(nn.Module):
    """FlashDet-Micro — Sub-0.5MB NMS-free object detector.

    Uses a single unified MicroBlock across backbone, neck, and head.
    Weight-shared head across all 3 FPN levels for maximum efficiency.

    Architecture:
      - MicroBackbone: stem(24) → stages [48, 96, 160] at strides [8, 16, 32]
      - MicroNeck: PAN-FPN with 64-ch MicroBlock fusion
      - MicroDualHead: shared o2o + o2m using MicroBlock
      - Same E2EDetectionLoss (STAL + ProgLoss) as larger FlashDet models

    Target: < 262K inference params = < 0.5MB FP16 weight file.
    """

    def __init__(
        self,
        num_classes: int = 80,
        strides: Tuple[int, ...] = (8, 16, 32),
        total_epochs: int = 100,
        stem_channels: int = 24,
        stage_channels: Tuple[int, ...] = (48, 96, 160),
        stage_depths: Tuple[int, ...] = (2, 3, 2),
        neck_channels: int = 64,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.size = "u"
        self.strides = strides
        self.total_epochs = total_epochs

        self.backbone = MicroBackbone(
            stem_channels=stem_channels,
            stage_channels=stage_channels,
            stage_depths=stage_depths,
        )

        self.neck = MicroNeck(
            in_channels=self.backbone.out_channels,
            out_channels=neck_channels,
        )

        self.head = MicroDualHead(num_classes, neck_channels)

        self.loss_fn = E2EDetectionLoss(
            num_classes=num_classes,
            strides=strides,
            alpha_init=kwargs.get("prog_alpha_init", 0.8),
            alpha_final=kwargs.get("prog_alpha_final", 0.2),
            o2m_topk=kwargs.get("o2m_topk", 10),
            o2o_topk=kwargs.get("o2o_topk", 7),
            box_weight=kwargs.get("box_weight", 7.5),
            cls_weight=kwargs.get("cls_weight", 1.0),
            l1_weight=kwargs.get("l1_weight", 1.5),
        )

        self._init_weights()

    def _init_weights(self):
        for module in (self.backbone, self.neck, self.head):
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="leaky_relu",
                    )
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
        for det_head in (self.head.o2o_head, self.head.o2m_head):
            nn.init.constant_(det_head.cls_pred.bias, -4.595)
            nn.init.normal_(det_head.reg_pred.weight, std=0.001)
            nn.init.zeros_(det_head.reg_pred.bias)

    # ------------------------------------------------------------------ fwd
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
            gt_bboxes_list = [
                torch.as_tensor(b, dtype=torch.float32, device=device).reshape(-1, 4)
                for b in gt_meta["gt_bboxes"]
            ]
            gt_labels_list = [
                torch.as_tensor(l, dtype=torch.long, device=device).reshape(-1)
                for l in gt_meta["gt_labels"]
            ]

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

    # -------------------------------------------------------------- anchors
    _cached_anchors: Optional[
        Tuple[Tuple[Tuple[int, int], ...], torch.Tensor, torch.Tensor]
    ] = None

    def _get_anchors(self, feat_sizes, device):
        key = tuple((h, w) for h, w in feat_sizes)
        if self._cached_anchors is not None and self._cached_anchors[0] == key:
            c, s = self._cached_anchors[1], self._cached_anchors[2]
            if c.device == device:
                return c, s
        centers, strides = make_anchor_grid(feat_sizes, list(self.strides), device)
        self._cached_anchors = (key, centers, strides)
        return centers, strides

    # ------------------------------------------------------------ inference
    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        img_metas: Optional[Dict] = None,
        score_thr: float = 0.25,
        max_det: int = 300,
        **kwargs,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """NMS-free inference — score threshold + top-k only."""
        self.eval()
        out = self.forward(x)
        anchor_centers, anchor_strides = self._get_anchors(
            out["feat_sizes"], x.device,
        )
        return decode_batch_nms_free(
            out["o2o_cls"],
            out["o2o_reg"],
            anchor_centers,
            anchor_strides,
            img_hw=(x.shape[2], x.shape[3]),
            score_thr=score_thr,
            max_det=max_det,
        )

    # ------------------------------------------------------------ deploy
    def strip_o2m(self):
        """Remove one-to-many head and loss for lean inference."""
        del self.head.o2m_head
        del self.loss_fn
        self.head.o2m_head = None  # type: ignore[assignment]
        self.loss_fn = None  # type: ignore[assignment]
        logger.info("Stripped o2m head + loss for NMS-free inference-only mode")
        return self

    def fuse(self):
        """Fuse all reparameterizable branches for deployment."""
        self.backbone.fuse()
        self.neck.fuse()
        for m in self.head.modules():
            if isinstance(m, MicroBlock):
                m.fuse()
        logger.info("FlashDetMicro: fused all MicroBlock branches")
        return self

    # ------------------------------------------------------------ info
    def get_model_info(self) -> Dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        o2m_params = sum(p.numel() for p in self.head.o2m_head.parameters())
        inference_params = total - o2m_params
        return {
            "name": "FlashDet-U",
            "num_classes": self.num_classes,
            "size": "u",
            "total_params": total,
            "trainable_params": trainable,
            "inference_params": inference_params,
            "params_mb": total * 4 / (1024 ** 2),
            "inference_params_mb": inference_params * 4 / (1024 ** 2),
            "inference_fp16_mb": inference_params * 2 / (1024 ** 2),
        }
