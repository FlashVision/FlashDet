"""
FlashDet detector.
Single-stage object detector with lightweight backbone and FPN.
"""

import copy
import os
import logging

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

from .backbone import ShuffleNetV2
from .neck import GhostPAN
from .head import FlashDetHead, SimpleConvHead

logger = logging.getLogger(__name__)

COCO_PRETRAINED_URLS = {
    ("1.0x", 96, 320): "https://sourceforge.net/projects/nanodet-plus.mirror/files/v1.0.0-alpha-1/nanodet-plus-m_320_checkpoint.ckpt/download",
    ("1.0x", 96, 416): "https://sourceforge.net/projects/nanodet-plus.mirror/files/v1.0.0-alpha-1/nanodet-plus-m_416_checkpoint.ckpt/download",
    ("1.5x", 128, 320): "https://sourceforge.net/projects/nanodet-plus.mirror/files/v1.0.0-alpha-1/nanodet-plus-m-1.5x_320_checkpoint.ckpt/download",
    ("1.5x", 128, 416): "https://sourceforge.net/projects/nanodet-plus.mirror/files/v1.0.0-alpha-1/nanodet-plus-m-1.5x_416_checkpoint.ckpt/download",
}

COCO_NUM_CLASSES = 80


class FlashDet(nn.Module):
    """
    FlashDet object detector.
    
    Ultra-lightweight detector with ShuffleNetV2 backbone.
    Matches official FlashDet implementation.
    
    Official Model Specs (for reference):
    - FlashDet-m (1.0x, fpn=96):      ~1.17M params, 2.3MB FP16, 1.2MB INT8
    - FlashDet-m-1.5x (1.5x, fpn=128): ~2.44M params, 4.7MB FP16, 2.3MB INT8
    - FlashDet-m-0.5x (0.5x, fpn=96):  ~0.49M params, ~0.9MB FP16 (ultra-lite)
    
    Args:
        num_classes: Number of detection classes.
        input_size: Input image size (width, height).
        backbone_size: Backbone variant ("0.5x", "1.0x", "1.5x").
        fpn_channels: FPN output channels.
        strides: Feature map strides.
        reg_max: Max value for distribution focal loss.
        pretrained: Whether to load pretrained backbone.
        use_aux_head: Whether to use auxiliary head for training.
    """
    
    # Output channels from ShuffleNetV2 stages 2, 3, 4
    # These must match the backbone's actual output channels
    BACKBONE_CHANNELS = {
        "0.5x": [48, 96, 192],      # ShuffleNetV2 0.5x: channels[2,3,4] = [48, 96, 192]
        "1.0x": [116, 232, 464],    # ShuffleNetV2 1.0x: channels[2,3,4] = [116, 232, 464]
        "1.5x": [176, 352, 704],    # ShuffleNetV2 1.5x: channels[2,3,4] = [176, 352, 704]
        "2.0x": [244, 488, 976],    # ShuffleNetV2 2.0x: channels[2,3,4] = [244, 488, 976]
    }
    
    def __init__(
        self,
        num_classes: int = 10,
        input_size: Tuple[int, int] = (320, 320),
        backbone_size: str = "0.5x",
        fpn_channels: int = 96,
        strides: List[int] = None,
        reg_max: int = 7,
        pretrained: bool = True,
        use_aux_head: bool = True,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.input_size = input_size
        self.strides = strides or [8, 16, 32, 64]
        self.use_aux_head = use_aux_head
        # Official FlashDet default: detach backbone from epoch 0 (always detached).
        # aux_fpn runs on detached backbone features so it never affects main head grads.
        self.detach_epoch = 0
        
        # Backbone — official FlashDet uses ReLU for ShuffleNetV2 so the
        # pretrained ImageNet weights (trained with ReLU) are applied correctly.
        self.backbone = ShuffleNetV2(
            model_size=backbone_size,
            out_stages=(2, 3, 4),
            pretrained=pretrained,
            activation="ReLU"
        )
        
        # FPN (Neck)
        in_channels = self.BACKBONE_CHANNELS[backbone_size]
        self.fpn = GhostPAN(
            in_channels=in_channels,
            out_channels=fpn_channels,
            kernel_size=5,
            num_extra_level=1,
            use_depthwise=True,
            activation="LeakyReLU"
        )
        
        # Detection head
        self.head = FlashDetHead(
            num_classes=num_classes,
            input_channel=fpn_channels,
            feat_channels=fpn_channels,
            stacked_convs=2,
            kernel_size=5,
            strides=strides,
            reg_max=reg_max,
            activation="LeakyReLU"
        )
        
        # Auxiliary head (AGM — Assign Guidance Module), training only.
        # Official: self.aux_fpn = deepcopy(self.fpn) feeds [fpn_feat, aux_fpn_feat]
        # (same-scale concatenation, 2×fpn_channels) into the aux head.
        # For lightweight backbones (0.5x), use fewer stacked convs to keep the
        # aux overhead proportional to the inference model size.
        if use_aux_head:
            self.aux_fpn = copy.deepcopy(self.fpn)
            aux_stacked = 2 if backbone_size == "0.5x" else 4
            self.aux_head = SimpleConvHead(
                num_classes=num_classes,
                input_channel=fpn_channels * 2,
                feat_channels=fpn_channels * 2,
                stacked_convs=aux_stacked,
                strides=strides,
                reg_max=reg_max,
                activation="LeakyReLU"
            )
    
    def forward(
        self,
        x: torch.Tensor,
        gt_meta: Dict = None,
        epoch: int = 0,
        compute_loss: bool = False,
        return_features: bool = False,
    ) -> Dict:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, 3, H, W].
            gt_meta: Ground truth metadata (for training).
            epoch: Current epoch (for aux head detachment).
            compute_loss: If True, compute loss even when not in training mode.
                         Used for validation with proper BatchNorm eval behavior.
            return_features: If True, include backbone and FPN features in
                output dict (used for Knowledge Distillation).
            
        Returns:
            In training (or compute_loss=True): Dict with 'loss' and 'loss_states'.
            In inference: Dict with 'preds'.
            When return_features=True: additionally includes 'fpn_features',
            'backbone_features', and 'preds'.
        """
        # Backbone features
        features = self.backbone(x)
        
        # FPN features
        fpn_features = self.fpn(features)
        
        # Detection head
        preds = self.head(fpn_features)
        
        # Compute loss if in training mode OR if explicitly requested (for validation)
        if (self.training or compute_loss) and gt_meta is not None:
            gt_meta["img"] = x
            
            # AGM auxiliary head — only during actual forward training (not validation).
            # Mirrors official FlashDet.forward_train:
            #   aux_fpn_feat = aux_fpn([f.detach() for f in backbone_feat])
            #   dual = cat([fpn_feat[i].detach(), aux_fpn_feat[i]])   when epoch >= detach_epoch
            #   dual = cat([fpn_feat[i],           aux_fpn_feat[i]])   otherwise
            aux_preds = None
            if self.training and self.use_aux_head and hasattr(self, "aux_fpn"):
                if epoch >= self.detach_epoch:
                    aux_fpn_feats = self.aux_fpn([f.detach() for f in features])
                    dual_feats = [
                        torch.cat([f.detach(), aux_f], dim=1)
                        for f, aux_f in zip(fpn_features, aux_fpn_feats)
                    ]
                else:
                    aux_fpn_feats = self.aux_fpn(features)
                    dual_feats = [
                        torch.cat([f, aux_f], dim=1)
                        for f, aux_f in zip(fpn_features, aux_fpn_feats)
                    ]
                aux_preds = self.aux_head(dual_feats)
            
            # Compute loss
            loss, loss_states = self.head.loss(preds, gt_meta, aux_preds)
            
            result = {
                "loss": loss,
                "loss_states": loss_states,
            }
            if return_features:
                result["preds"] = preds
                result["fpn_features"] = fpn_features
                result["backbone_features"] = features
            return result
        else:
            result = {"preds": preds}
            if return_features:
                result["fpn_features"] = fpn_features
                result["backbone_features"] = features
            return result
    
    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        img_metas: Optional[Dict] = None,
        score_thr: float = 0.05,
        nms_thr: float = 0.6
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Run inference.
        
        Args:
            x: Input tensor [B, 3, H, W].
            img_metas: Image metadata (height, width).
            score_thr: Score threshold.
            nms_thr: NMS threshold.
            
        Returns:
            List of (det_bboxes, det_labels) per image.
        """
        self.eval()
        
        if img_metas is None:
            img_metas = {"img": x}
        else:
            img_metas["img"] = x
        
        output = self.forward(x)
        preds = output["preds"]
        
        results = self.head.get_bboxes(preds, img_metas, score_thr, nms_thr)
        return results
    
    def get_model_info(self) -> Dict:
        """Get model information (inference and training param counts)."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Inference params exclude aux_fpn and aux_head (training-only)
        aux_params = 0
        for name in ("aux_fpn", "aux_head"):
            mod = getattr(self, name, None)
            if mod is not None:
                aux_params += sum(p.numel() for p in mod.parameters())
        inference_params = total_params - aux_params

        return {
            "name": "FlashDet",
            "num_classes": self.num_classes,
            "input_size": self.input_size,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "inference_params": inference_params,
            "params_mb": total_params * 4 / (1024 ** 2),
            "inference_params_mb": inference_params * 4 / (1024 ** 2),
            "inference_fp16_mb": inference_params * 2 / (1024 ** 2),
        }


def _map_official_key(key: str) -> str:
    """Map an official NanoDet checkpoint key to our model's naming convention.

    Differences:
      - Official uses ``dwnorm`` / ``pwnorm`` for DepthwiseConvModule BN layers;
        our implementation uses ``bn1`` / ``bn2``.
    """
    key = key.replace(".dwnorm.", ".bn1.")
    key = key.replace(".pwnorm.", ".bn2.")
    return key


def _download_checkpoint(url: str, cache_dir: str = "pretrained") -> str:
    """Download a checkpoint file if it is not already cached."""
    os.makedirs(cache_dir, exist_ok=True)
    # Derive a human-readable filename from the URL
    basename = url.split("/")[-2]  # e.g. "nanodet-plus-m_416_checkpoint.ckpt"
    local_path = os.path.join(cache_dir, basename)
    if os.path.isfile(local_path):
        logger.info(f"Using cached COCO checkpoint: {local_path}")
        return local_path
    logger.info(f"Downloading COCO pretrained checkpoint to {local_path} ...")
    torch.hub.download_url_to_file(url, local_path, progress=True)
    return local_path


def load_coco_pretrained(
    model: "FlashDet",
    backbone_size: str = "1.0x",
    fpn_channels: int = 96,
    input_size: int = 416,
    checkpoint_path: str = None,
    use_ema: bool = True,
) -> Dict[str, list]:
    """Load official FlashDet COCO-pretrained weights into *model*.

    The official checkpoint is trained on COCO (80 classes).  Backbone, FPN
    (GhostPAN) and head convolution layers are loaded directly.  For the
    final ``gfl_cls`` prediction layer (which has a different number of output
    channels because of a different class count), only the **regression**
    channels are transferred — classification channels are left at their
    random-init values so the model can be fine-tuned on a custom dataset.

    After loading, the ``aux_fpn`` module (if present) is re-created as a
    deep-copy of the loaded ``fpn`` so it also benefits from pretrained
    features.

    Args:
        model: A ``FlashDet`` instance (already constructed).
        backbone_size: One of ``"1.0x"``, ``"1.5x"`` — must match the model.
        fpn_channels: FPN output channels — must match the model.
        input_size: 320 or 416 — used to pick the right checkpoint URL.
        checkpoint_path: If given, load from this local file instead of
            downloading from SourceForge.
        use_ema: If ``True`` (default), load the EMA ("avg_model") weights
            which are higher quality than the raw training weights.

    Returns:
        A dict ``{"loaded": [...], "skipped": [...]}`` listing which keys
        were loaded or skipped, for logging purposes.
    """
    # ---- resolve the checkpoint file ----
    if checkpoint_path and os.path.isfile(checkpoint_path):
        ckpt_path = checkpoint_path
    else:
        lookup_key = (backbone_size, fpn_channels, input_size)
        url = COCO_PRETRAINED_URLS.get(lookup_key)
        if url is None:
            avail = list(COCO_PRETRAINED_URLS.keys())
            raise ValueError(
                f"No COCO pretrained checkpoint for {lookup_key}. "
                f"Available configs: {avail}"
            )
        ckpt_path = _download_checkpoint(url)

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    prefix = "avg_model." if use_ema else "model."
    src_sd = {
        k[len(prefix):]: v
        for k, v in raw["state_dict"].items()
        if k.startswith(prefix)
    }
    if not src_sd:
        raise RuntimeError(
            f"Checkpoint has no keys with prefix '{prefix}'. "
            f"Available top-level prefixes: "
            f"{set(k.split('.')[0] for k in raw['state_dict'])}"
        )

    src_sd = {_map_official_key(k): v for k, v in src_sd.items()}

    our_sd = model.state_dict()
    num_classes = model.num_classes
    reg_channels = 4 * (model.head.reg_max + 1)  # e.g. 4*8 = 32

    loaded_keys = []
    skipped_keys = []

    new_sd = {}
    for key, our_tensor in our_sd.items():
        # Skip aux_fpn / aux_head — we re-init them afterwards
        if key.startswith("aux_fpn.") or key.startswith("aux_head."):
            skipped_keys.append(key)
            continue

        src_tensor = src_sd.get(key)
        if src_tensor is None:
            skipped_keys.append(key)
            continue

        # Direct load if shapes match exactly
        if src_tensor.shape == our_tensor.shape:
            new_sd[key] = src_tensor
            loaded_keys.append(key)
            continue

        # gfl_cls weight/bias: partial load (regression channels only)
        if "gfl_cls" in key:
            # Official layout: [COCO_cls + reg, ...]
            # Our layout:      [num_cls   + reg, ...]
            coco_cls = COCO_NUM_CLASSES
            if src_tensor.shape[0] == coco_cls + reg_channels:
                patched = our_tensor.clone()
                # Copy regression weights (last reg_channels)
                patched[num_classes:num_classes + reg_channels] = src_tensor[coco_cls:]
                new_sd[key] = patched
                loaded_keys.append(f"{key} (reg-only)")
                continue

        skipped_keys.append(key)

    # Apply the loaded weights
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    # `missing` will include all keys we didn't put in new_sd — that's expected

    # Re-create aux_fpn from the now-loaded fpn for better initialisation
    if hasattr(model, "aux_fpn"):
        model.aux_fpn = copy.deepcopy(model.fpn)
        logger.info("aux_fpn re-initialised as deep-copy of loaded fpn")

    logger.info(
        f"COCO pretrained weights loaded: {len(loaded_keys)} keys loaded, "
        f"{len(skipped_keys)} keys skipped (aux/cls/missing)"
    )
    return {"loaded": loaded_keys, "skipped": skipped_keys}


def build_model(config) -> FlashDet:
    """
    Build model from config.
    
    Args:
        config: Model configuration.
        
    Returns:
        FlashDet model.
    """
    return FlashDet(
        num_classes=config.model.num_classes,
        input_size=config.model.input_size,
        backbone_size=config.model.backbone_size,
        fpn_channels=config.model.fpn_out_channels,
        strides=getattr(config.model, "strides", None),
        reg_max=getattr(config.model, "reg_max", 7),
        pretrained=config.model.backbone_pretrained,
        use_aux_head=getattr(config.model, "use_aux_head", True),
    )
