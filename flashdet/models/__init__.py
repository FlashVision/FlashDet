# Backbone
from .backbone import (
    ShuffleNetV2,
    ResNetBackbone,
    ResNetMultiScaleBackbone,
    YOLOv9Backbone,
    YOLOv10Backbone,
    YOLOv11Backbone,
)

# Neck (FPN)
from .neck import (
    GhostPAN,
    GhostBottleneck,
    GhostModule,
    HybridEncoder,
    AIFI,
    YOLOv9Neck,
    YOLOv10Neck,
    YOLOv11Neck,
)

# Head
from .head import (
    FlashDetHead,
    SimpleConvHead,
    Integral,
    OBBHead,
    YOLODetectionHead,
    DualHeadOne2One,
    DualHeadOne2Many,
    PGIAuxBranch,
    DETRHead,
    RTDETRDecoder,
    E2EDetHead,
    E2EDualHead,
)

# Shared layers / building blocks
from .layers import (
    ConvBNSiLU,
    DownSample,
    RepConv,
    RepVGGBlock,
    C2f,
    C3k2,
    CSPRepLayer,
    ELAN,
    GELAN,
    Bottleneck,
    SPPF,
    PSA,
    C2PSA,
    SCDown,
)

# Transformer
from .transformer import PositionalEncoding2D, DETRTransformer

# Assignment
from .assignment import DynamicSoftLabelAssigner, AssignResult, HungarianMatcher, STALAssigner

# Detector
from .detector import FlashDet, build_model, load_coco_pretrained

# Architectures (full detectors)
from .architectures import DETR, RTDETR, YOLOv9, YOLOv10, YOLOv11, GroundingDINO

# LoRA / QLoRA
from .lora import (
    apply_lora, apply_qlora, merge_lora_weights, get_lora_state_dict,
    LORA_VARIANTS, get_variant_description, get_ortho_regularization_loss,
    get_lora_plus_param_groups,
)

__all__ = [
    # Backbone
    "ShuffleNetV2",
    "ResNetBackbone",
    "ResNetMultiScaleBackbone",
    "YOLOv9Backbone",
    "YOLOv10Backbone",
    "YOLOv11Backbone",
    # Neck
    "GhostPAN",
    "GhostBottleneck",
    "GhostModule",
    "HybridEncoder",
    "AIFI",
    "YOLOv9Neck",
    "YOLOv10Neck",
    "YOLOv11Neck",
    # Head
    "FlashDetHead",
    "SimpleConvHead",
    "Integral",
    "OBBHead",
    "YOLODetectionHead",
    "DualHeadOne2One",
    "DualHeadOne2Many",
    "PGIAuxBranch",
    "DETRHead",
    "RTDETRDecoder",
    "E2EDetHead",
    "E2EDualHead",
    # Layers
    "ConvBNSiLU",
    "DownSample",
    "RepConv",
    "RepVGGBlock",
    "C2f",
    "C3k2",
    "CSPRepLayer",
    "ELAN",
    "GELAN",
    "Bottleneck",
    "SPPF",
    "PSA",
    "C2PSA",
    "SCDown",
    # Transformer
    "PositionalEncoding2D",
    "DETRTransformer",
    # Assignment
    "DynamicSoftLabelAssigner",
    "AssignResult",
    "HungarianMatcher",
    "STALAssigner",
    # Detector
    "FlashDet",
    "build_model",
    "load_coco_pretrained",
    # Architectures
    "DETR",
    "RTDETR",
    "YOLOv9",
    "YOLOv10",
    "YOLOv11",
    "GroundingDINO",
    # LoRA / QLoRA
    "apply_lora",
    "apply_qlora",
    "merge_lora_weights",
    "get_lora_state_dict",
    "LORA_VARIANTS",
    "get_variant_description",
    "get_ortho_regularization_loss",
    "get_lora_plus_param_groups",
]
