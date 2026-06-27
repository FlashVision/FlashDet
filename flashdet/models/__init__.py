# Backbone
from .backbone import (
    LiteBackbone,
    PicoBackbone,
    FlashBackbone,
    YOLOv8Backbone,
    YOLOv9Backbone,
    YOLOv10Backbone,
    YOLOv11Backbone,
    YOLOXBackbone,
)

# Neck (FPN)
from .neck import (
    PicoNeck,
    LiteBlock,
    LiteModule,
    YOLOv8Neck,
    YOLOv9Neck,
    YOLOv10Neck,
    YOLOv11Neck,
    YOLOXNeck,
)

# Head
from .head import (
    E2EDetHead,
    E2EDualHead,
    YOLODetectionHead,
    DualHeadOne2One,
    DualHeadOne2Many,
    PGIAuxBranch,
    YOLOXHead,
)

# Shared layers / building blocks
from .layers import (
    ConvBlock,
    ConvModule,
    DepthwiseConvModule,
    SpatialPool,
    PicoBlock,
    StrideDown,
    MultiScaleConv,
    FusedDWConv,
    # YOLO blocks
    ConvBNSiLU,
    Bottleneck,
    C2f,
    SCDown,
    PSA,
    DownSample,
    GELAN,
    RepConv,
)

# Assignment
from .assignment import STALAssigner

# Detector
from .detector import FlashDet, build_model
from .architectures import FlashDetPico, SIZE_CONFIGS, YOLOv8, YOLOv9, YOLOv10, YOLOv11, YOLOX

# LoRA / QLoRA
from .lora import (
    apply_lora, apply_qlora, merge_lora_weights, get_lora_state_dict,
    LORA_VARIANTS, get_variant_description, get_ortho_regularization_loss,
    get_lora_plus_param_groups,
)

__all__ = [
    # Backbone
    "LiteBackbone",
    "PicoBackbone",
    "FlashBackbone",
    "YOLOv8Backbone",
    "YOLOv9Backbone",
    "YOLOv10Backbone",
    "YOLOv11Backbone",
    "YOLOXBackbone",
    # Neck
    "PicoNeck",
    "LiteBlock",
    "LiteModule",
    "YOLOv8Neck",
    "YOLOv9Neck",
    "YOLOv10Neck",
    "YOLOv11Neck",
    "YOLOXNeck",
    # Head
    "E2EDetHead",
    "E2EDualHead",
    "YOLODetectionHead",
    "DualHeadOne2One",
    "DualHeadOne2Many",
    "PGIAuxBranch",
    "YOLOXHead",
    # Layers
    "ConvBlock",
    "ConvModule",
    "DepthwiseConvModule",
    "SpatialPool",
    "PicoBlock",
    "StrideDown",
    "MultiScaleConv",
    "FusedDWConv",
    "ConvBNSiLU",
    "Bottleneck",
    "C2f",
    "SCDown",
    "PSA",
    "DownSample",
    "GELAN",
    "RepConv",
    # Assignment
    "STALAssigner",
    # Detector
    "FlashDet",
    "FlashDetPico",
    "SIZE_CONFIGS",
    "YOLOv8",
    "YOLOv9",
    "YOLOv10",
    "YOLOv11",
    "YOLOX",
    "build_model",
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
