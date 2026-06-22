"""
Configuration for FlashDet Model.

All paths and class counts are left empty/generic by default so nothing
is hardcoded to a specific dataset.  ``train.py`` reads them from the
annotation JSON automatically — these are just fallbacks.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class DataConfig:
    """Dataset paths — point to your COCO-format data directory.

    Override with your own paths or download a dataset via:
        flashdet download --dataset coco2017
    """
    train_images: str = ""
    train_annotations: str = ""
    val_images: str = ""
    val_annotations: str = ""
    test_images: str = ""
    test_annotations: str = ""
    num_workers: int = 4


@dataclass
class ModelConfig:
    """Model architecture configuration.

    Supported architectures: flashdet, detr, rt-detr, yolov9, yolov10, yolov11, grounding-dino.

    Official FlashDet model specifications:
    - FlashDet-m:      backbone=1.0x, fpn=96,  ~1.17M params, 2.3MB FP16
    - FlashDet-m-1.5x: backbone=1.5x, fpn=128, ~2.44M params, 4.7MB FP16
    - FlashDet-m-0.5x: backbone=0.5x, fpn=96,  ~0.49M params, ~0.9MB FP16 (ultra-lite)
    """
    name: str = "FlashDet"
    architecture: str = "flashdet"
    num_classes: int = 80
    input_size: Tuple[int, int] = (320, 320)
    
    # Backbone: 1.0x for FlashDet-m, 1.5x for m-1.5x, 0.5x for m-0.5x
    backbone: str = "ShuffleNetV2"
    backbone_size: str = "1.0x"  # Default matches official FlashDet-m
    backbone_pretrained: bool = True
    
    # FPN (96 for m, 128 for m-1.5x)
    fpn_in_channels: List[int] = field(default_factory=lambda: [116, 232, 464])
    fpn_out_channels: int = 96
    
    # Head
    head_channels: int = 96
    stacked_convs: int = 2
    strides: List[int] = field(default_factory=lambda: [8, 16, 32, 64])
    reg_max: int = 7
    
    # Loss weights
    loss_qfl_weight: float = 1.0
    loss_dfl_weight: float = 0.25
    loss_bbox_weight: float = 2.0


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    epochs: int = 300
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    grad_clip: float = 35.0
    val_interval: int = 5
    save_dir: str = "workspace/flashdet_output"
    resume: Optional[str] = None
    patience: int = 50
    pretrained_coco: bool = False

    # --- Performance ---
    amp: bool = False
    multi_gpu: bool = False
    grad_accum: int = 1

    # --- torchtune-inspired memory & performance optimizations ---
    enable_activation_checkpointing: bool = False
    enable_activation_offloading: bool = False
    optimizer_in_bwd: bool = False
    use_8bit_optimizer: bool = False
    compile_model: bool = False
    chunked_cross_entropy: bool = False
    ce_chunk_size: int = 1024

    # --- LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning ---
    use_lora: bool = False
    lora_variant: str = "standard"
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["backbone", "fpn"])

    # --- QLoRA (Quantized LoRA) ---
    use_qlora: bool = False
    qlora_quant_dtype: str = "int8"

    # --- Knowledge Distillation (torchtune-style) ---
    use_kd: bool = False
    kd_teacher_checkpoint: Optional[str] = None
    kd_teacher_model_size: str = "m-1.5x"
    kd_temperature: float = 4.0
    kd_logit_weight: float = 1.0
    kd_feature_weight: float = 0.5
    kd_hard_loss_weight: float = 1.0

    # --- Augmentations ---
    mosaic: bool = False
    mixup: bool = False
    copy_paste: bool = False


@dataclass
class AugmentConfig:
    """Data augmentation configuration."""
    scale: Tuple[float, float] = (0.6, 1.4)
    stretch: Tuple[Tuple[float, float], Tuple[float, float]] = ((0.8, 1.2), (0.8, 1.2))
    flip_prob: float = 0.5
    brightness: float = 0.2
    contrast: Tuple[float, float] = (0.6, 1.4)
    saturation: Tuple[float, float] = (0.5, 1.2)
    normalize_mean: List[float] = field(default_factory=lambda: [123.675, 116.28, 103.53])
    normalize_std: List[float] = field(default_factory=lambda: [58.395, 57.12, 57.375])


@dataclass
class Config:
    """Top-level configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)

    class_names: List[str] = field(default_factory=list)


MODEL_SIZE_MAP = {
    # Legacy NanoDet-era sizes (backward compat)
    "m-0.5x": ("0.5x", [58, 116, 232], 96),
    "m": ("1.0x", [116, 232, 464], 96),
    "m-1.5x": ("1.5x", [176, 352, 704], 128),
}

# YOLO26-based FlashDet sizes
YOLO26_SIZE_MAP = {
    "n": "n",
    "s": "s",
    "m": "m",
    "l": "l",
    "x": "x",
}


def get_config(
    model_size: str = "n",
    input_size: int = 320,
    num_classes: int = 80,
    **overrides,
) -> Config:
    """Return configuration for a given model size.

    Args:
        model_size: YOLO26-based: "n", "s", "m", "l", "x".
            Legacy NanoDet: "m-0.5x", "m-1.5x" (backward compat).
        input_size: Input image dimension (square).
        num_classes: Number of detection classes.
        **overrides: Additional overrides applied to the Config.
    """
    cfg = Config()

    # YOLO26-based FlashDet size
    if model_size in YOLO26_SIZE_MAP:
        cfg.model.size = YOLO26_SIZE_MAP[model_size]
    elif model_size in MODEL_SIZE_MAP:
        backbone_size, fpn_in, fpn_out = MODEL_SIZE_MAP[model_size]
        cfg.model.backbone_size = backbone_size
        cfg.model.fpn_in_channels = fpn_in
        cfg.model.fpn_out_channels = fpn_out
        cfg.model.head_channels = fpn_out

    cfg.model.input_size = (input_size, input_size)
    cfg.model.num_classes = num_classes

    for key, value in overrides.items():
        parts = key.split(".")
        obj = cfg
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)

    return cfg


def load_yaml_config(yaml_path: str) -> Config:
    """Load configuration from a YAML file.

    YAML structure mirrors the Config dataclass hierarchy:
        model:
          backbone_size: "1.0x"
          num_classes: 10
          input_size: [320, 320]
        data:
          train_images: data/my_dataset/train
        train:
          epochs: 100
    """
    import yaml

    with open(yaml_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cfg = Config()

    if "model" in raw:
        for key, value in raw["model"].items():
            if key == "input_size" and isinstance(value, list):
                value = tuple(value)
            if hasattr(cfg.model, key):
                setattr(cfg.model, key, value)

    if "data" in raw:
        for key, value in raw["data"].items():
            if hasattr(cfg.data, key):
                setattr(cfg.data, key, value)

    if "train" in raw:
        for key, value in raw["train"].items():
            if hasattr(cfg.train, key):
                setattr(cfg.train, key, value)

    if "augment" in raw:
        for key, value in raw["augment"].items():
            if key in ("scale", "contrast", "saturation") and isinstance(value, list):
                value = tuple(value)
            if hasattr(cfg.augment, key):
                setattr(cfg.augment, key, value)

    # Derive fpn_in_channels from backbone_size if not explicitly set
    if "model" in raw and "fpn_in_channels" not in raw["model"]:
        bs = cfg.model.backbone_size
        if bs in ("0.5x", "1.0x", "1.5x"):
            size_key = f"m-{bs}" if bs != "1.0x" else "m"
            if bs == "0.5x":
                size_key = "m-0.5x"
            elif bs == "1.5x":
                size_key = "m-1.5x"
            if size_key in MODEL_SIZE_MAP:
                _, fpn_in, _ = MODEL_SIZE_MAP[size_key]
                cfg.model.fpn_in_channels = fpn_in

    return cfg
