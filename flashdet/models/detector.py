"""
Model builder — registry-based multi-architecture support.
"""

import logging

from flashdet.models.architectures.flashdet import FlashDet
import flashdet.models.architectures  # trigger all @DETECTORS registrations
from flashdet.registry import DETECTORS

logger = logging.getLogger(__name__)


def build_model(config, architecture: str = None):
    """Build a detection model from config using the DETECTORS registry.

    Supports: flashdet, yolov8, yolov9, yolov10, yolov11, yolox.

    Args:
        config: Config object with model.num_classes, model.size, epochs, etc.
        architecture: Architecture name (case-insensitive). Defaults to 'flashdet'.
    """
    arch = (architecture or "flashdet").lower().strip()
    num_classes = config.model.num_classes
    total_epochs = getattr(config, "total_epochs", getattr(config, "epochs", 100))

    if arch in ("flashdet", ""):
        size = getattr(config.model, "size", "n")
        return FlashDet(num_classes=num_classes, size=size, total_epochs=total_epochs)

    # YOLO-family models
    width_mult = getattr(config.model, "width_mult", 1.0)
    depth_mult = getattr(config.model, "depth_mult", 1.0)
    reg_max = getattr(config.model, "reg_max", 16)

    arch_map = {
        "yolov8": "YOLOv8",
        "yolov9": "YOLOv9",
        "yolov10": "YOLOv10",
        "yolov11": "YOLOv11",
        "yolox": "YOLOX",
    }

    registry_name = arch_map.get(arch)
    if registry_name is None:
        available = list(DETECTORS._registry.keys())
        raise ValueError(f"Unknown architecture '{arch}'. Available: {available}")

    kwargs = {"num_classes": num_classes, "width_mult": width_mult, "depth_mult": depth_mult}
    if arch == "yolov8":
        kwargs["reg_max"] = reg_max
    elif arch == "yolov9":
        kwargs["use_pgi"] = getattr(config.model, "use_pgi", True)
        kwargs["reg_max"] = reg_max
    elif arch == "yolov10":
        kwargs["use_psa"] = getattr(config.model, "use_psa", True)
        kwargs["reg_max"] = reg_max
    elif arch == "yolov11":
        kwargs["use_c2psa"] = getattr(config.model, "use_c2psa", True)
        kwargs["reg_max"] = reg_max

    model = DETECTORS.build(registry_name, **kwargs)
    logger.info(f"Built {registry_name} (num_classes={num_classes}, width={width_mult}, depth={depth_mult})")
    return model
