"""
Model builder for FlashDet.
"""

import logging
import warnings
from typing import Dict

from flashdet.models.architectures.flashdet import FlashDet

logger = logging.getLogger(__name__)


def load_coco_pretrained(model, **kwargs) -> Dict[str, list]:
    """Compatibility stub — YOLO26-based FlashDet uses standard checkpoint loading.

    The old NanoDet-specific COCO pretrained loader is no longer needed.
    Use ``torch.load`` / ``model.load_state_dict`` for checkpoints instead.
    """
    warnings.warn(
        "load_coco_pretrained() is deprecated for YOLO26-based FlashDet. "
        "Use standard checkpoint loading (torch.load + load_state_dict).",
        DeprecationWarning,
        stacklevel=2,
    )
    return {"loaded": [], "skipped": []}

ARCHITECTURE_REGISTRY = {
    "flashdet": "FlashDet",
    "detr": "DETR",
    "rt-detr": "RTDETR",
    "rtdetr": "RTDETR",
    "yolov9": "YOLOv9",
    "yolov10": "YOLOv10",
    "yolov11": "YOLOv11",
    "grounding-dino": "GroundingDINO",
    "groundingdino": "GroundingDINO",
}


def build_model(config, architecture: str = None):
    """Build a detection model from config.

    Args:
        config: Model configuration.
        architecture: Architecture name. If ``None`` or ``"flashdet"``,
            builds the YOLO26-based FlashDet model. Other options:
            ``"detr"``, ``"rt-detr"``, ``"yolov9"``, ``"yolov10"``,
            ``"yolov11"``, ``"grounding-dino"``.

    Returns:
        An ``nn.Module`` with ``forward(x, gt_meta)`` and ``predict(x)`` interfaces.
    """
    arch = (architecture or getattr(config.model, "architecture", "flashdet")).lower()

    if arch in ("flashdet", ""):
        size = getattr(config.model, "size", "n")
        total_epochs = getattr(config, "total_epochs", getattr(config, "epochs", 100))
        return FlashDet(
            num_classes=config.model.num_classes,
            size=size,
            total_epochs=total_epochs,
        )

    class_name = ARCHITECTURE_REGISTRY.get(arch)
    if class_name is None:
        available = ", ".join(sorted(ARCHITECTURE_REGISTRY.keys()))
        raise ValueError(f"Unknown architecture '{arch}'. Available: {available}")

    from flashdet.registry import DETECTORS
    cls = DETECTORS.get(class_name)
    num_classes = config.model.num_classes

    return cls(num_classes=num_classes)
