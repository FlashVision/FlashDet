from .e2e_loss import E2EDetectionLoss
from .yolo_loss import compute_yolo_loss
from .varifocal_loss import VarifocalLoss, SigmoidFocalLoss
from flashdet.registry import LOSSES

LOSSES.register("E2EDetectionLoss")(E2EDetectionLoss)
LOSSES.register("VarifocalLoss")(VarifocalLoss)
LOSSES.register("SigmoidFocalLoss")(SigmoidFocalLoss)

__all__ = [
    "E2EDetectionLoss",
    "compute_yolo_loss",
    "VarifocalLoss",
    "SigmoidFocalLoss",
]
