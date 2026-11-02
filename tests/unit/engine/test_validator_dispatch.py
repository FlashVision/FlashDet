"""Validator must rebuild the architecture recorded in the checkpoint."""

from flashdet.engine.evaluation.validator import Validator
from flashdet.models.architectures.yolov8 import YOLOv8
from flashdet.utils.checkpoint import save_inference_weights


def test_validator_rebuilds_yolo_checkpoint(tmp_path):
    model = YOLOv8(num_classes=2, width_mult=0.25, depth_mult=0.33)
    path = tmp_path / "yolo.pth"
    save_inference_weights(
        model,
        str(path),
        config={
            "num_classes": 2,
            "model_size": "s",
            "architecture": "yolov8",
            "width_mult": 0.25,
            "depth_mult": 0.33,
            "input_size": [64, 64],
            "class_names": ["a", "b"],
        },
    )

    validator = Validator(model_path=str(path), device="cpu")
    assert type(validator.model).__name__ == "YOLOv8"
    assert len(validator.class_names) == 2
