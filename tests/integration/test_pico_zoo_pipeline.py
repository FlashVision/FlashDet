"""Integration tests for the Pico-Zoo backbone/neck end-to-end flow."""

import numpy as np
import torch

from flashdet.engine.inference.predictor import Predictor
from flashdet.models.architectures.flashdet import FlashDet
from flashdet.utils.checkpoint import save_inference_weights


def _config(backbone, neck, num_classes=4, size=128):
    return {
        "num_classes": num_classes,
        "model_size": "p",
        "architecture": "flashdet",
        "backbone_type": backbone,
        "neck_type": neck,
        "input_size": [size, size],
        "class_names": [f"c{i}" for i in range(num_classes)],
    }


def test_pico_zoo_train_save_load_predict(tmp_path):
    model = FlashDet(
        num_classes=4, size="p", backbone_type="repvit", neck_type="rep"
    )
    model.train()

    x = torch.randn(2, 3, 128, 128)
    gt = {
        "gt_bboxes": [
            np.array([[10, 10, 40, 40]], dtype=np.float32),
            np.array([[20, 20, 60, 60]], dtype=np.float32),
        ],
        "gt_labels": [
            np.array([0], dtype=np.int64),
            np.array([1], dtype=np.int64),
        ],
    }
    out = model(x, gt_meta=gt, epoch=0, compute_loss=True)
    out["loss"].backward()
    assert torch.isfinite(out["loss"])

    path = tmp_path / "pico_repvit.pth"
    save_inference_weights(model, str(path), config=_config("repvit", "rep"))

    predictor = Predictor(model_path=str(path), device="cpu")
    assert predictor.model.backbone_type == "repvit"
    assert predictor.model.neck_type == "rep"
    detections = predictor(np.random.randint(0, 255, (64, 80, 3), dtype=np.uint8))
    assert isinstance(detections, list)


def test_old_pico_v2_checkpoint_is_still_detected(tmp_path):
    # Old checkpoints did not store backbone_type. The loader must sniff
    # backbone.stem/backbone.stages and rebuild PicoBackbone, not LiteBackbone.
    model = FlashDet(num_classes=4, size="p", backbone_type="pico_v2")
    path = tmp_path / "old_pico.pth"
    save_inference_weights(
        model,
        str(path),
        config={
            "num_classes": 4,
            "model_size": "p",
            "architecture": "flashdet",
            "input_size": [128, 128],
        },
    )
    predictor = Predictor(model_path=str(path), device="cpu")
    assert type(predictor.model.backbone).__name__ == "PicoBackbone"


def test_pico_zoo_checkpoint_mismatch_is_not_silently_wrong(tmp_path):
    # A zoo checkpoint with config must rebuild the zoo backbone, not RepNeXt.
    model = FlashDet(num_classes=4, size="p", backbone_type="mambaout")
    path = tmp_path / "pico_mamba.pth"
    save_inference_weights(
        model,
        str(path),
        config=_config("mambaout", "mamba"),
    )
    predictor = Predictor(model_path=str(path), device="cpu")
    assert type(predictor.model.backbone).__name__ == "PicoZooBackbone"
    assert predictor.model.backbone.arch == "mambaout"
    assert predictor.model.neck_type == "mamba"
