"""Security tests for checkpoint loading."""

import os
import pickle

import pytest
import torch

from flashdet.models.architectures.flashdet import FlashDet
from flashdet.utils.checkpoint import (
    load_checkpoint,
    load_inference_model,
    safe_torch_load,
    save_inference_weights,
)


def _malicious_pickle(path, marker):
    class Evil:
        def __reduce__(self):
            return (os.system, (f"touch {marker}",))

    with open(path, "wb") as f:
        pickle.dump(Evil(), f)


def test_safe_torch_load_loads_normal_checkpoint(tmp_path):
    path = tmp_path / "normal.pth"
    torch.save({"model_state_dict": {"x": torch.zeros(1)}, "epoch": 3}, path)
    ckpt = safe_torch_load(path)
    assert ckpt["epoch"] == 3


def test_safe_torch_load_rejects_pickle_rce(tmp_path):
    marker = tmp_path / "pwned"
    path = tmp_path / "evil.pth"
    _malicious_pickle(path, marker)

    with pytest.raises(RuntimeError, match="Refusing to load untrusted checkpoint"):
        safe_torch_load(path)
    assert not marker.exists(), "unsafe pickle payload was executed"


def test_load_checkpoint_rejects_pickle_rce(tmp_path):
    marker = tmp_path / "pwned"
    path = tmp_path / "evil.pth"
    _malicious_pickle(path, marker)
    model = torch.nn.Linear(2, 2)

    with pytest.raises(RuntimeError, match="Refusing to load untrusted checkpoint"):
        load_checkpoint(model, str(path), device="cpu")
    assert not marker.exists()


def test_load_inference_model_rebuilds_pico_zoo(tmp_path):
    model = FlashDet(num_classes=4, size="p", backbone_type="edgenext").eval()
    path = tmp_path / "pico.pth"
    save_inference_weights(
        model,
        str(path),
        config={
            "num_classes": 4,
            "model_size": "p",
            "architecture": "flashdet",
            "backbone_type": "edgenext",
            "neck_type": "attention",
            "input_size": [128, 128],
        },
    )
    loaded = load_inference_model(str(path), device="cpu")
    assert loaded.backbone_type == "edgenext"
    assert loaded.neck_type == "attention"
    with torch.no_grad():
        out = loaded(torch.randn(1, 3, 128, 128))
    assert "o2o_cls" in out
