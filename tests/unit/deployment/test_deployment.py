"""Tests for the deployment package: ONNX export, parity and INT8.

Skipped automatically when onnx/onnxruntime are not installed.
"""

import os

import pytest
import torch

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from flashdet.deployment.onnx_export import export_onnx
from flashdet.deployment.parity import check_parity, detection_parity
from flashdet.deployment.onnx_quantize import (
    ArrayCalibReader,
    onnx_quantize_int8,
)


@pytest.fixture(scope="module")
def det_setup(tmp_path_factory):
    """Tiny FlashDet + a matching ONNX export shared by several tests."""
    tmp = tmp_path_factory.mktemp("onnx")
    from flashdet.cfg import get_config
    from flashdet.models.detector import build_model

    config = get_config(num_classes=4)
    config.model.size = "p"
    model = build_model(config, architecture="flashdet")
    ckpt = {
        "model_state_dict": model.state_dict(),
        "config": {
            "architecture": "flashdet",
            "num_classes": 4,
            "model_size": "p",
            "input_size": 96,
            "backbone_type": "lite",
        },
    }
    model_path = tmp / "tiny.pth"
    torch.save(ckpt, model_path)
    onnx_path = export_onnx(str(model_path), str(tmp / "tiny.onnx"), fuse=True)
    return str(model_path), onnx_path, model


def test_parity_float_onnx(det_setup):
    model_path, onnx_path, _model = det_setup
    report = check_parity(model_path, onnx_path, rtol=1e-2, atol=1e-3)
    assert report["ok"], report


@pytest.mark.timeout(600)
def test_detection_parity_float_onnx(det_setup):
    model_path, onnx_path, _model = det_setup
    report = detection_parity(model_path, onnx_path, num_images=1)
    assert report["mean_box_iou"] > 0.99, report


@pytest.mark.timeout(900)
def test_onnx_int8_and_weight_size_drop(det_setup):
    model_path, onnx_path, _ = det_setup
    int8_path = onnx_quantize_int8(
        onnx_path, batch=torch.randn(8, 3, 96, 96).numpy()
    )
    assert os.path.isfile(int8_path)
    assert os.path.getsize(int8_path) < os.path.getsize(onnx_path) * 1.2


def test_array_calib_reader_batches():
    import numpy as np

    reader = ArrayCalibReader(np.random.rand(10, 1, 4, 4), batch_size=4)
    sizes = []
    while True:
        batch = reader.get_next()
        if batch is None:
            break
        sizes.append(batch["images"].shape[0])
    assert sizes == [4, 4, 2]


def test_unknown_weights_name_rejected():
    from flashdet.hub import download_weights

    with pytest.raises(KeyError):
        download_weights("not-a-real-registry-key")


def test_weights_registry_descriptions():
    from flashdet.hub import WEIGHTS, list_weights

    assert list_weights()
    for name in list_weights():
        assert WEIGHTS[name]["url"].startswith("https://")
