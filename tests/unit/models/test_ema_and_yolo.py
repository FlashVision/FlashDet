"""Regression tests for EMA warmup and YOLOX batched decode."""

import torch

from flashdet.engine.core.ema import ModelEMA
from flashdet.models.architectures.yolov8 import YOLOv8
from flashdet.models.architectures.yolox import YOLOX
from flashdet.models.architectures.yolov10 import YOLOv10


def test_ema_reaches_target_scale_in_reasonable_updates():
    model = torch.nn.Linear(2, 2)
    ema = ModelEMA(model, decay=0.9998, warmup=2000)
    assert ema.decay == 0.0

    ema.num_updates = 2000
    assert ema.decay > 0.6 * 0.9998

    ema.num_updates = 10000
    assert ema.decay > 0.99 * 0.9998


def test_yolox_decode_returns_one_result_per_image():
    model = YOLOX(num_classes=2, width_mult=0.25, depth_mult=0.33)
    model.eval()
    with torch.no_grad():
        results = model.predict(torch.randn(2, 3, 64, 64), score_thr=1.1)
    assert len(results) == 2
    assert all(dets.shape[0] == 0 for dets, _ in results)


def test_yolov8_accepts_training_loss():
    model = YOLOv8(num_classes=2, width_mult=0.25, depth_mult=0.33)
    model.train()
    gt = {
        "gt_bboxes": [[[10, 10, 40, 40]]],
        "gt_labels": [[0]],
    }
    out = model(torch.randn(1, 3, 64, 64), gt_meta=gt)
    assert "loss" in out
    assert out["loss"].requires_grad
    out["loss"].backward()


def test_yolov10_trains_both_heads():
    model = YOLOv10(num_classes=2, width_mult=0.25, depth_mult=0.33)
    model.train()
    gt = {
        "gt_bboxes": [[[10, 10, 40, 40]]],
        "gt_labels": [[0]],
    }
    out = model(torch.randn(1, 3, 64, 64), gt_meta=gt)
    assert "o2m_loss" in out["loss_states"]
    assert "o2o_loss" in out["loss_states"]
    assert out["loss"].requires_grad
    out["loss"].backward()
    assert any(p.grad is not None for p in model.o2o_heads.parameters())
