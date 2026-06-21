"""Unit tests for FlashDet models."""

import torch
import numpy as np
import pytest
from flashdet.cfg import get_config
from flashdet.models import build_model, apply_lora
from flashdet.models.detector import FlashDet


@pytest.mark.parametrize("model_size", ["m-0.5x", "m", "m-1.5x"])
def test_model_forward(model_size):
    cfg = get_config(model_size=model_size, input_size=320, num_classes=10)
    model = build_model(cfg)
    model.eval()
    x = torch.randn(1, 3, 320, 320)
    with torch.no_grad():
        out = model(x)
    assert out is not None


@pytest.mark.parametrize("input_size", [320, 416])
def test_input_sizes(input_size):
    cfg = get_config(model_size="m", input_size=input_size, num_classes=5)
    model = build_model(cfg)
    model.eval()
    x = torch.randn(1, 3, input_size, input_size)
    with torch.no_grad():
        out = model(x)
    assert out is not None


def test_lora_reduces_trainable_params():
    cfg = get_config(model_size="m", input_size=320, num_classes=10)
    model = build_model(cfg)
    total_before = sum(p.numel() for p in model.parameters() if p.requires_grad)

    apply_lora(model, rank=4, alpha=8.0)
    total_after = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert total_after < total_before


def test_model_relative_sizes():
    """Larger model variants should have more parameters."""
    params = {}
    for size in ["m-0.5x", "m", "m-1.5x"]:
        cfg = get_config(model_size=size, input_size=320, num_classes=10)
        model = build_model(cfg)
        params[size] = sum(p.numel() for p in model.parameters())

    assert params["m-0.5x"] < params["m"] < params["m-1.5x"]


# ======================================================================
# Custom FlashDet Training Tests
# ======================================================================


def _make_gt_meta(batch_size=2, num_classes=5, img_size=320):
    gt_meta = {"gt_bboxes": [], "gt_labels": []}
    for _ in range(batch_size):
        n_objs = np.random.randint(1, 5)
        x1y1 = np.random.rand(n_objs, 2).astype(np.float32) * (img_size * 0.6)
        wh = np.random.rand(n_objs, 2).astype(np.float32) * (img_size * 0.3) + 10
        boxes = np.concatenate([x1y1, x1y1 + wh], axis=1)
        boxes = np.clip(boxes, 0, img_size - 1)
        labels = np.random.randint(0, num_classes, size=(n_objs,)).astype(np.int64)
        gt_meta["gt_bboxes"].append(boxes)
        gt_meta["gt_labels"].append(labels)
    return gt_meta


class TestFlashDetTraining:
    """Tests for the core FlashDet training forward pass."""

    def test_training_step_basic(self):
        model = FlashDet(num_classes=5, input_size=(320, 320), backbone_size="0.5x",
                         fpn_channels=96, pretrained=False, use_aux_head=True)
        model.train()
        x = torch.randn(2, 3, 320, 320)
        gt_meta = _make_gt_meta(2, 5)
        out = model(x, gt_meta=gt_meta, epoch=1)
        assert "loss" in out
        assert "loss_states" in out
        assert out["loss"].requires_grad
        assert not torch.isnan(out["loss"])
        assert out["loss"].item() > 0

    def test_training_backward(self):
        model = FlashDet(num_classes=5, input_size=(320, 320), backbone_size="0.5x",
                         fpn_channels=96, pretrained=False, use_aux_head=False)
        model.train()
        x = torch.randn(2, 3, 320, 320)
        gt_meta = _make_gt_meta(2, 5)
        out = model(x, gt_meta=gt_meta, epoch=0)
        out["loss"].backward()
        grad_count = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
        assert grad_count > 0

    def test_training_loss_states_with_aux(self):
        model = FlashDet(num_classes=5, input_size=(320, 320), backbone_size="0.5x",
                         fpn_channels=96, pretrained=False, use_aux_head=True)
        model.train()
        x = torch.randn(2, 3, 320, 320)
        gt_meta = _make_gt_meta(2, 5)
        out = model(x, gt_meta=gt_meta, epoch=1)
        states = out["loss_states"]
        expected_keys = ["loss_qfl", "loss_bbox", "loss_dfl",
                         "aux_loss_qfl", "aux_loss_bbox", "aux_loss_dfl"]
        for key in expected_keys:
            assert key in states, f"Missing key: {key}"

    def test_training_loss_states_without_aux(self):
        model = FlashDet(num_classes=5, input_size=(320, 320), backbone_size="0.5x",
                         fpn_channels=96, pretrained=False, use_aux_head=False)
        model.train()
        x = torch.randn(2, 3, 320, 320)
        gt_meta = _make_gt_meta(2, 5)
        out = model(x, gt_meta=gt_meta, epoch=0)
        states = out["loss_states"]
        assert "loss_qfl" in states
        assert "loss_bbox" in states
        assert "loss_dfl" in states
        # no aux keys
        assert "aux_loss_qfl" not in states

    def test_training_empty_gt(self):
        model = FlashDet(num_classes=5, input_size=(320, 320), backbone_size="0.5x",
                         fpn_channels=96, pretrained=False, use_aux_head=False)
        model.train()
        x = torch.randn(2, 3, 320, 320)
        gt_meta = {
            "gt_bboxes": [np.zeros((0, 4), dtype=np.float32),
                          np.zeros((0, 4), dtype=np.float32)],
            "gt_labels": [np.array([], dtype=np.int64),
                          np.array([], dtype=np.int64)],
        }
        out = model(x, gt_meta=gt_meta, epoch=0)
        assert "loss" in out
        assert not torch.isnan(out["loss"])

    def test_training_single_image(self):
        model = FlashDet(num_classes=3, input_size=(320, 320), backbone_size="0.5x",
                         fpn_channels=96, pretrained=False, use_aux_head=True)
        model.train()
        x = torch.randn(1, 3, 320, 320)
        gt_meta = _make_gt_meta(1, 3)
        out = model(x, gt_meta=gt_meta, epoch=0)
        assert "loss" in out
        out["loss"].backward()

    def test_training_detach_epoch(self):
        """Before detach_epoch, aux_fpn gets un-detached features."""
        model = FlashDet(num_classes=5, input_size=(320, 320), backbone_size="0.5x",
                         fpn_channels=96, pretrained=False, use_aux_head=True)
        model.detach_epoch = 5
        model.train()
        x = torch.randn(1, 3, 320, 320)
        gt_meta = _make_gt_meta(1, 5)
        # epoch < detach_epoch: un-detached path
        out = model(x, gt_meta=gt_meta, epoch=0)
        assert out["loss"].requires_grad
        out["loss"].backward()

    def test_compute_loss_eval_mode(self):
        """compute_loss=True should work in eval mode (for validation)."""
        model = FlashDet(num_classes=5, input_size=(320, 320), backbone_size="0.5x",
                         fpn_channels=96, pretrained=False, use_aux_head=False)
        model.eval()
        x = torch.randn(2, 3, 320, 320)
        gt_meta = _make_gt_meta(2, 5)
        out = model(x, gt_meta=gt_meta, compute_loss=True)
        assert "loss" in out
        assert "loss_states" in out

    def test_gradient_flow_to_input(self):
        model = FlashDet(num_classes=5, input_size=(320, 320), backbone_size="0.5x",
                         fpn_channels=96, pretrained=False, use_aux_head=False)
        model.train()
        x = torch.randn(1, 3, 320, 320, requires_grad=True)
        gt_meta = _make_gt_meta(1, 5)
        out = model(x, gt_meta=gt_meta, epoch=0)
        out["loss"].backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_training_with_lora(self):
        model = FlashDet(num_classes=5, input_size=(320, 320), backbone_size="0.5x",
                         fpn_channels=96, pretrained=False, use_aux_head=False)
        apply_lora(model, rank=4, alpha=8.0)
        model.train()
        x = torch.randn(1, 3, 320, 320)
        gt_meta = _make_gt_meta(1, 5)
        out = model(x, gt_meta=gt_meta, epoch=0)
        assert out["loss"].requires_grad
        out["loss"].backward()
        # Verify LoRA params get gradients
        lora_grads = 0
        for name, p in model.named_parameters():
            if "lora" in name and p.grad is not None and p.grad.abs().sum() > 0:
                lora_grads += 1
        assert lora_grads > 0

    def test_multi_step_optimization(self):
        """Simulate multiple optimizer steps and verify loss is finite & bounded."""
        model = FlashDet(num_classes=5, input_size=(320, 320), backbone_size="0.5x",
                         fpn_channels=96, pretrained=False, use_aux_head=False)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        torch.manual_seed(42)
        x = torch.randn(2, 3, 320, 320)
        gt_meta = _make_gt_meta(2, 5)

        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            out = model(x, gt_meta=gt_meta, epoch=0)
            out["loss"].backward()
            optimizer.step()
            losses.append(out["loss"].item())

        # All losses should be finite and non-NaN
        assert all(not np.isnan(l) for l in losses)
        assert all(not np.isinf(l) for l in losses)
        # Loss should not explode to unreasonable values
        assert max(losses) < 100.0


class TestFlashDetInference:
    """Tests for FlashDet inference."""

    def test_inference_output_shape(self):
        model = FlashDet(num_classes=10, input_size=(320, 320), backbone_size="0.5x",
                         fpn_channels=96, pretrained=False, use_aux_head=False)
        model.eval()
        x = torch.randn(1, 3, 320, 320)
        with torch.no_grad():
            out = model(x)
        assert "preds" in out
        B, N, C = out["preds"].shape
        assert B == 1
        assert C == 10 + 4 * 8  # num_classes + 4*(reg_max+1)

    def test_inference_batch(self):
        model = FlashDet(num_classes=5, input_size=(320, 320), backbone_size="0.5x",
                         fpn_channels=96, pretrained=False, use_aux_head=False)
        model.eval()
        x = torch.randn(4, 3, 320, 320)
        with torch.no_grad():
            out = model(x)
        assert out["preds"].shape[0] == 4

    def test_predict_returns_detections(self):
        model = FlashDet(num_classes=5, input_size=(320, 320), backbone_size="0.5x",
                         fpn_channels=96, pretrained=False, use_aux_head=False)
        x = torch.randn(2, 3, 320, 320)
        results = model.predict(x, score_thr=0.01)
        assert len(results) == 2
        for det_bboxes, det_labels in results:
            if det_bboxes.numel() > 0:
                assert det_bboxes.shape[-1] == 5  # x1,y1,x2,y2,score
            assert det_labels.ndim == 1

    def test_predict_high_threshold(self):
        model = FlashDet(num_classes=5, input_size=(320, 320), backbone_size="0.5x",
                         fpn_channels=96, pretrained=False, use_aux_head=False)
        x = torch.randn(1, 3, 320, 320)
        results = model.predict(x, score_thr=0.99)
        det_bboxes, det_labels = results[0]
        # Very high threshold should yield few/no detections
        assert det_bboxes.shape[0] <= 10

    def test_return_features(self):
        model = FlashDet(num_classes=5, input_size=(320, 320), backbone_size="0.5x",
                         fpn_channels=96, pretrained=False, use_aux_head=False)
        model.eval()
        x = torch.randn(1, 3, 320, 320)
        with torch.no_grad():
            out = model(x, return_features=True)
        assert "fpn_features" in out
        assert "backbone_features" in out
        assert "preds" in out
        assert len(out["fpn_features"]) == 4
        assert len(out["backbone_features"]) == 3

    def test_no_grad_in_eval(self):
        model = FlashDet(num_classes=5, input_size=(320, 320), backbone_size="0.5x",
                         fpn_channels=96, pretrained=False, use_aux_head=False)
        model.eval()
        x = torch.randn(1, 3, 320, 320)
        with torch.no_grad():
            out = model(x)
        assert not out["preds"].requires_grad


class TestModelInfo:
    """Tests for model information utilities."""

    def test_get_model_info(self):
        model = FlashDet(num_classes=10, input_size=(320, 320), backbone_size="0.5x",
                         fpn_channels=96, pretrained=False, use_aux_head=True)
        info = model.get_model_info()
        assert info["name"] == "FlashDet"
        assert info["num_classes"] == 10
        assert info["total_params"] > 0
        assert info["inference_params"] < info["total_params"]
        assert info["params_mb"] > 0
        assert info["inference_fp16_mb"] > 0

    def test_inference_params_exclude_aux(self):
        model_aux = FlashDet(num_classes=5, input_size=(320, 320), backbone_size="0.5x",
                             fpn_channels=96, pretrained=False, use_aux_head=True)
        model_no_aux = FlashDet(num_classes=5, input_size=(320, 320), backbone_size="0.5x",
                                fpn_channels=96, pretrained=False, use_aux_head=False)
        info_aux = model_aux.get_model_info()
        info_no_aux = model_no_aux.get_model_info()
        # Inference params of model with aux should roughly equal total of model without aux
        assert abs(info_aux["inference_params"] - info_no_aux["total_params"]) < 100


class TestModelEMA:
    """Tests for Exponential Moving Average."""

    def test_ema_creation(self):
        from flashdet.engine.trainer import ModelEMA

        model = FlashDet(num_classes=5, input_size=(320, 320), backbone_size="0.5x",
                         fpn_channels=96, pretrained=False, use_aux_head=False)
        ema = ModelEMA(model, decay=0.999, warmup=10)
        assert ema.num_updates == 0
        for p in ema.ema.parameters():
            assert not p.requires_grad

    def test_ema_update_changes_params(self):
        from flashdet.engine.trainer import ModelEMA

        model = FlashDet(num_classes=5, input_size=(320, 320), backbone_size="0.5x",
                         fpn_channels=96, pretrained=False, use_aux_head=False)
        ema = ModelEMA(model, decay=0.999, warmup=10)

        # Mutate model params
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        ema.update(model)
        assert ema.num_updates == 1

    def test_ema_warmup_ramps_decay(self):
        from flashdet.engine.trainer import ModelEMA

        model = FlashDet(num_classes=5, input_size=(320, 320), backbone_size="0.5x",
                         fpn_channels=96, pretrained=False, use_aux_head=False)
        ema = ModelEMA(model, decay=0.9999, warmup=100)

        ema.num_updates = 0
        d0 = ema.decay
        ema.num_updates = 50
        d50 = ema.decay
        ema.num_updates = 10000
        d_late = ema.decay

        assert d0 < d50 < d_late
        assert d_late <= 0.9999

    def test_ema_state_dict_roundtrip(self):
        from flashdet.engine.trainer import ModelEMA

        model = FlashDet(num_classes=5, input_size=(320, 320), backbone_size="0.5x",
                         fpn_channels=96, pretrained=False, use_aux_head=False)
        ema = ModelEMA(model, decay=0.998, warmup=500)
        ema.num_updates = 42

        sd = ema.state_dict()
        ema2 = ModelEMA(model, decay=0.5, warmup=1)
        ema2.load_state_dict(sd)

        assert ema2.num_updates == 42
        assert ema2.target_decay == 0.998
        assert ema2.warmup == 500
