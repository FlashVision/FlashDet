"""Tests for FlashDet detector — forward pass, training, inference."""

import numpy as np
import pytest
import torch

from flashdet.models.detector import FlashDet, build_model
from flashdet.cfg import get_config


class TestFlashDetForward:
    """Basic forward pass tests."""

    @pytest.mark.parametrize("size", ["n", "s", "m"])
    def test_forward_all_sizes(self, size):
        model = FlashDet(num_classes=10, size=size)
        model.eval()
        x = torch.randn(1, 3, 320, 320)
        with torch.no_grad():
            out = model(x)
        assert "preds" in out
        assert out["preds"].shape[0] == 1

    @pytest.mark.parametrize("input_size", [320, 416, 640])
    def test_forward_various_resolutions(self, input_size):
        model = FlashDet(num_classes=5, size="n")
        model.eval()
        x = torch.randn(1, 3, input_size, input_size)
        with torch.no_grad():
            out = model(x)
        assert out["preds"] is not None

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_forward_batch_sizes(self, batch_size):
        model = FlashDet(num_classes=5, size="n")
        model.eval()
        x = torch.randn(batch_size, 3, 320, 320)
        with torch.no_grad():
            out = model(x)
        assert out["preds"].shape[0] == batch_size

    def test_output_keys_eval_mode(self):
        model = FlashDet(num_classes=10, size="n")
        model.eval()
        x = torch.randn(1, 3, 320, 320)
        with torch.no_grad():
            out = model(x)
        assert "preds" in out
        assert "o2o_cls" in out
        assert "o2o_reg" in out

    def test_output_class_dimension(self):
        nc = 15
        model = FlashDet(num_classes=nc, size="n")
        model.eval()
        x = torch.randn(1, 3, 320, 320)
        with torch.no_grad():
            out = model(x)
        assert out["o2o_cls"].shape[-1] == nc

    def test_no_grad_in_eval(self):
        model = FlashDet(num_classes=5, size="n")
        model.eval()
        x = torch.randn(1, 3, 320, 320)
        with torch.no_grad():
            out = model(x)
        assert not out["preds"].requires_grad


class TestFlashDetTraining:
    """Training forward + backward tests."""

    def _make_gt(self, batch_size=2, num_classes=5, img_size=320):
        gt = {"gt_bboxes": [], "gt_labels": []}
        for _ in range(batch_size):
            n = np.random.randint(1, 5)
            x1y1 = np.random.rand(n, 2).astype(np.float32) * (img_size * 0.6)
            wh = np.random.rand(n, 2).astype(np.float32) * (img_size * 0.3) + 10
            boxes = np.clip(np.concatenate([x1y1, x1y1 + wh], axis=1), 0, img_size - 1)
            labels = np.random.randint(0, num_classes, size=(n,)).astype(np.int64)
            gt["gt_bboxes"].append(boxes)
            gt["gt_labels"].append(labels)
        return gt

    def test_training_produces_loss(self):
        model = FlashDet(num_classes=5, size="n")
        model.train()
        x = torch.randn(2, 3, 320, 320)
        out = model(x, gt_meta=self._make_gt(), epoch=1)
        assert "loss" in out
        assert out["loss"].requires_grad
        assert not torch.isnan(out["loss"])
        assert out["loss"].item() > 0

    def test_backward_produces_gradients(self):
        model = FlashDet(num_classes=5, size="n")
        model.train()
        x = torch.randn(2, 3, 320, 320)
        out = model(x, gt_meta=self._make_gt(), epoch=0)
        out["loss"].backward()
        grads = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
        assert grads > 0

    def test_loss_states_keys(self):
        model = FlashDet(num_classes=5, size="n")
        model.train()
        out = model(torch.randn(2, 3, 320, 320), gt_meta=self._make_gt(), epoch=1)
        expected = ["loss_total", "o2m_loss", "o2o_loss", "prog_alpha"]
        for key in expected:
            assert key in out["loss_states"], f"Missing: {key}"

    def test_empty_ground_truth(self):
        model = FlashDet(num_classes=5, size="n")
        model.train()
        gt = {
            "gt_bboxes": [np.zeros((0, 4), dtype=np.float32)] * 2,
            "gt_labels": [np.array([], dtype=np.int64)] * 2,
        }
        out = model(torch.randn(2, 3, 320, 320), gt_meta=gt, epoch=0)
        assert not torch.isnan(out["loss"])

    def test_prog_loss_decreases_over_epochs(self):
        model = FlashDet(num_classes=5, size="n", total_epochs=100)
        model.train()
        x = torch.randn(1, 3, 320, 320)
        gt = self._make_gt(1, 5)
        early = model(x, gt_meta=gt, epoch=0)
        late = model(x, gt_meta=gt, epoch=99)
        assert early["loss_states"]["prog_alpha"] > late["loss_states"]["prog_alpha"]

    def test_multi_step_no_nan(self):
        model = FlashDet(num_classes=5, size="n")
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        torch.manual_seed(42)
        x = torch.randn(2, 3, 320, 320)
        gt = self._make_gt(2, 5)
        for _ in range(3):
            opt.zero_grad()
            out = model(x, gt_meta=gt, epoch=0)
            out["loss"].backward()
            opt.step()
            assert not torch.isnan(out["loss"])
            assert out["loss"].item() < 10000.0

    def test_gradient_flows_to_input(self):
        model = FlashDet(num_classes=5, size="n")
        model.train()
        x = torch.randn(1, 3, 320, 320, requires_grad=True)
        out = model(x, gt_meta=self._make_gt(1, 5), epoch=0)
        out["loss"].backward()
        assert x.grad is not None and x.grad.abs().sum() > 0


class TestFlashDetInference:
    """Inference / predict tests."""

    def test_predict_output_format(self):
        model = FlashDet(num_classes=5, size="n")
        results = model.predict(torch.randn(2, 3, 320, 320), score_thr=0.01)
        assert len(results) == 2
        for bboxes, labels in results:
            if bboxes.numel() > 0:
                assert bboxes.shape[-1] == 5
            assert labels.ndim == 1

    def test_high_threshold_few_detections(self):
        model = FlashDet(num_classes=5, size="n")
        results = model.predict(torch.randn(1, 3, 320, 320), score_thr=0.99)
        bboxes, _ = results[0]
        assert bboxes.shape[0] <= 10

    def test_return_features(self):
        model = FlashDet(num_classes=5, size="n")
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(1, 3, 320, 320), return_features=True)
        assert "fpn_features" in out
        assert "backbone_features" in out
        assert len(out["fpn_features"]) == 3
        assert len(out["backbone_features"]) == 3


class TestModelInfo:
    """Model metadata / info tests."""

    def test_get_model_info(self):
        model = FlashDet(num_classes=10, size="n")
        info = model.get_model_info()
        assert info["name"] == "FlashDet-N"
        assert info["num_classes"] == 10
        assert info["total_params"] > 0
        assert info["params_mb"] > 0

    def test_larger_model_more_params(self):
        params = {}
        for size in ["n", "s", "m"]:
            m = FlashDet(num_classes=10, size=size)
            params[size] = sum(p.numel() for p in m.parameters())
        assert params["n"] < params["s"] < params["m"]

    def test_build_model_from_config(self):
        cfg = get_config(model_size="n", input_size=320, num_classes=10)
        model = build_model(cfg)
        model.eval()
        out = model(torch.randn(1, 3, 320, 320))
        assert out is not None
