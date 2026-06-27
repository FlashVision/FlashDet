"""Tests for loss functions — E2E loss, training loss computation."""

import numpy as np
import pytest
import torch

from flashdet.models.detector import FlashDet


class TestE2ELoss:
    """End-to-end dual-head loss (o2m + o2o with ProgLoss)."""

    @pytest.fixture
    def model_and_input(self):
        model = FlashDet(num_classes=5, size="n")
        model.train()
        x = torch.randn(2, 3, 320, 320)
        gt = {"gt_bboxes": [], "gt_labels": []}
        for _ in range(2):
            n = np.random.randint(1, 4)
            xy = np.random.rand(n, 2).astype(np.float32) * 200
            wh = np.random.rand(n, 2).astype(np.float32) * 80 + 20
            boxes = np.clip(np.concatenate([xy, xy + wh], axis=1), 0, 319)
            gt["gt_bboxes"].append(boxes)
            gt["gt_labels"].append(np.random.randint(0, 5, (n,)).astype(np.int64))
        return model, x, gt

    def test_loss_is_positive(self, model_and_input):
        model, x, gt = model_and_input
        out = model(x, gt_meta=gt, epoch=0)
        assert out["loss"].item() > 0

    def test_loss_is_finite(self, model_and_input):
        model, x, gt = model_and_input
        out = model(x, gt_meta=gt, epoch=0)
        assert torch.isfinite(out["loss"])

    def test_loss_has_grad(self, model_and_input):
        model, x, gt = model_and_input
        out = model(x, gt_meta=gt, epoch=0)
        assert out["loss"].requires_grad

    def test_o2m_and_o2o_components(self, model_and_input):
        model, x, gt = model_and_input
        out = model(x, gt_meta=gt, epoch=1)
        states = out["loss_states"]
        assert "o2m_loss" in states
        assert "o2o_loss" in states
        assert states["o2m_loss"] >= 0
        assert states["o2o_loss"] >= 0

    def test_loss_stable_over_steps(self, model_and_input):
        model, x, gt = model_and_input
        opt = torch.optim.SGD(model.parameters(), lr=1e-5)
        losses = []
        for _ in range(5):
            opt.zero_grad()
            out = model(x, gt_meta=gt, epoch=0)
            out["loss"].backward()
            opt.step()
            losses.append(out["loss"].item())
        # No explosion
        assert all(l < 10000 for l in losses)
        assert all(not np.isnan(l) for l in losses)

    def test_empty_gt_no_crash(self):
        model = FlashDet(num_classes=5, size="n")
        model.train()
        x = torch.randn(1, 3, 320, 320)
        gt = {
            "gt_bboxes": [np.zeros((0, 4), dtype=np.float32)],
            "gt_labels": [np.array([], dtype=np.int64)],
        }
        out = model(x, gt_meta=gt, epoch=0)
        assert torch.isfinite(out["loss"])

    def test_single_object_gt(self):
        model = FlashDet(num_classes=5, size="n")
        model.train()
        x = torch.randn(1, 3, 320, 320)
        gt = {
            "gt_bboxes": [np.array([[50, 50, 150, 150]], dtype=np.float32)],
            "gt_labels": [np.array([2], dtype=np.int64)],
        }
        out = model(x, gt_meta=gt, epoch=0)
        assert out["loss"].item() > 0
        out["loss"].backward()

    def test_many_objects_gt(self):
        model = FlashDet(num_classes=10, size="n")
        model.train()
        x = torch.randn(1, 3, 320, 320)
        n = 20
        boxes = np.random.rand(n, 4).astype(np.float32) * 300
        boxes[:, 2:] = boxes[:, :2] + np.random.rand(n, 2).astype(np.float32) * 50 + 10
        boxes = np.clip(boxes, 0, 319)
        gt = {
            "gt_bboxes": [boxes],
            "gt_labels": [np.random.randint(0, 10, (n,)).astype(np.int64)],
        }
        out = model(x, gt_meta=gt, epoch=0)
        assert torch.isfinite(out["loss"])
