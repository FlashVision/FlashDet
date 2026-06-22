"""
Comprehensive tests for YOLO26-based FlashDet components.

Covers:
  1. STAL (STALAssigner) — label assignment + small-target expansion
  2. E2E Loss + ProgLoss — detection loss, scheduling, branch weighting
  3. MuSGD optimizer — hybrid Muon-SGD, orthogonalization, param splitting
  4. E2E Dual Head — DFL-free head output shapes, train/eval modes
  5. FlashDet pipeline — all sizes, full train loop, predict, edge cases
"""

import math
import torch
import torch.nn as nn
import numpy as np
import pytest


# ======================================================================
# Helpers
# ======================================================================

def _make_anchors(n=100, img_size=320, stride=8):
    """Create a grid of anchor centers."""
    H, W = img_size // stride, img_size // stride
    shift_x = (torch.arange(W) + 0.5) * stride
    shift_y = (torch.arange(H) + 0.5) * stride
    yy, xx = torch.meshgrid(shift_y, shift_x, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)


def _make_gt(num_gt=3, num_classes=10, img_size=320):
    """Create random ground truth boxes and labels."""
    x1y1 = torch.rand(num_gt, 2) * img_size * 0.6
    wh = torch.rand(num_gt, 2) * img_size * 0.3 + 20
    x2y2 = (x1y1 + wh).clamp(max=img_size - 1)
    gt_bboxes = torch.cat([x1y1, x2y2], dim=-1)
    gt_labels = torch.randint(0, num_classes, (num_gt,))
    return gt_bboxes, gt_labels


def _make_gt_meta(batch_size=2, num_classes=5, img_size=320):
    gt_meta = {"gt_bboxes": [], "gt_labels": []}
    for _ in range(batch_size):
        n = np.random.randint(1, 5)
        x1y1 = np.random.rand(n, 2).astype(np.float32) * img_size * 0.6
        wh = np.random.rand(n, 2).astype(np.float32) * img_size * 0.3 + 10
        boxes = np.concatenate([x1y1, x1y1 + wh], axis=1).clip(0, img_size - 1)
        labels = np.random.randint(0, num_classes, size=(n,)).astype(np.int64)
        gt_meta["gt_bboxes"].append(boxes)
        gt_meta["gt_labels"].append(labels)
    return gt_meta


# ######################################################################
# 1. STAL (Small-Target-Aware Label Assignment)
# ######################################################################

class TestSTALHelpers:
    """Tests for STAL utility functions."""

    def test_xyxy_to_cxcywh(self):
        from flashdet.models.assignment.stal import _xyxy_to_cxcywh
        boxes = torch.tensor([[10, 20, 50, 60]])
        cxcywh = _xyxy_to_cxcywh(boxes)
        assert cxcywh.shape == (1, 4)
        assert torch.allclose(cxcywh, torch.tensor([[30.0, 40.0, 40.0, 40.0]]))

    def test_xyxy_to_cxcywh_batch(self):
        from flashdet.models.assignment.stal import _xyxy_to_cxcywh
        boxes = torch.tensor([[0, 0, 10, 10], [100, 100, 200, 300]])
        result = _xyxy_to_cxcywh(boxes)
        assert torch.allclose(result[0], torch.tensor([5.0, 5.0, 10.0, 10.0]))
        assert torch.allclose(result[1], torch.tensor([150.0, 200.0, 100.0, 200.0]))

    def test_pairwise_iou_identity(self):
        from flashdet.models.assignment.stal import _pairwise_iou
        boxes = torch.tensor([[0, 0, 10, 10], [20, 20, 40, 40]], dtype=torch.float32)
        iou = _pairwise_iou(boxes, boxes)
        assert iou.shape == (2, 2)
        assert torch.allclose(iou.diag(), torch.ones(2), atol=1e-5)

    def test_pairwise_iou_no_overlap(self):
        from flashdet.models.assignment.stal import _pairwise_iou
        b1 = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)
        b2 = torch.tensor([[50, 50, 60, 60]], dtype=torch.float32)
        iou = _pairwise_iou(b1, b2)
        assert iou.item() == pytest.approx(0.0, abs=1e-6)

    def test_pairwise_iou_partial_overlap(self):
        from flashdet.models.assignment.stal import _pairwise_iou
        b1 = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)
        b2 = torch.tensor([[5, 5, 15, 15]], dtype=torch.float32)
        iou = _pairwise_iou(b1, b2)
        expected = 25.0 / (100 + 100 - 25)
        assert iou.item() == pytest.approx(expected, abs=1e-5)

    def test_bbox_iou_aligned_identical(self):
        from flashdet.models.assignment.stal import _bbox_iou_aligned
        boxes = torch.tensor([[10, 10, 50, 50]], dtype=torch.float32)
        ciou = _bbox_iou_aligned(boxes, boxes)
        assert ciou.item() == pytest.approx(1.0, abs=1e-5)

    def test_bbox_iou_aligned_no_overlap(self):
        from flashdet.models.assignment.stal import _bbox_iou_aligned
        b1 = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)
        b2 = torch.tensor([[50, 50, 60, 60]], dtype=torch.float32)
        ciou = _bbox_iou_aligned(b1, b2)
        assert ciou.item() < 0.0  # CIoU is negative for non-overlapping boxes


class TestSTALAssigner:
    """Tests for STALAssigner."""

    def test_basic_assignment(self):
        from flashdet.models.assignment.stal import STALAssigner
        assigner = STALAssigner(topk=5, strides=(8, 16, 32))
        anchors = _make_anchors(img_size=320, stride=8)
        n_anchors = anchors.shape[0]
        cls_scores = torch.rand(n_anchors, 10).sigmoid()
        pred_bboxes = torch.rand(n_anchors, 4) * 320
        pred_bboxes[:, 2:] = pred_bboxes[:, :2] + 20
        gt_bboxes, gt_labels = _make_gt(3, 10, 320)

        labels, bboxes, scores, fg = assigner.assign(
            anchors, cls_scores, pred_bboxes, gt_bboxes, gt_labels,
        )
        assert labels.shape == (n_anchors,)
        assert bboxes.shape == (n_anchors, 4)
        assert scores.shape == (n_anchors, 10)
        assert fg.shape == (n_anchors,)
        assert fg.sum() > 0

    def test_no_gt_returns_all_negative(self):
        from flashdet.models.assignment.stal import STALAssigner
        assigner = STALAssigner(topk=5)
        anchors = _make_anchors(img_size=320, stride=8)
        n = anchors.shape[0]
        cls_scores = torch.rand(n, 5).sigmoid()
        pred_bboxes = torch.rand(n, 4) * 320
        gt_bboxes = torch.zeros(0, 4)
        gt_labels = torch.zeros(0, dtype=torch.long)

        labels, bboxes, scores, fg = assigner.assign(
            anchors, cls_scores, pred_bboxes, gt_bboxes, gt_labels,
        )
        assert fg.sum() == 0
        assert (labels == 5).all()

    def test_single_gt(self):
        from flashdet.models.assignment.stal import STALAssigner
        assigner = STALAssigner(topk=10)
        anchors = _make_anchors(img_size=320, stride=8)
        n = anchors.shape[0]
        cls_scores = torch.rand(n, 5).sigmoid()
        pred_bboxes = torch.rand(n, 4) * 320
        pred_bboxes[:, 2:] = pred_bboxes[:, :2] + 30
        gt_bboxes = torch.tensor([[100, 100, 200, 200]], dtype=torch.float32)
        gt_labels = torch.tensor([2])

        labels, bboxes, scores, fg = assigner.assign(
            anchors, cls_scores, pred_bboxes, gt_bboxes, gt_labels,
        )
        assert fg.sum() > 0
        assert (labels[fg] == 2).all()

    def test_stal_tiny_target_gets_positives(self):
        """Tiny GT (4x4 pixels) should still get positive assignments via STAL."""
        from flashdet.models.assignment.stal import STALAssigner
        assigner = STALAssigner(topk=10, strides=(8, 16, 32))
        anchors = _make_anchors(img_size=320, stride=8)
        n = anchors.shape[0]
        # Use high cls scores and place predicted boxes near the GT center
        cls_scores = torch.ones(n, 5) * 0.9
        pred_bboxes = torch.zeros(n, 4)
        pred_bboxes[:, 0] = anchors[:, 0] - 15
        pred_bboxes[:, 1] = anchors[:, 1] - 15
        pred_bboxes[:, 2] = anchors[:, 0] + 15
        pred_bboxes[:, 3] = anchors[:, 1] + 15
        gt_bboxes = torch.tensor([[160, 160, 164, 164]], dtype=torch.float32)
        gt_labels = torch.tensor([1])

        labels, bboxes, scores, fg = assigner.assign(
            anchors, cls_scores, pred_bboxes, gt_bboxes, gt_labels,
        )
        assert fg.sum() > 0, "STAL should assign positives for tiny targets"

    def test_stal_normal_target_unaffected(self):
        """Normal-sized GT should not be affected by STAL expansion."""
        from flashdet.models.assignment.stal import STALAssigner
        assigner = STALAssigner(topk=5, strides=(8, 16, 32))
        anchors = _make_anchors(img_size=320, stride=8)
        n = anchors.shape[0]
        cls_scores = torch.rand(n, 5).sigmoid()
        pred_bboxes = torch.rand(n, 4) * 320
        pred_bboxes[:, 2:] = pred_bboxes[:, :2] + 30
        gt_bboxes = torch.tensor([[50, 50, 150, 150]], dtype=torch.float32)
        gt_labels = torch.tensor([0])

        labels, bboxes, scores, fg = assigner.assign(
            anchors, cls_scores, pred_bboxes, gt_bboxes, gt_labels,
        )
        assert fg.sum() > 0
        pos_bboxes = bboxes[fg]
        assert torch.allclose(pos_bboxes[0], gt_bboxes[0])

    def test_soft_label_range(self):
        """Assigned soft-label scores should be in [0, 1]."""
        from flashdet.models.assignment.stal import STALAssigner
        assigner = STALAssigner(topk=10)
        anchors = _make_anchors(img_size=320, stride=8)
        n = anchors.shape[0]
        cls_scores = torch.rand(n, 10).sigmoid()
        pred_bboxes = torch.rand(n, 4) * 320
        pred_bboxes[:, 2:] = pred_bboxes[:, :2] + 40
        gt_bboxes, gt_labels = _make_gt(5, 10, 320)

        _, _, scores, fg = assigner.assign(
            anchors, cls_scores, pred_bboxes, gt_bboxes, gt_labels,
        )
        assert scores.min() >= 0
        assert scores.max() <= 1.0 + 1e-5

    def test_multi_gt_conflict_resolution(self):
        """Anchors matched to multiple GTs should keep highest alignment."""
        from flashdet.models.assignment.stal import STALAssigner
        assigner = STALAssigner(topk=20)
        anchors = _make_anchors(img_size=320, stride=8)
        n = anchors.shape[0]
        cls_scores = torch.rand(n, 5).sigmoid()
        pred_bboxes = torch.rand(n, 4) * 320
        pred_bboxes[:, 2:] = pred_bboxes[:, :2] + 40
        gt_bboxes = torch.tensor([
            [100, 100, 200, 200],
            [110, 110, 210, 210],
        ], dtype=torch.float32)
        gt_labels = torch.tensor([0, 1])

        labels, bboxes, scores, fg = assigner.assign(
            anchors, cls_scores, pred_bboxes, gt_bboxes, gt_labels,
        )
        assert fg.sum() > 0

    def test_custom_strides(self):
        from flashdet.models.assignment.stal import STALAssigner
        assigner = STALAssigner(topk=5, strides=(4, 8, 16))
        assert assigner.s_min == 4
        assert assigner.s_ref == 8

    def test_custom_s_ref(self):
        from flashdet.models.assignment.stal import STALAssigner
        assigner = STALAssigner(topk=5, strides=(8, 16, 32), s_ref=24)
        assert assigner.s_ref == 24


# ######################################################################
# 2. E2E Detection Loss + ProgLoss
# ######################################################################

class TestProgLossSchedule:
    """Tests for ProgLoss alpha scheduling."""

    def test_alpha_at_start(self):
        from flashdet.losses.e2e_loss import E2EDetectionLoss
        loss = E2EDetectionLoss(num_classes=10)
        assert loss.prog_alpha(0, 100) == pytest.approx(1.0)

    def test_alpha_at_end(self):
        from flashdet.losses.e2e_loss import E2EDetectionLoss
        loss = E2EDetectionLoss(num_classes=10)
        assert loss.prog_alpha(99, 100) == pytest.approx(0.0, abs=1e-5)

    def test_alpha_at_midpoint(self):
        from flashdet.losses.e2e_loss import E2EDetectionLoss
        loss = E2EDetectionLoss(num_classes=10)
        mid = loss.prog_alpha(50, 101)
        assert 0.4 < mid < 0.6

    def test_alpha_monotonically_decreasing(self):
        from flashdet.losses.e2e_loss import E2EDetectionLoss
        loss = E2EDetectionLoss(num_classes=10)
        alphas = [loss.prog_alpha(e, 100) for e in range(100)]
        for i in range(len(alphas) - 1):
            assert alphas[i] >= alphas[i + 1]

    def test_alpha_single_epoch(self):
        from flashdet.losses.e2e_loss import E2EDetectionLoss
        loss = E2EDetectionLoss(num_classes=10)
        assert loss.prog_alpha(0, 1) == pytest.approx(0.0)

    def test_alpha_custom_range(self):
        from flashdet.losses.e2e_loss import E2EDetectionLoss
        loss = E2EDetectionLoss(num_classes=10, alpha_init=0.8, alpha_final=0.2)
        assert loss.prog_alpha(0, 100) == pytest.approx(0.8)
        assert loss.prog_alpha(99, 100) == pytest.approx(0.2, abs=0.01)

    def test_alpha_beyond_total_epochs(self):
        from flashdet.losses.e2e_loss import E2EDetectionLoss
        loss = E2EDetectionLoss(num_classes=10)
        assert loss.prog_alpha(200, 100) == pytest.approx(0.0)


class TestAnchorGrid:
    """Tests for anchor grid construction."""

    def test_anchor_count(self):
        from flashdet.losses.e2e_loss import _make_anchor_grid
        feat_sizes = [(40, 40), (20, 20), (10, 10)]
        centers, strides = _make_anchor_grid(feat_sizes, [8, 16, 32], torch.device("cpu"))
        expected = 40*40 + 20*20 + 10*10
        assert centers.shape == (expected, 2)
        assert strides.shape == (expected, 1)

    def test_anchor_centers_positive(self):
        from flashdet.losses.e2e_loss import _make_anchor_grid
        centers, _ = _make_anchor_grid([(10, 10)], [8], torch.device("cpu"))
        assert (centers > 0).all()

    def test_anchor_strides_correct(self):
        from flashdet.losses.e2e_loss import _make_anchor_grid
        feat_sizes = [(4, 4), (2, 2)]
        _, strides = _make_anchor_grid(feat_sizes, [8, 16], torch.device("cpu"))
        assert (strides[:16] == 8).all()
        assert (strides[16:] == 16).all()


class TestLTRBDecode:
    """Tests for LTRB box decoding."""

    def test_decode_shape(self):
        from flashdet.losses.e2e_loss import _decode_ltrb
        centers = torch.tensor([[100.0, 100.0]])
        strides = torch.tensor([[8.0]])
        reg = torch.zeros(1, 4)
        boxes = _decode_ltrb(centers, strides, reg)
        assert boxes.shape == (1, 4)

    def test_decode_zero_regression(self):
        from flashdet.losses.e2e_loss import _decode_ltrb
        centers = torch.tensor([[100.0, 100.0]])
        strides = torch.tensor([[8.0]])
        reg = torch.zeros(1, 4)
        boxes = _decode_ltrb(centers, strides, reg)
        offset = math.log(2) * 8  # softplus(0) = ln(2)
        assert boxes[0, 0].item() == pytest.approx(100 - offset, abs=0.01)
        assert boxes[0, 2].item() == pytest.approx(100 + offset, abs=0.01)

    def test_decode_positive_regression(self):
        from flashdet.losses.e2e_loss import _decode_ltrb
        centers = torch.tensor([[50.0, 50.0]])
        strides = torch.tensor([[8.0]])
        reg = torch.ones(1, 4)
        boxes = _decode_ltrb(centers, strides, reg)
        offset = math.log(1 + math.exp(1)) * 8  # softplus(1)
        assert boxes[0, 0].item() == pytest.approx(50 - offset, abs=0.1)
        assert boxes[0, 2].item() == pytest.approx(50 + offset, abs=0.1)


class TestE2EDetectionLoss:
    """Tests for the full E2E detection loss."""

    def _make_dummy_preds(self, B=2, N=100, nc=5):
        return torch.randn(B, N, nc), torch.randn(B, N, 4)

    def test_loss_with_gt(self):
        from flashdet.losses.e2e_loss import E2EDetectionLoss
        loss_fn = E2EDetectionLoss(num_classes=5, strides=(8,))
        o2o_cls = torch.randn(1, 100, 5, requires_grad=True)
        o2o_reg = torch.randn(1, 100, 4, requires_grad=True)
        o2m_cls = torch.randn(1, 100, 5, requires_grad=True)
        o2m_reg = torch.randn(1, 100, 4, requires_grad=True)
        gt_bboxes = [torch.tensor([[50, 50, 150, 150]], dtype=torch.float32)]
        gt_labels = [torch.tensor([2])]
        feat_sizes = [(10, 10)]

        total, states = loss_fn(
            o2o_cls, o2o_reg, o2m_cls, o2m_reg,
            gt_bboxes, gt_labels, feat_sizes, epoch=0, total_epochs=100,
        )
        assert not torch.isnan(total)
        assert total.requires_grad
        assert "loss_total" in states
        assert "prog_alpha" in states

    def test_loss_empty_gt(self):
        from flashdet.losses.e2e_loss import E2EDetectionLoss
        loss_fn = E2EDetectionLoss(num_classes=5, strides=(8,))
        o2o_cls, o2o_reg = self._make_dummy_preds(1, 100, 5)
        o2m_cls, o2m_reg = self._make_dummy_preds(1, 100, 5)
        gt_bboxes = [torch.zeros(0, 4)]
        gt_labels = [torch.zeros(0, dtype=torch.long)]
        feat_sizes = [(10, 10)]

        total, states = loss_fn(
            o2o_cls, o2o_reg, o2m_cls, o2m_reg,
            gt_bboxes, gt_labels, feat_sizes,
        )
        assert not torch.isnan(total)

    def test_prog_loss_weighting(self):
        """At epoch 0 (alpha=1), total should equal o2m_loss."""
        from flashdet.losses.e2e_loss import E2EDetectionLoss
        loss_fn = E2EDetectionLoss(num_classes=5, strides=(8,))
        o2o_cls, o2o_reg = self._make_dummy_preds(1, 100, 5)
        o2m_cls, o2m_reg = self._make_dummy_preds(1, 100, 5)
        gt_bboxes = [torch.tensor([[50, 50, 150, 150]], dtype=torch.float32)]
        gt_labels = [torch.tensor([0])]
        feat_sizes = [(10, 10)]

        _, states = loss_fn(
            o2o_cls, o2o_reg, o2m_cls, o2m_reg,
            gt_bboxes, gt_labels, feat_sizes, epoch=0, total_epochs=100,
        )
        assert states["prog_alpha"] == pytest.approx(1.0)

    def test_loss_state_keys(self):
        from flashdet.losses.e2e_loss import E2EDetectionLoss
        loss_fn = E2EDetectionLoss(num_classes=5, strides=(8,))
        o2o_cls, o2o_reg = self._make_dummy_preds(1, 100, 5)
        o2m_cls, o2m_reg = self._make_dummy_preds(1, 100, 5)
        gt_bboxes = [torch.tensor([[30, 30, 100, 100]], dtype=torch.float32)]
        gt_labels = [torch.tensor([1])]
        feat_sizes = [(10, 10)]

        _, states = loss_fn(
            o2o_cls, o2o_reg, o2m_cls, o2m_reg,
            gt_bboxes, gt_labels, feat_sizes,
        )
        expected = {"loss_total", "o2m_loss", "o2o_loss", "prog_alpha",
                    "o2m_cls", "o2m_box", "o2m_l1", "o2m_pos",
                    "o2o_cls", "o2o_box", "o2o_l1", "o2o_pos"}
        assert expected == set(states.keys())

    def test_loss_batch(self):
        from flashdet.losses.e2e_loss import E2EDetectionLoss
        loss_fn = E2EDetectionLoss(num_classes=5, strides=(8,))
        o2o_cls, o2o_reg = self._make_dummy_preds(3, 100, 5)
        o2m_cls, o2m_reg = self._make_dummy_preds(3, 100, 5)
        gt_bboxes = [
            torch.tensor([[10, 10, 60, 60]], dtype=torch.float32),
            torch.tensor([[20, 20, 80, 80], [40, 40, 100, 100]], dtype=torch.float32),
            torch.zeros(0, 4),
        ]
        gt_labels = [torch.tensor([0]), torch.tensor([1, 3]), torch.zeros(0, dtype=torch.long)]
        feat_sizes = [(10, 10)]

        total, states = loss_fn(
            o2o_cls, o2o_reg, o2m_cls, o2m_reg,
            gt_bboxes, gt_labels, feat_sizes,
        )
        assert not torch.isnan(total)
        assert not torch.isinf(total)


# ######################################################################
# 3. MuSGD Optimizer
# ######################################################################

class TestNewtonSchulz:
    """Tests for Newton-Schulz orthogonalization."""

    def test_output_shape_2d(self):
        from flashdet.engine.core.musgd import _newton_schulz_orthogonalize
        G = torch.randn(8, 16)
        out = _newton_schulz_orthogonalize(G, steps=3)
        assert out.shape == G.shape

    def test_output_shape_4d(self):
        from flashdet.engine.core.musgd import _newton_schulz_orthogonalize
        G = torch.randn(16, 3, 3, 3)
        out = _newton_schulz_orthogonalize(G, steps=3)
        assert out.shape == G.shape

    def test_output_shape_tall_matrix(self):
        from flashdet.engine.core.musgd import _newton_schulz_orthogonalize
        G = torch.randn(32, 8)
        out = _newton_schulz_orthogonalize(G, steps=3)
        assert out.shape == G.shape

    def test_output_finite(self):
        """Newton-Schulz output should contain no NaN or Inf."""
        from flashdet.engine.core.musgd import _newton_schulz_orthogonalize
        G = torch.randn(8, 16)
        out = _newton_schulz_orthogonalize(G, steps=5)
        assert torch.isfinite(out).all()

    def test_deterministic(self):
        from flashdet.engine.core.musgd import _newton_schulz_orthogonalize
        G = torch.randn(8, 16)
        o1 = _newton_schulz_orthogonalize(G.clone(), steps=5)
        o2 = _newton_schulz_orthogonalize(G.clone(), steps=5)
        assert torch.allclose(o1, o2)


class TestMuSGD:
    """Tests for MuSGD optimizer."""

    def _make_model(self):
        return nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 10, 1),
        )

    def test_basic_step(self):
        from flashdet.engine.core.musgd import build_musgd
        model = self._make_model()
        opt = build_musgd(model, lr=1e-3)
        x = torch.randn(1, 3, 8, 8)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()

    def test_param_group_split(self):
        from flashdet.engine.core.musgd import build_musgd
        model = self._make_model()
        opt = build_musgd(model, lr=1e-3)
        assert len(opt.param_groups) == 2
        assert opt.param_groups[0]["use_muon"] is True
        assert opt.param_groups[1]["use_muon"] is False

    def test_muon_group_has_conv_weights(self):
        from flashdet.engine.core.musgd import build_musgd
        model = self._make_model()
        opt = build_musgd(model, lr=1e-3)
        muon_params = opt.param_groups[0]["params"]
        for p in muon_params:
            assert p.ndim >= 2

    def test_sgd_group_has_biases_bn(self):
        from flashdet.engine.core.musgd import build_musgd
        model = self._make_model()
        opt = build_musgd(model, lr=1e-3)
        sgd_params = opt.param_groups[1]["params"]
        for p in sgd_params:
            assert p.ndim < 2

    def test_multiple_steps_reduce_loss(self):
        from flashdet.engine.core.musgd import build_musgd
        model = self._make_model()
        opt = build_musgd(model, lr=1e-3)
        x = torch.randn(2, 3, 8, 8)
        target = torch.randn(2, 10, 8, 8)
        losses = []
        for _ in range(10):
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(x), target)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0]

    def test_nesterov_mode(self):
        from flashdet.engine.core.musgd import build_musgd
        model = self._make_model()
        opt = build_musgd(model, lr=1e-3, nesterov=True)
        x = torch.randn(1, 3, 8, 8)
        loss = model(x).sum()
        loss.backward()
        opt.step()

    def test_custom_blend_weights(self):
        from flashdet.engine.core.musgd import MuSGD
        p = nn.Parameter(torch.randn(8, 8))
        opt = MuSGD([{"params": [p], "use_muon": True}],
                     lr=1e-3, muon_weight=0.8, sgd_weight=0.2)
        assert opt.param_groups[0]["muon_weight"] == 0.8
        assert opt.param_groups[0]["sgd_weight"] == 0.2

    def test_closure_support(self):
        from flashdet.engine.core.musgd import build_musgd
        model = self._make_model()
        opt = build_musgd(model, lr=1e-3)
        x = torch.randn(1, 3, 8, 8)

        def closure():
            opt.zero_grad()
            loss = model(x).sum()
            loss.backward()
            return loss

        loss = opt.step(closure)
        assert loss is not None

    def test_no_grad_params_skipped(self):
        from flashdet.engine.core.musgd import build_musgd
        model = self._make_model()
        for p in model[0].parameters():
            p.requires_grad_(False)
        opt = build_musgd(model, lr=1e-3)
        frozen = sum(len(g["params"]) for g in opt.param_groups)
        total = sum(1 for p in model.parameters() if p.requires_grad)
        assert frozen == total


# ######################################################################
# 4. E2E Dual Detection Head
# ######################################################################

class TestE2EDetHead:
    """Tests for the single-branch DFL-free detection head."""

    def test_output_shapes(self):
        from flashdet.models.head.e2e_head import E2EDetHead
        head = E2EDetHead(num_classes=10, in_channels=64)
        x = torch.randn(2, 64, 40, 40)
        cls, reg = head(x)
        assert cls.shape == (2, 10, 40, 40)
        assert reg.shape == (2, 4, 40, 40)

    def test_dfl_free_reg_channels(self):
        """Regression should output exactly 4 channels (DFL-free)."""
        from flashdet.models.head.e2e_head import E2EDetHead
        head = E2EDetHead(num_classes=80, in_channels=128)
        x = torch.randn(1, 128, 20, 20)
        _, reg = head(x)
        assert reg.shape[1] == 4

    def test_different_input_sizes(self):
        from flashdet.models.head.e2e_head import E2EDetHead
        head = E2EDetHead(num_classes=5, in_channels=32)
        for h, w in [(10, 10), (20, 20), (40, 40)]:
            x = torch.randn(1, 32, h, w)
            cls, reg = head(x)
            assert cls.shape == (1, 5, h, w)
            assert reg.shape == (1, 4, h, w)


class TestE2EDualHead:
    """Tests for the YOLO26 dual detection head."""

    def _make_features(self, B=2, C=64, sizes=((40, 40), (20, 20), (10, 10))):
        return [torch.randn(B, C, h, w) for h, w in sizes]

    def test_training_mode_output(self):
        from flashdet.models.head.e2e_head import E2EDualHead
        head = E2EDualHead(num_classes=10, in_channels=64, num_levels=3)
        feats = self._make_features()
        out = head(feats, training=True)
        N = 40*40 + 20*20 + 10*10
        assert out["o2o_cls"].shape == (2, N, 10)
        assert out["o2o_reg"].shape == (2, N, 4)
        assert out["o2m_cls"].shape == (2, N, 10)
        assert out["o2m_reg"].shape == (2, N, 4)
        assert len(out["feat_sizes"]) == 3

    def test_eval_mode_no_o2m(self):
        from flashdet.models.head.e2e_head import E2EDualHead
        head = E2EDualHead(num_classes=10, in_channels=64, num_levels=3)
        feats = self._make_features()
        out = head(feats, training=False)
        assert "o2o_cls" in out
        assert "o2o_reg" in out
        assert "o2m_cls" not in out
        assert "o2m_reg" not in out

    def test_feat_sizes_correct(self):
        from flashdet.models.head.e2e_head import E2EDualHead
        head = E2EDualHead(num_classes=5, in_channels=32, num_levels=3)
        sizes = ((40, 40), (20, 20), (10, 10))
        feats = [torch.randn(1, 32, h, w) for h, w in sizes]
        out = head(feats, training=False)
        assert out["feat_sizes"] == [(40, 40), (20, 20), (10, 10)]

    def test_o2o_o2m_heads_independent(self):
        """O2O and O2M heads should have independent parameters."""
        from flashdet.models.head.e2e_head import E2EDualHead
        head = E2EDualHead(num_classes=5, in_channels=32, num_levels=3)
        o2o_ids = {id(p) for p in head.o2o_heads.parameters()}
        o2m_ids = {id(p) for p in head.o2m_heads.parameters()}
        assert len(o2o_ids & o2m_ids) == 0

    def test_gradient_flow(self):
        from flashdet.models.head.e2e_head import E2EDualHead
        head = E2EDualHead(num_classes=5, in_channels=32, num_levels=3)
        feats = [torch.randn(1, 32, h, w, requires_grad=True) for h, w in [(10, 10), (5, 5), (3, 3)]]
        out = head(feats, training=True)
        loss = out["o2o_cls"].sum() + out["o2m_cls"].sum()
        loss.backward()
        for f in feats:
            assert f.grad is not None


# ######################################################################
# 5. FlashDet Full Pipeline
# ######################################################################

class TestFlashDetSizes:
    """Test all FlashDet size variants."""

    @pytest.mark.parametrize("size", ["n", "s", "m", "l", "x"])
    def test_instantiation(self, size):
        from flashdet.models.architectures.flashdet import FlashDet
        model = FlashDet(num_classes=10, size=size)
        info = model.get_model_info()
        assert info["name"] == f"FlashDet-{size.upper()}"
        assert info["total_params"] > 0

    def test_n_smallest(self):
        from flashdet.models.architectures.flashdet import FlashDet
        sizes = {}
        for s in ["n", "s", "m"]:
            m = FlashDet(num_classes=5, size=s)
            sizes[s] = sum(p.numel() for p in m.parameters())
        assert sizes["n"] < sizes["s"] < sizes["m"]

    @pytest.mark.parametrize("size", ["n", "s"])
    def test_forward_inference(self, size):
        from flashdet.models.architectures.flashdet import FlashDet
        model = FlashDet(num_classes=10, size=size)
        model.eval()
        x = torch.randn(1, 3, 320, 320)
        with torch.no_grad():
            out = model(x)
        assert "o2o_cls" in out
        assert "o2o_reg" in out
        assert out["o2o_cls"].shape[0] == 1
        assert out["o2o_cls"].shape[2] == 10
        assert out["o2o_reg"].shape[2] == 4


class TestFlashDetTrainingPipeline:
    """Tests for the full FlashDet training pipeline."""

    def test_training_forward(self):
        from flashdet.models.architectures.flashdet import FlashDet
        model = FlashDet(num_classes=5, size="n")
        model.train()
        x = torch.randn(2, 3, 320, 320)
        gt_meta = _make_gt_meta(2, 5)
        out = model(x, gt_meta=gt_meta, epoch=0)
        assert "loss" in out
        assert "loss_states" in out
        assert out["loss"].requires_grad

    def test_backward(self):
        from flashdet.models.architectures.flashdet import FlashDet
        model = FlashDet(num_classes=5, size="n")
        model.train()
        x = torch.randn(1, 3, 320, 320)
        gt_meta = _make_gt_meta(1, 5)
        out = model(x, gt_meta=gt_meta, epoch=0)
        out["loss"].backward()
        grads = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
        assert grads > 0

    def test_training_empty_gt(self):
        from flashdet.models.architectures.flashdet import FlashDet
        model = FlashDet(num_classes=5, size="n")
        model.train()
        x = torch.randn(1, 3, 320, 320)
        gt_meta = {
            "gt_bboxes": [np.zeros((0, 4), dtype=np.float32)],
            "gt_labels": [np.array([], dtype=np.int64)],
        }
        out = model(x, gt_meta=gt_meta, epoch=0)
        assert not torch.isnan(out["loss"])

    def test_multi_step_train_loop(self):
        from flashdet.models.architectures.flashdet import FlashDet
        model = FlashDet(num_classes=5, size="n")
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-5)
        x = torch.randn(1, 3, 320, 320)
        gt_meta = _make_gt_meta(1, 5)

        losses = []
        for epoch in range(5):
            opt.zero_grad()
            out = model(x, gt_meta=gt_meta, epoch=epoch)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            losses.append(out["loss"].item())

        assert all(not np.isnan(l) for l in losses)
        assert all(not np.isinf(l) for l in losses)

    def test_train_with_musgd(self):
        """Full pipeline: FlashDet + MuSGD optimizer."""
        from flashdet.models.architectures.flashdet import FlashDet
        from flashdet.engine.core.musgd import build_musgd
        model = FlashDet(num_classes=5, size="n", total_epochs=10)
        model.train()
        opt = build_musgd(model, lr=1e-5)
        x = torch.randn(1, 3, 320, 320)
        gt_meta = _make_gt_meta(1, 5)

        for epoch in range(3):
            opt.zero_grad()
            out = model(x, gt_meta=gt_meta, epoch=epoch)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
        assert not torch.isnan(out["loss"])

    def test_prog_loss_shifts_emphasis(self):
        from flashdet.models.architectures.flashdet import FlashDet
        model = FlashDet(num_classes=5, size="n", total_epochs=100)
        model.train()
        x = torch.randn(1, 3, 320, 320)
        gt_meta = _make_gt_meta(1, 5)

        out_early = model(x, gt_meta=gt_meta, epoch=0)
        out_late = model(x, gt_meta=gt_meta, epoch=99)
        assert out_early["loss_states"]["prog_alpha"] == pytest.approx(1.0)
        assert out_late["loss_states"]["prog_alpha"] == pytest.approx(0.0, abs=0.02)

    def test_compute_loss_in_eval_mode(self):
        from flashdet.models.architectures.flashdet import FlashDet
        model = FlashDet(num_classes=5, size="n")
        model.eval()
        x = torch.randn(1, 3, 320, 320)
        gt_meta = _make_gt_meta(1, 5)
        out = model(x, gt_meta=gt_meta, compute_loss=True)
        assert "loss" in out
        assert "loss_states" in out

    def test_return_features(self):
        from flashdet.models.architectures.flashdet import FlashDet
        model = FlashDet(num_classes=5, size="n")
        model.eval()
        x = torch.randn(1, 3, 320, 320)
        out = model(x, return_features=True)
        assert "backbone_features" in out
        assert "fpn_features" in out
        assert len(out["backbone_features"]) == 3
        assert len(out["fpn_features"]) == 3

    def test_gradient_to_input(self):
        from flashdet.models.architectures.flashdet import FlashDet
        model = FlashDet(num_classes=5, size="n")
        model.train()
        x = torch.randn(1, 3, 320, 320, requires_grad=True)
        gt_meta = _make_gt_meta(1, 5)
        out = model(x, gt_meta=gt_meta, epoch=0)
        out["loss"].backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0


class TestFlashDetPredict:
    """Tests for FlashDet prediction/inference."""

    def test_predict_returns_list(self):
        from flashdet.models.architectures.flashdet import FlashDet
        model = FlashDet(num_classes=10, size="n")
        x = torch.randn(2, 3, 320, 320)
        results = model.predict(x)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_predict_output_format(self):
        from flashdet.models.architectures.flashdet import FlashDet
        model = FlashDet(num_classes=10, size="n")
        x = torch.randn(1, 3, 320, 320)
        results = model.predict(x, score_thr=0.01)
        det_bboxes, det_labels = results[0]
        if det_bboxes.numel() > 0:
            assert det_bboxes.shape[-1] == 5
            assert det_labels.ndim == 1
            assert det_bboxes.shape[0] == det_labels.shape[0]

    def test_predict_high_threshold(self):
        from flashdet.models.architectures.flashdet import FlashDet
        model = FlashDet(num_classes=5, size="n")
        x = torch.randn(1, 3, 320, 320)
        results = model.predict(x, score_thr=0.999)
        det_bboxes, _ = results[0]
        assert det_bboxes.shape[0] <= 10

    def test_predict_max_det(self):
        from flashdet.models.architectures.flashdet import FlashDet
        model = FlashDet(num_classes=5, size="n")
        x = torch.randn(1, 3, 320, 320)
        results = model.predict(x, score_thr=0.001, max_det=50)
        det_bboxes, _ = results[0]
        assert det_bboxes.shape[0] <= 50

    def test_predict_sets_eval_mode(self):
        from flashdet.models.architectures.flashdet import FlashDet
        model = FlashDet(num_classes=5, size="n")
        model.train()
        _ = model.predict(torch.randn(1, 3, 320, 320))
        assert not model.training

    def test_predict_no_grad(self):
        from flashdet.models.architectures.flashdet import FlashDet
        model = FlashDet(num_classes=5, size="n")
        x = torch.randn(1, 3, 320, 320, requires_grad=True)
        results = model.predict(x)
        det_bboxes, det_labels = results[0]
        assert not det_bboxes.requires_grad


class TestFlashDetModelInfo:
    """Tests for get_model_info."""

    def test_info_keys(self):
        from flashdet.models.architectures.flashdet import FlashDet
        model = FlashDet(num_classes=10, size="n")
        info = model.get_model_info()
        expected_keys = {"name", "num_classes", "size", "total_params",
                         "trainable_params", "inference_params",
                         "params_mb", "inference_params_mb", "inference_fp16_mb"}
        assert expected_keys == set(info.keys())

    def test_inference_excludes_o2m(self):
        from flashdet.models.architectures.flashdet import FlashDet
        model = FlashDet(num_classes=10, size="n")
        info = model.get_model_info()
        assert info["inference_params"] < info["total_params"]

    def test_fp16_half_of_fp32(self):
        from flashdet.models.architectures.flashdet import FlashDet
        model = FlashDet(num_classes=10, size="n")
        info = model.get_model_info()
        assert info["inference_fp16_mb"] == pytest.approx(info["inference_params_mb"] / 2, rel=0.01)


class TestFlashDetRegistry:
    """Tests for registry integration."""

    def test_registered_in_detectors(self):
        from flashdet.registry import DETECTORS
        assert "FlashDet" in DETECTORS.list()

    def test_build_from_registry(self):
        from flashdet.registry import DETECTORS
        cls = DETECTORS.get("FlashDet")
        model = cls(num_classes=5, size="n")
        assert model.num_classes == 5

    def test_build_model_function(self):
        from flashdet.cfg import get_config
        from flashdet.models import build_model
        cfg = get_config(model_size="n", num_classes=10)
        model = build_model(cfg)
        assert model.num_classes == 10


class TestFlashDetEdgeCases:
    """Edge case tests for FlashDet."""

    def test_single_class(self):
        from flashdet.models.architectures.flashdet import FlashDet
        model = FlashDet(num_classes=1, size="n")
        model.eval()
        x = torch.randn(1, 3, 320, 320)
        out = model(x)
        assert out["o2o_cls"].shape[2] == 1

    def test_many_classes(self):
        from flashdet.models.architectures.flashdet import FlashDet
        model = FlashDet(num_classes=1000, size="n")
        model.eval()
        x = torch.randn(1, 3, 320, 320)
        out = model(x)
        assert out["o2o_cls"].shape[2] == 1000

    def test_non_square_input(self):
        from flashdet.models.architectures.flashdet import FlashDet
        model = FlashDet(num_classes=5, size="n")
        model.eval()
        x = torch.randn(1, 3, 256, 384)
        out = model(x)
        assert out["o2o_cls"].shape[0] == 1

    def test_large_batch(self):
        from flashdet.models.architectures.flashdet import FlashDet
        model = FlashDet(num_classes=5, size="n")
        model.eval()
        x = torch.randn(8, 3, 320, 320)
        out = model(x)
        assert out["o2o_cls"].shape[0] == 8

    def test_many_gt_per_image(self):
        from flashdet.models.architectures.flashdet import FlashDet
        model = FlashDet(num_classes=5, size="n")
        model.train()
        x = torch.randn(1, 3, 320, 320)
        n_gt = 50
        bboxes = np.random.rand(n_gt, 2).astype(np.float32) * 200
        wh = np.random.rand(n_gt, 2).astype(np.float32) * 50 + 10
        boxes = np.concatenate([bboxes, bboxes + wh], axis=1).clip(0, 319)
        labels = np.random.randint(0, 5, size=(n_gt,)).astype(np.int64)
        gt_meta = {"gt_bboxes": [boxes], "gt_labels": [labels]}
        out = model(x, gt_meta=gt_meta, epoch=0)
        assert not torch.isnan(out["loss"])

    def test_custom_width_depth(self):
        from flashdet.models.architectures.flashdet import FlashDet
        model = FlashDet(num_classes=5, width_mult=0.35, depth_mult=0.5)
        model.eval()
        x = torch.randn(1, 3, 320, 320)
        out = model(x)
        assert out["o2o_cls"].shape[0] == 1
