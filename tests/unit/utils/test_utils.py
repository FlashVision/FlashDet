"""Tests for utility functions — bbox decoding, anchor grid."""

import pytest
import torch

from flashdet.utils.bbox import make_anchor_grid, bbox_iou_aligned


class TestMakeAnchorGrid:
    """Anchor grid generation tests."""

    def test_single_level(self):
        feat_sizes = [(10, 10)]
        strides = [8]
        centers, stride_tensor = make_anchor_grid(feat_sizes, strides, device=torch.device("cpu"))
        assert centers.shape == (100, 2)
        assert stride_tensor.shape == (100, 1)
        assert (stride_tensor == 8).all()

    def test_multi_level(self):
        feat_sizes = [(40, 40), (20, 20), (10, 10)]
        strides = [8, 16, 32]
        centers, stride_tensor = make_anchor_grid(feat_sizes, strides, device=torch.device("cpu"))
        total = 40*40 + 20*20 + 10*10  # 2100
        assert centers.shape == (total, 2)
        assert stride_tensor.shape == (total, 1)

    def test_anchor_centers_positive(self):
        feat_sizes = [(20, 20)]
        strides = [16]
        centers, _ = make_anchor_grid(feat_sizes, strides, device=torch.device("cpu"))
        assert (centers > 0).all()

    def test_anchor_centers_spacing(self):
        feat_sizes = [(4, 4)]
        strides = [8]
        centers, _ = make_anchor_grid(feat_sizes, strides, device=torch.device("cpu"))
        # First center should be at 0.5 * stride = 4.0
        assert torch.isclose(centers[0, 0], torch.tensor(4.0))
        assert torch.isclose(centers[0, 1], torch.tensor(4.0))

    def test_total_anchors_matches_feature_sizes(self):
        feat_sizes = [(8, 8), (4, 4), (2, 2)]
        strides = [8, 16, 32]
        centers, strides_t = make_anchor_grid(feat_sizes, strides, device=torch.device("cpu"))
        expected_total = 64 + 16 + 4
        assert centers.shape[0] == expected_total

    def test_stride_values_correct(self):
        feat_sizes = [(4, 4), (2, 2)]
        strides = [16, 32]
        _, stride_tensor = make_anchor_grid(feat_sizes, strides, device=torch.device("cpu"))
        # First 16 should be stride 16, next 4 should be stride 32
        assert (stride_tensor[:16] == 16).all()
        assert (stride_tensor[16:] == 32).all()


class TestBboxIoUAligned:
    """Aligned (element-wise) IoU computation."""

    def test_identical_boxes(self):
        boxes = torch.tensor([[10, 10, 50, 50], [100, 100, 200, 200]], dtype=torch.float32)
        iou = bbox_iou_aligned(boxes, boxes)
        assert torch.allclose(iou, torch.ones(2), atol=1e-4)

    def test_result_shape(self):
        a = torch.rand(10, 4) * 100
        a[:, 2:] = a[:, :2] + 50
        iou = bbox_iou_aligned(a, a)
        assert iou.shape == (10,) or iou.numel() == 10
