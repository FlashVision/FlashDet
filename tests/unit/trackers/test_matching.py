"""Tests for tracker matching functions — IoU batch computation."""

import numpy as np
import pytest

from flashdet.trackers.matching.iou import iou_batch


class TestIoUBatch:
    """IoU batch distance/cost matrix."""

    def test_identical_boxes_iou_1(self):
        boxes = np.array([[10, 10, 50, 50], [100, 100, 200, 200]], dtype=np.float64)
        iou = iou_batch(boxes, boxes)
        assert iou.shape == (2, 2)
        assert np.allclose(np.diag(iou), 1.0, atol=1e-6)

    def test_non_overlapping_boxes_iou_0(self):
        a = np.array([[0, 0, 10, 10]], dtype=np.float64)
        b = np.array([[100, 100, 200, 200]], dtype=np.float64)
        iou = iou_batch(a, b)
        assert np.isclose(iou[0, 0], 0.0, atol=1e-6)

    def test_partial_overlap(self):
        a = np.array([[0, 0, 100, 100]], dtype=np.float64)
        b = np.array([[50, 50, 150, 150]], dtype=np.float64)
        iou = iou_batch(a, b)
        # intersection = 50*50 = 2500, union = 10000 + 10000 - 2500 = 17500
        expected = 2500 / 17500
        assert np.isclose(iou[0, 0], expected, atol=1e-4)

    def test_output_shape(self):
        a = np.random.rand(5, 4).astype(np.float64) * 100
        a[:, 2:] = a[:, :2] + 50
        b = np.random.rand(3, 4).astype(np.float64) * 100
        b[:, 2:] = b[:, :2] + 50
        iou = iou_batch(a, b)
        assert iou.shape == (5, 3)

    def test_values_in_range(self):
        a = np.random.rand(10, 4).astype(np.float64) * 200
        a[:, 2:] = a[:, :2] + np.random.rand(10, 2) * 50 + 1
        b = np.random.rand(8, 4).astype(np.float64) * 200
        b[:, 2:] = b[:, :2] + np.random.rand(8, 2) * 50 + 1
        iou = iou_batch(a, b)
        assert (iou >= 0).all()
        assert (iou <= 1.0 + 1e-6).all()

    def test_symmetric(self):
        a = np.array([[10, 10, 50, 50], [60, 60, 100, 100]], dtype=np.float64)
        b = np.array([[20, 20, 60, 60], [70, 70, 110, 110]], dtype=np.float64)
        iou_ab = iou_batch(a, b)
        iou_ba = iou_batch(b, a)
        assert np.allclose(iou_ab, iou_ba.T, atol=1e-6)

    def test_contained_box(self):
        a = np.array([[0, 0, 100, 100]], dtype=np.float64)
        b = np.array([[25, 25, 75, 75]], dtype=np.float64)
        iou = iou_batch(a, b)
        # Contained: inter = 50*50 = 2500, union = 10000
        expected = 2500 / 10000
        assert np.isclose(iou[0, 0], expected, atol=1e-4)
