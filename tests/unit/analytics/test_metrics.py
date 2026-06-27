"""Tests for DetectionMetrics — mAP, AP, precision, recall, F1."""

import numpy as np
import pytest

from flashdet.analytics.metrics import DetectionMetrics


class TestDetectionMetrics:
    """Core metrics computation."""

    @pytest.fixture
    def perfect_metrics(self):
        """Perfect predictions — all GT matched exactly."""
        m = DetectionMetrics(num_classes=3, class_names=["cat", "dog", "car"])
        m.add_ground_truths(0, np.array([[10, 10, 50, 50], [60, 60, 100, 100]]), np.array([0, 1]))
        m.add_predictions(0, np.array([[10, 10, 50, 50], [60, 60, 100, 100]]), np.array([0.95, 0.90]), np.array([0, 1]))
        return m

    @pytest.fixture
    def imperfect_metrics(self):
        """Some misses and false positives."""
        m = DetectionMetrics(num_classes=3, class_names=["cat", "dog", "car"])
        m.add_ground_truths(0, np.array([[10, 10, 50, 50], [60, 60, 100, 100], [200, 200, 250, 250]]), np.array([0, 1, 2]))
        m.add_predictions(0, np.array([[12, 12, 48, 48], [300, 300, 350, 350]]), np.array([0.9, 0.8]), np.array([0, 1]))
        return m

    def test_perfect_map50_is_1(self, perfect_metrics):
        result = perfect_metrics.compute()
        assert result["mAP_50"] == 1.0

    def test_imperfect_map_less_than_1(self, imperfect_metrics):
        result = imperfect_metrics.compute()
        assert result["mAP_50"] < 1.0

    def test_no_predictions_map_is_0(self):
        m = DetectionMetrics(num_classes=2)
        m.add_ground_truths(0, np.array([[10, 10, 50, 50]]), np.array([0]))
        result = m.compute()
        assert result["mAP"] == 0.0

    def test_per_class_results(self, perfect_metrics):
        result = perfect_metrics.compute()
        assert "per_class" in result
        assert len(result["per_class"]) > 0
        for cls in result["per_class"]:
            assert "AP" in cls
            assert "precision" in cls
            assert "recall" in cls
            assert "f1" in cls

    def test_summary_string(self, perfect_metrics):
        summary = perfect_metrics.summary()
        assert "mAP" in summary
        assert "cat" in summary

    def test_reset_clears_data(self, perfect_metrics):
        perfect_metrics.reset()
        result = perfect_metrics.compute()
        assert result["num_predictions"] == 0
        assert result["num_ground_truths"] == 0

    def test_multiple_images(self):
        m = DetectionMetrics(num_classes=2)
        for img_id in range(5):
            gt_box = np.array([[img_id * 10, 10, img_id * 10 + 40, 50]])
            m.add_ground_truths(img_id, gt_box, np.array([0]))
            pred_box = gt_box + np.array([[2, 2, 2, 2]])
            m.add_predictions(img_id, pred_box, np.array([0.9]), np.array([0]))
        result = m.compute()
        assert result["mAP_50"] > 0.5

    def test_size_stratified_metrics(self, perfect_metrics):
        result = perfect_metrics.compute()
        assert "mAP_small" in result
        assert "mAP_medium" in result
        assert "mAP_large" in result

    @pytest.mark.parametrize("nc", [1, 5, 20, 80])
    def test_various_class_counts(self, nc):
        m = DetectionMetrics(num_classes=nc)
        gt_box = np.array([[10, 10, 50, 50]])
        m.add_ground_truths(0, gt_box, np.array([0]))
        m.add_predictions(0, gt_box, np.array([0.9]), np.array([0]))
        result = m.compute()
        assert result["mAP_50"] > 0.0
