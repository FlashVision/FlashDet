"""Tests for DetectionErrorAnalyzer — TIDE-style error categorization."""

import numpy as np
import pytest

from flashdet.analytics.detection_analysis import DetectionErrorAnalyzer


class TestDetectionErrorAnalyzer:
    """Error analysis tests."""

    def test_perfect_predictions_no_errors(self):
        ea = DetectionErrorAnalyzer(num_classes=2)
        ea.add_ground_truths(0, np.array([[10, 10, 50, 50]]), np.array([0]))
        ea.add_predictions(0, np.array([[10, 10, 50, 50]]), np.array([0.9]), np.array([0]))
        result = ea.analyze()
        assert result["total_errors"] == 0

    def test_classification_error(self):
        ea = DetectionErrorAnalyzer(num_classes=3, class_names=["cat", "dog", "car"])
        ea.add_ground_truths(0, np.array([[10, 10, 50, 50]]), np.array([0]))
        ea.add_predictions(0, np.array([[10, 10, 50, 50]]), np.array([0.9]), np.array([1]))  # wrong class
        result = ea.analyze()
        assert result["summary"].get("classification", 0) > 0

    def test_localization_error(self):
        ea = DetectionErrorAnalyzer(num_classes=2)
        ea.add_ground_truths(0, np.array([[10, 10, 50, 50]]), np.array([0]))
        ea.add_predictions(0, np.array([[30, 30, 70, 70]]), np.array([0.9]), np.array([0]))  # shifted
        result = ea.analyze()
        assert result["summary"].get("localization", 0) > 0

    def test_background_error(self):
        ea = DetectionErrorAnalyzer(num_classes=2)
        ea.add_ground_truths(0, np.array([[10, 10, 50, 50]]), np.array([0]))
        ea.add_predictions(0, np.array([[500, 500, 600, 600]]), np.array([0.9]), np.array([0]))
        result = ea.analyze()
        assert result["summary"].get("background", 0) > 0

    def test_missed_detection(self):
        ea = DetectionErrorAnalyzer(num_classes=2)
        ea.add_ground_truths(0, np.array([[10, 10, 50, 50], [200, 200, 300, 300]]), np.array([0, 1]))
        ea.add_predictions(0, np.array([[10, 10, 50, 50]]), np.array([0.9]), np.array([0]))
        result = ea.analyze()
        assert result["summary"].get("missed", 0) == 1

    def test_duplicate_detection(self):
        ea = DetectionErrorAnalyzer(num_classes=2)
        ea.add_ground_truths(0, np.array([[10, 10, 50, 50]]), np.array([0]))
        ea.add_predictions(
            0,
            np.array([[10, 10, 50, 50], [11, 11, 51, 51]]),
            np.array([0.95, 0.85]),
            np.array([0, 0]),
        )
        result = ea.analyze()
        assert result["summary"].get("duplicate", 0) == 1

    def test_confusion_pairs(self):
        ea = DetectionErrorAnalyzer(num_classes=3, class_names=["cat", "dog", "car"])
        for _ in range(5):
            ea.add_ground_truths(0, np.array([[10, 10, 50, 50]]), np.array([0]))
            ea.add_predictions(0, np.array([[10, 10, 50, 50]]), np.array([0.9]), np.array([1]))
        result = ea.analyze()
        assert len(result["confusion_pairs"]) > 0
        assert result["confusion_pairs"][0]["predicted"] == "dog"
        assert result["confusion_pairs"][0]["actual"] == "cat"

    def test_summary_string(self):
        ea = DetectionErrorAnalyzer(num_classes=2)
        ea.add_ground_truths(0, np.array([[10, 10, 50, 50]]), np.array([0]))
        ea.add_predictions(0, np.array([[500, 500, 600, 600]]), np.array([0.9]), np.array([0]))
        summary = ea.summary()
        assert "Error" in summary
        assert "Background" in summary

    def test_reset(self):
        ea = DetectionErrorAnalyzer(num_classes=2)
        ea.add_ground_truths(0, np.array([[10, 10, 50, 50]]), np.array([0]))
        ea.reset()
        result = ea.analyze()
        assert result["total_predictions"] == 0
        assert result["total_ground_truths"] == 0

    def test_score_threshold_filters(self):
        ea = DetectionErrorAnalyzer(num_classes=2, score_threshold=0.5)
        ea.add_ground_truths(0, np.array([[10, 10, 50, 50]]), np.array([0]))
        ea.add_predictions(0, np.array([[500, 500, 600, 600]]), np.array([0.3]), np.array([0]))
        result = ea.analyze()
        assert result["total_predictions"] == 0
