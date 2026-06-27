"""Tests for all FlashDet solutions — process_frame interface contract."""

import numpy as np
import pytest

from flashdet.solutions._base import BaseSolution, run_detector


class MockPredictor:
    """Fake predictor that returns fixed detections."""

    def __init__(self, detections=None):
        if detections is None:
            detections = np.array([
                [100, 100, 200, 200, 0.9, 0],
                [300, 300, 400, 400, 0.8, 1],
            ], dtype=np.float64)
        self.detections = detections

    def __call__(self, frame):
        return self.detections


class TestRunDetector:
    """Test the universal detector output normalizer."""

    def test_numpy_array_passthrough(self):
        dets = np.array([[10, 10, 50, 50, 0.9, 0, 0.1]], dtype=np.float64)
        predictor = lambda f: dets
        result = run_detector(predictor, np.zeros((100, 100, 3), dtype=np.uint8))
        assert result.shape == (1, 6)

    def test_list_of_tuples_format(self):
        predictor = lambda f: [
            (np.array([10, 10, 50, 50]), 0.9, 0),
            (np.array([60, 60, 100, 100]), 0.8, 1),
        ]
        result = run_detector(predictor, np.zeros((100, 100, 3), dtype=np.uint8))
        assert result.shape == (2, 6)

    def test_list_of_dicts_format(self):
        predictor = lambda f: [
            {"bbox": [10, 10, 50, 50], "confidence": 0.9, "class_id": 0},
        ]
        result = run_detector(predictor, np.zeros((100, 100, 3), dtype=np.uint8))
        assert result.shape == (1, 6)

    def test_empty_list(self):
        predictor = lambda f: []
        result = run_detector(predictor, np.zeros((100, 100, 3), dtype=np.uint8))
        assert result.shape == (0, 6)


class TestObjectCounter:
    """ObjectCounter solution."""

    def test_process_frame_interface(self):
        from flashdet.solutions.object_counter import ObjectCounter
        predictor = MockPredictor()
        counter = ObjectCounter(predictor=predictor, line_points=[(0, 240), (640, 240)])
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated, results = counter.process_frame(frame)
        assert annotated.shape == frame.shape
        assert isinstance(results, dict)

    def test_get_results(self):
        from flashdet.solutions.object_counter import ObjectCounter
        predictor = MockPredictor()
        counter = ObjectCounter(predictor=predictor, line_points=[(0, 240), (640, 240)])
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        counter.process_frame(frame)
        results = counter.get_results()
        assert isinstance(results, dict)

    def test_reset(self):
        from flashdet.solutions.object_counter import ObjectCounter
        predictor = MockPredictor()
        counter = ObjectCounter(predictor=predictor, line_points=[(0, 240), (640, 240)])
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        counter.process_frame(frame)
        counter.reset()
        results = counter.get_results()
        assert isinstance(results, dict)


class TestHeatmap:
    """Heatmap solution."""

    def test_process_frame(self):
        from flashdet.solutions.heatmap import Heatmap
        predictor = MockPredictor()
        hm = Heatmap(predictor=predictor)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated, results = hm.process_frame(frame)
        assert annotated.shape == frame.shape


class TestAnalyticsDashboard:
    """AnalyticsDashboard solution."""

    def test_process_frame(self):
        from flashdet.solutions.analytics_dashboard import AnalyticsDashboard
        predictor = MockPredictor()
        dash = AnalyticsDashboard(predictor=predictor, tracker=None, class_names=["cat", "dog"])
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated, results = dash.process_frame(frame)
        assert annotated.shape == frame.shape
        assert "total_detections" in results

    def test_get_summary_report(self):
        from flashdet.solutions.analytics_dashboard import AnalyticsDashboard
        predictor = MockPredictor()
        dash = AnalyticsDashboard(predictor=predictor, tracker=None)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dash.process_frame(frame)
        report = dash.get_summary_report()
        assert "FlashDet Analytics" in report


class TestSpeedEstimator:
    """SpeedEstimator solution."""

    def test_process_frame(self):
        from flashdet.solutions.speed_estimator import SpeedEstimator
        predictor = MockPredictor()
        se = SpeedEstimator(predictor=predictor)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated, results = se.process_frame(frame)
        assert annotated.shape == frame.shape
