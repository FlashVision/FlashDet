"""Regression tests for counting/flow/heatmap fixes."""

from collections import deque

import numpy as np
import pytest

from flashdet.solutions.crowd_density import CrowdDensity
from flashdet.solutions.heatmap import Heatmap
from flashdet.solutions.parking_manager import ParkingManager
from flashdet.solutions.speed_estimator import SpeedEstimator
from flashdet.solutions.traffic_flow import TrafficFlow


class _Predictor:
    def __init__(self, detections):
        self.detections = np.asarray(detections, dtype=np.float64)

    def __call__(self, _frame):
        return self.detections


class _IdentityTracker:
    def update(self, detections):
        if len(detections) == 0:
            return np.empty((0, 7), dtype=np.float64)
        # Solutions expect tracker output: [x1,y1,x2,y2,id,score,class].
        return np.column_stack([
            detections[:, :4],
            np.arange(1, len(detections) + 1),
            detections[:, 4],
            detections[:, 5],
        ]).astype(np.float64)


def test_direction_labels_always_match_bin_count():
    for n in (1, 3, 5, 6, 7, 8, 12):
        labels = TrafficFlow._evenly_spaced_labels(n)
        assert len(labels) == n


def test_speed_uses_frame_delta_not_sample_count():
    estimator = SpeedEstimator(_Predictor([]), tracker=_IdentityTracker(), fps=10)
    speed = estimator._compute_speed(
        deque([(0.0, 0.0), (10.0, 0.0)]),
        deque([1, 6]),
    )
    assert speed == pytest.approx(2.0)  # 10 px over 5 frames, not 1 frame


def test_traffic_flow_speed_uses_frame_delta():
    flow = TrafficFlow(
        _Predictor([]), tracker=_IdentityTracker(),
        min_displacement=0.0, history_length=10, fps=10,
    )
    points = deque([(float(i * 10), 0.0) for i in range(5)])
    frames = deque([1, 2, 3, 4, 10])
    _direction, speed, _angle = flow._compute_flow(points, frames)
    assert speed == pytest.approx(40.0 / 9.0)


def test_crowd_density_clamps_negative_cells():
    predictor = _Predictor([[-10, -10, 10, 10, 0.9, 0]])
    crowd = CrowdDensity(
        predictor, tracker=_IdentityTracker(), grid_rows=2, grid_cols=2,
    )
    crowd.process_frame(np.zeros((20, 20, 3), dtype=np.uint8))
    assert crowd._last_grid[0, 0] == 1
    assert crowd._last_grid.sum() == 1


def test_heatmap_reinitialises_on_resolution_change():
    predictor = _Predictor([[5, 5, 10, 10, 0.9, 0]])
    heat = Heatmap(predictor, tracker=_IdentityTracker(), radius=2, decay=0.5)
    heat.process_frame(np.zeros((20, 20, 3), dtype=np.uint8))
    heat.process_frame(np.zeros((30, 30, 3), dtype=np.uint8))
    assert heat._heat.shape == (30, 30)


def test_parking_spot_ids_must_be_contiguous():
    manager = ParkingManager(_Predictor([]), tracker=_IdentityTracker())
    polygon = np.array([[0, 0], [10, 0], [10, 10]], dtype=np.int32)
    with pytest.raises(ValueError, match="contiguously"):
        manager.add_spot(polygon, spot_id=3)
    assert manager.add_spot(polygon) == 0
    assert manager.add_spot(polygon, spot_id=1) == 1
