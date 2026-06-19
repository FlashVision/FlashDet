"""Unit tests for trackers."""

import numpy as np
import pytest
from flashdet.trackers import ByteTracker, SORTTracker, BoTSORT


@pytest.fixture
def sample_detections():
    return np.array([
        [100, 100, 200, 200, 0.9, 0],
        [300, 300, 400, 400, 0.8, 1],
    ], dtype=np.float32)


class TestByteTracker:
    def test_empty_input(self):
        tracker = ByteTracker()
        tracks = tracker.update(np.empty((0, 6), dtype=np.float32))
        assert tracks.shape == (0, 7)

    def test_track_assignment(self, sample_detections):
        tracker = ByteTracker(min_hits=1)
        for _ in range(3):
            tracks = tracker.update(sample_detections)
        assert tracks.shape[0] >= 1
        assert tracks.shape[1] == 7

    def test_reset(self, sample_detections):
        tracker = ByteTracker(min_hits=1)
        for _ in range(5):
            tracker.update(sample_detections)
        tracker.reset()
        tracks = tracker.update(sample_detections)
        assert tracks.shape[0] == 0 or tracks[0, 4] == 1  # new ID after reset


class TestSORTTracker:
    def test_empty_input(self):
        tracker = SORTTracker()
        tracks = tracker.update(np.empty((0, 6), dtype=np.float32))
        assert tracks.shape == (0, 7)

    def test_track_persistence(self, sample_detections):
        tracker = SORTTracker(min_hits=1)
        for _ in range(3):
            tracks = tracker.update(sample_detections)
        assert tracks.shape[0] >= 1


class TestBoTSORT:
    def test_empty_input(self):
        tracker = BoTSORT()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tracks = tracker.update(np.empty((0, 6), dtype=np.float32), frame)
        assert tracks.shape == (0, 7)
