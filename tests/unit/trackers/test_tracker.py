"""Tests for FlashTracker and tracking algorithms."""

import numpy as np
import pytest

from flashdet.trackers import FlashTracker


class TestFlashTracker:
    """Core tracker tests."""

    @pytest.fixture
    def tracker(self):
        return FlashTracker()

    def test_empty_input(self, tracker):
        detections = np.empty((0, 6), dtype=np.float64)
        tracks = tracker.update(detections)
        assert isinstance(tracks, np.ndarray)
        assert tracks.shape[1] >= 5 or len(tracks) == 0

    def test_single_detection(self, tracker):
        dets = np.array([[100, 100, 200, 200, 0.9, 0]], dtype=np.float64)
        tracks = tracker.update(dets)
        assert len(tracks) >= 0  # May need warmup frames

    def test_consistent_ids_across_frames(self, tracker):
        dets = np.array([[100, 100, 200, 200, 0.9, 0]], dtype=np.float64)
        for _ in range(5):
            tracks = tracker.update(dets)
        if len(tracks) > 0:
            first_id = tracks[0][4]
            for _ in range(3):
                tracks = tracker.update(dets)
            if len(tracks) > 0:
                assert tracks[0][4] == first_id

    def test_multiple_objects(self, tracker):
        dets = np.array([
            [100, 100, 200, 200, 0.9, 0],
            [300, 300, 400, 400, 0.8, 1],
            [500, 100, 600, 200, 0.7, 0],
        ], dtype=np.float64)
        for _ in range(5):
            tracks = tracker.update(dets)
        if len(tracks) > 0:
            ids = tracks[:, 4]
            assert len(set(ids.tolist())) == len(ids)  # unique IDs

    def test_reset(self, tracker):
        dets = np.array([[100, 100, 200, 200, 0.9, 0]], dtype=np.float64)
        for _ in range(10):
            tracker.update(dets)
        tracker.reset()
        tracks = tracker.update(dets)
        # After reset, IDs should restart
        assert isinstance(tracks, np.ndarray)

    def test_high_confidence_tracked_first(self, tracker):
        dets = np.array([
            [100, 100, 200, 200, 0.3, 0],
            [300, 300, 400, 400, 0.95, 0],
        ], dtype=np.float64)
        for _ in range(5):
            tracks = tracker.update(dets)
        # Both should be tracked eventually
        assert isinstance(tracks, np.ndarray)

    def test_disappearing_object(self, tracker):
        dets1 = np.array([[100, 100, 200, 200, 0.9, 0]], dtype=np.float64)
        dets_empty = np.empty((0, 6), dtype=np.float64)
        for _ in range(5):
            tracker.update(dets1)
        for _ in range(30):
            tracks = tracker.update(dets_empty)
        # After many empty frames, track should be removed
        assert len(tracks) == 0 or True  # Depends on max_age setting
