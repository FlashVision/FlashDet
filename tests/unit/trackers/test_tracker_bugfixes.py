"""Regression tests for tracker ID, KCF and linear predictor fixes."""

import numpy as np

from flashdet.trackers.core.kalman import KalmanFilter
from flashdet.trackers.core.predictors.kcf_predictor import KCFPredictor
from flashdet.trackers.core.predictors.linear_predictor import LinearPredictor
from flashdet.trackers.core.track import Track


def _detection(x=10.0, y=10.0):
    return np.array([x, y, x + 20, y + 20, 0.9, 0], dtype=np.float64)


def test_track_id_counter_reset_applies_to_new_tracks():
    kf = KalmanFilter()
    Track.reset_id_counter()
    first = Track(_detection(), kf)
    Track.reset_id_counter()
    second = Track(_detection(), kf)
    assert first.track_id == 1
    assert second.track_id == 1


def test_kcf_zero_motion_does_not_jump():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (96, 96, 3), dtype=np.uint8)
    kcf = KCFPredictor()
    kcf.set_frame(frame)
    box = np.array([24.0, 24.0, 64.0, 64.0])
    kcf.initiate(box)
    predicted = kcf.predict()
    assert np.allclose(predicted, box, atol=2.0)


def test_linear_predictor_advances_on_repeated_calls():
    predictor = LinearPredictor(history_length=5)
    predictor.initiate(np.array([0, 0, 10, 10], dtype=np.float64))
    predictor.update(np.array([2, 0, 12, 10], dtype=np.float64))

    first = predictor.predict()
    second = predictor.predict()
    assert not np.allclose(first, second)
    assert second[0] > first[0]
