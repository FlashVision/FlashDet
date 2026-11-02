"""Accuracy-regression tests for mAP matching semantics."""

import numpy as np
import pytest

from flashdet.utils.metrics import compute_map


def test_duplicate_detection_cannot_steal_another_gt():
    gt = {
        "boxes": np.array([[0, 0, 10, 10], [0, 0, 12, 12]], dtype=np.float32),
        "labels": np.array([0, 0], dtype=np.int64),
    }
    # Both detections are the first GT. The duplicate overlaps the second GT
    # enough to match under the old matcher, but COCO says it is a false
    # positive once its best GT is already taken.
    pred = {
        "boxes": np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32),
        "scores": np.array([0.9, 0.8], dtype=np.float32),
        "labels": np.array([0, 0], dtype=np.int64),
    }
    result = compute_map([pred], [gt], iou_threshold=0.5, num_classes=1)
    assert result["mAP"] == pytest.approx(0.5)


def test_plain_python_lists_are_supported():
    pred = {
        "boxes": [[0, 0, 10, 10]],
        "scores": [0.9],
        "labels": [0],
    }
    gt = {
        "boxes": [[0, 0, 10, 10]],
        "labels": [0],
    }
    result = compute_map([pred], [gt], iou_threshold=0.5, num_classes=1)
    assert result["mAP"] == pytest.approx(1.0)


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        compute_map([], [{"boxes": [], "labels": []}], num_classes=1)
