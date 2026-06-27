"""Unit test fixtures — lightweight, no disk I/O."""

import numpy as np
import pytest
import torch


@pytest.fixture
def small_detections():
    """Synthetic Nx6 detection array: [x1, y1, x2, y2, score, class_id]."""
    return np.array([
        [100, 100, 200, 200, 0.95, 0],
        [300, 300, 400, 400, 0.80, 1],
        [150, 150, 250, 250, 0.60, 0],
        [500, 100, 600, 200, 0.45, 2],
    ], dtype=np.float64)


@pytest.fixture
def dummy_config():
    """Minimal FlashDet config for unit tests."""
    from flashdet.cfg import get_config
    return get_config(model_size="n", input_size=320, num_classes=5)
