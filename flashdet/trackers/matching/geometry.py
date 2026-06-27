"""Bounding-box coordinate conversions."""

from __future__ import annotations

import numpy as np


def xyxy_to_cxywh(bbox: np.ndarray) -> np.ndarray:
    """Convert [x1, y1, x2, y2] to [cx, cy, w, h]."""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return np.array([bbox[0] + w / 2, bbox[1] + h / 2, w, h], dtype=np.float64)


def cxywh_to_xyxy(cxywh: np.ndarray) -> np.ndarray:
    """Convert [cx, cy, w, h] to [x1, y1, x2, y2]."""
    cx, cy, w, h = cxywh
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float64)
