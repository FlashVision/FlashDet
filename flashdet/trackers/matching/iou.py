"""IoU (Intersection over Union) computation for bounding boxes."""

from __future__ import annotations

import numpy as np


def iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of [x1, y1, x2, y2] boxes.

    Parameters
    ----------
    boxes_a : np.ndarray, shape (N, 4+)
    boxes_b : np.ndarray, shape (M, 4+)

    Returns
    -------
    np.ndarray, shape (N, M)
    """
    x1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0])
    y1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1])
    x2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2])
    y2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3])

    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)
