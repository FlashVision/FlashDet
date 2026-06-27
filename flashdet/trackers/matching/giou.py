"""Generalized IoU, Distance IoU, and Complete IoU for box matching.

Standard IoU fails when boxes don't overlap.  These variants provide
meaningful gradients/distances even for non-overlapping boxes:

- **GIoU** (Rezatofighi et al., CVPR 2019) — penalises enclosing area.
- **DIoU** (Zheng et al., AAAI 2020) — penalises centre distance.
- **CIoU** (Zheng et al., AAAI 2020) — adds aspect-ratio consistency.
"""

from __future__ import annotations

import numpy as np


def giou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise GIoU between two sets of [x1, y1, x2, y2] boxes.

    Returns shape (N, M), values in [-1, 1].
    """
    x1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0])
    y1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1])
    x2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2])
    y2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter

    iou = np.where(union > 0, inter / union, 0.0)

    # Enclosing box
    ex1 = np.minimum(boxes_a[:, 0:1], boxes_b[:, 0])
    ey1 = np.minimum(boxes_a[:, 1:2], boxes_b[:, 1])
    ex2 = np.maximum(boxes_a[:, 2:3], boxes_b[:, 2])
    ey2 = np.maximum(boxes_a[:, 3:4], boxes_b[:, 3])
    enclose_area = (ex2 - ex1) * (ey2 - ey1)

    giou = iou - np.where(enclose_area > 0, (enclose_area - union) / enclose_area, 0.0)
    return giou


def diou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise DIoU between two sets of [x1, y1, x2, y2] boxes.

    Returns shape (N, M), values in [-1, 1].
    """
    x1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0])
    y1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1])
    x2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2])
    y2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    iou = np.where(union > 0, inter / union, 0.0)

    # Centre distance
    cx_a = (boxes_a[:, 0:1] + boxes_a[:, 2:3]) / 2
    cy_a = (boxes_a[:, 1:2] + boxes_a[:, 3:4]) / 2
    cx_b = (boxes_b[:, 0] + boxes_b[:, 2]) / 2
    cy_b = (boxes_b[:, 1] + boxes_b[:, 3]) / 2
    rho2 = (cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2

    # Enclosing diagonal
    ex1 = np.minimum(boxes_a[:, 0:1], boxes_b[:, 0])
    ey1 = np.minimum(boxes_a[:, 1:2], boxes_b[:, 1])
    ex2 = np.maximum(boxes_a[:, 2:3], boxes_b[:, 2])
    ey2 = np.maximum(boxes_a[:, 3:4], boxes_b[:, 3])
    c2 = (ex2 - ex1) ** 2 + (ey2 - ey1) ** 2

    diou = iou - np.where(c2 > 0, rho2 / c2, 0.0)
    return diou


def ciou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise CIoU between two sets of [x1, y1, x2, y2] boxes.

    Returns shape (N, M), values in [-1, 1].
    """
    x1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0])
    y1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1])
    x2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2])
    y2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    w_a = boxes_a[:, 2] - boxes_a[:, 0]
    h_a = boxes_a[:, 3] - boxes_a[:, 1]
    w_b = boxes_b[:, 2] - boxes_b[:, 0]
    h_b = boxes_b[:, 3] - boxes_b[:, 1]

    area_a = w_a * h_a
    area_b = w_b * h_b
    union = area_a[:, None] + area_b[None, :] - inter
    iou = np.where(union > 0, inter / union, 0.0)

    cx_a = (boxes_a[:, 0:1] + boxes_a[:, 2:3]) / 2
    cy_a = (boxes_a[:, 1:2] + boxes_a[:, 3:4]) / 2
    cx_b = (boxes_b[:, 0] + boxes_b[:, 2]) / 2
    cy_b = (boxes_b[:, 1] + boxes_b[:, 3]) / 2
    rho2 = (cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2

    ex1 = np.minimum(boxes_a[:, 0:1], boxes_b[:, 0])
    ey1 = np.minimum(boxes_a[:, 1:2], boxes_b[:, 1])
    ex2 = np.maximum(boxes_a[:, 2:3], boxes_b[:, 2])
    ey2 = np.maximum(boxes_a[:, 3:4], boxes_b[:, 3])
    c2 = (ex2 - ex1) ** 2 + (ey2 - ey1) ** 2

    # Aspect-ratio consistency
    v = (4.0 / np.pi ** 2) * (
        np.arctan(w_a[:, None] / (h_a[:, None] + 1e-6))
        - np.arctan(w_b[None, :] / (h_b[None, :] + 1e-6))
    ) ** 2
    alpha = np.where(iou > 0.5, v / (1.0 - iou + v + 1e-6), 0.0)

    ciou = iou - np.where(c2 > 0, rho2 / c2, 0.0) - alpha * v
    return ciou
