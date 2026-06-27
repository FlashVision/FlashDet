"""Color histogram appearance matching (no deep learning required).

Computes HSV color histograms for cropped bounding-box regions and
measures similarity via histogram correlation or Bhattacharyya distance.
Useful as a lightweight appearance cue when a ReID model is unavailable.
"""

from __future__ import annotations


import cv2
import numpy as np


def extract_histograms(
    frame: np.ndarray,
    boxes: np.ndarray,
    bins: tuple = (16, 16, 8),
    ranges: tuple = (0, 180, 0, 256, 0, 256),
) -> np.ndarray:
    """Extract normalised HSV histograms for each bounding box.

    Parameters
    ----------
    frame : np.ndarray
        BGR image (H, W, 3).
    boxes : np.ndarray, shape (N, 4+)
        Bounding boxes in [x1, y1, x2, y2, ...] format.
    bins : tuple
        Histogram bins per channel (H, S, V).
    ranges : tuple
        Value ranges per channel.

    Returns
    -------
    np.ndarray, shape (N, prod(bins))
        Flattened, L2-normalised histograms.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, w = frame.shape[:2]
    n_bins = int(np.prod(bins))
    histograms = np.zeros((len(boxes), n_bins), dtype=np.float32)

    for i, box in enumerate(boxes):
        x1 = max(0, int(box[0]))
        y1 = max(0, int(box[1]))
        x2 = min(w, int(box[2]))
        y2 = min(h, int(box[3]))
        if x2 <= x1 or y2 <= y1:
            continue

        roi = hsv[y1:y2, x1:x2]
        hist = cv2.calcHist(
            [roi], [0, 1, 2], None,
            list(bins), list(ranges),
        )
        hist = hist.flatten().astype(np.float32)
        norm = np.linalg.norm(hist) + 1e-6
        histograms[i] = hist / norm

    return histograms


def histogram_distance(
    hist_a: np.ndarray,
    hist_b: np.ndarray,
    method: str = "correlation",
) -> np.ndarray:
    """Pairwise distance between two sets of histograms.

    Parameters
    ----------
    hist_a : np.ndarray, shape (N, D)
    hist_b : np.ndarray, shape (M, D)
    method : str
        ``"correlation"`` — 1 - correlation (0 = identical, 2 = opposite).
        ``"bhattacharyya"`` — Bhattacharyya distance (0 = identical).
        ``"cosine"`` — 1 - cosine similarity.

    Returns
    -------
    np.ndarray, shape (N, M)
    """
    n, m = len(hist_a), len(hist_b)
    cost = np.ones((n, m), dtype=np.float64)

    if method == "cosine":
        a_norm = hist_a / (np.linalg.norm(hist_a, axis=1, keepdims=True) + 1e-6)
        b_norm = hist_b / (np.linalg.norm(hist_b, axis=1, keepdims=True) + 1e-6)
        return 1.0 - (a_norm @ b_norm.T).astype(np.float64)

    cv_method = {
        "correlation": cv2.HISTCMP_CORREL,
        "bhattacharyya": cv2.HISTCMP_BHATTACHARYYA,
    }.get(method, cv2.HISTCMP_CORREL)

    for i in range(n):
        for j in range(m):
            sim = cv2.compareHist(
                hist_a[i].astype(np.float32),
                hist_b[j].astype(np.float32),
                cv_method,
            )
            if method == "correlation":
                cost[i, j] = 1.0 - sim  # correlation: 1=identical → cost 0
            else:
                cost[i, j] = sim  # bhattacharyya: 0=identical

    return cost
