"""Hungarian (linear sum) assignment with threshold gating.

Matches detection bounding boxes to tracking bounding boxes using the
Hungarian algorithm, then separates results into matched pairs and
unmatched detections/tracks based on a cost threshold.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def linear_assignment(
    cost_matrix: np.ndarray,
    threshold: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Run the Hungarian algorithm and split results by cost threshold.

    Parameters
    ----------
    cost_matrix : np.ndarray, shape (N, M)
        Cost values (lower is better).  Rows = detections, cols = tracks.
    threshold : float
        Maximum cost for a valid match.

    Returns
    -------
    matched : list of (row, col) pairs
    unmatched_rows : list of int
    unmatched_cols : list of int
    """
    if cost_matrix.size == 0:
        return (
            [],
            list(range(cost_matrix.shape[0])),
            list(range(cost_matrix.shape[1])),
        )

    from scipy.optimize import linear_sum_assignment as scipy_lsa

    row_idx, col_idx = scipy_lsa(cost_matrix)

    matched, unmatched_rows, unmatched_cols = [], [], []
    row_set = set(range(cost_matrix.shape[0]))
    col_set = set(range(cost_matrix.shape[1]))

    for r, c in zip(row_idx, col_idx):
        if cost_matrix[r, c] > threshold:
            unmatched_rows.append(r)
            unmatched_cols.append(c)
        else:
            matched.append((r, c))
        row_set.discard(r)
        col_set.discard(c)

    unmatched_rows.extend(row_set)
    unmatched_cols.extend(col_set)
    return matched, unmatched_rows, unmatched_cols
