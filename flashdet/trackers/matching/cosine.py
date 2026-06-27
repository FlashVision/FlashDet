"""Cosine distance for appearance (ReID) matching."""

from __future__ import annotations

import numpy as np


def cosine_distance(embeddings_a: np.ndarray, embeddings_b: np.ndarray) -> np.ndarray:
    """Cosine distance matrix between two sets of feature vectors.

    Parameters
    ----------
    embeddings_a : np.ndarray, shape (N, D)
    embeddings_b : np.ndarray, shape (M, D)

    Returns
    -------
    np.ndarray, shape (N, M) with values in [0, 2].
    """
    a_norm = embeddings_a / (np.linalg.norm(embeddings_a, axis=1, keepdims=True) + 1e-6)
    b_norm = embeddings_b / (np.linalg.norm(embeddings_b, axis=1, keepdims=True) + 1e-6)
    return 1.0 - a_norm @ b_norm.T
