"""Mahalanobis distance for statistically-gated track association.

Uses the Kalman filter's projected state covariance to weight the
distance, so uncertain tracks have a larger gating region.  This is
the core matching strategy of Deep SORT (Wojke et al., 2017).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from flashdet.trackers.matching.geometry import xyxy_to_cxywh

# Chi-squared 95% confidence gate for 4-DOF measurement (cx, cy, w, h)
CHI2_THRESHOLD_95 = chi2.ppf(0.95, df=4)


def mahalanobis_distance(
    kalman_filter,
    tracks,
    detections: np.ndarray,
    gating_threshold: float = CHI2_THRESHOLD_95,
) -> np.ndarray:
    """Squared Mahalanobis distance between tracks and detections.

    Parameters
    ----------
    kalman_filter : KalmanFilter
        The shared Kalman filter instance (provides projection matrices).
    tracks : list of Track
        Each track must have ``.mean`` and ``.covariance`` attributes.
    detections : np.ndarray, shape (N, 4+)
        Detections in [x1, y1, x2, y2, ...] format.
    gating_threshold : float
        Entries exceeding this are set to ``inf`` (gated out).

    Returns
    -------
    np.ndarray, shape (len(tracks), N)
        Squared Mahalanobis distances.  Gated entries are ``inf``.
    """
    n_trk = len(tracks)
    n_det = len(detections)
    cost = np.full((n_trk, n_det), np.inf, dtype=np.float64)

    measurements = np.array(
        [xyxy_to_cxywh(d[:4]) for d in detections], dtype=np.float64,
    )

    kf = kalman_filter
    H = kf._H  # (4, 8) observation matrix

    for i, trk in enumerate(tracks):
        projected_mean = H @ trk.mean
        projected_cov = H @ trk.covariance @ H.T

        # Add measurement noise
        std = np.array([
            kf._std_pos * trk.mean[2], kf._std_pos * trk.mean[3],
            kf._std_pos * trk.mean[2], kf._std_pos * trk.mean[3],
        ], dtype=np.float64)
        projected_cov += np.diag(std ** 2)

        try:
            chol = np.linalg.cholesky(projected_cov)
        except np.linalg.LinAlgError:
            continue

        diff = measurements - projected_mean
        z = np.linalg.solve(chol, diff.T).T  # whitened residuals
        sq_dist = np.sum(z * z, axis=1)

        cost[i] = sq_dist
        cost[i, sq_dist > gating_threshold] = np.inf

    return cost
