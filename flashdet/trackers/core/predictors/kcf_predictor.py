"""KCFPredictor — Kernelized Correlation Filter bbox prediction.

Uses HOG features and a Gaussian kernel ridge regression in the
Fourier domain for efficient single-object tracking.  The entire
train/detect loop runs in O(N log N) via FFT.

Reference:
    Henriques et al., "High-Speed Tracking with Kernelized
    Correlation Filters", TPAMI 2015.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from flashdet.trackers.core.predictors.base import BasePredictor


def _fft2(x: np.ndarray) -> np.ndarray:
    return np.fft.fft2(x, axes=(0, 1))


def _ifft2(x: np.ndarray) -> np.ndarray:
    return np.fft.ifft2(x, axes=(0, 1)).real


def _gaussian_label(size: tuple, sigma: float) -> np.ndarray:
    """Create 2-D Gaussian regression target (centred at origin)."""
    h, w = size
    ys = np.roll(np.arange(h) - h // 2, -h // 2)
    xs = np.roll(np.arange(w) - w // 2, -w // 2)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    g = np.exp(-0.5 * (xx ** 2 + yy ** 2) / (sigma ** 2))
    return g / g.max()


def _hog_features(patch: np.ndarray, cell_size: int = 4) -> np.ndarray:
    """Compute simplified HOG-like features for KCF.

    Returns an (H/cell, W/cell, nbins) feature map.
    """
    if len(patch.shape) == 3:
        grey = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        grey = patch.astype(np.float32)

    gx = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=1)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    angle = np.arctan2(gy, gx) * (180.0 / np.pi) % 180.0

    nbins = 9
    bin_width = 180.0 / nbins
    h, w = grey.shape
    ch = h // cell_size
    cw = w // cell_size
    if ch < 1 or cw < 1:
        return np.zeros((max(ch, 1), max(cw, 1), nbins), dtype=np.float32)

    hog = np.zeros((ch, cw, nbins), dtype=np.float32)
    for ci in range(ch):
        for cj in range(cw):
            y0, y1 = ci * cell_size, (ci + 1) * cell_size
            x0, x1 = cj * cell_size, (cj + 1) * cell_size
            cell_mag = magnitude[y0:y1, x0:x1]
            cell_ang = angle[y0:y1, x0:x1]
            for b in range(nbins):
                lo = b * bin_width
                hi = lo + bin_width
                mask = (cell_ang >= lo) & (cell_ang < hi)
                hog[ci, cj, b] = cell_mag[mask].sum()
    # L2-normalise per cell
    norm = np.linalg.norm(hog, axis=2, keepdims=True) + 1e-6
    hog = hog / norm
    return hog


def _gaussian_kernel(x1: np.ndarray, x2: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian kernel correlation in Fourier domain."""
    c = np.real(np.fft.ifft2(
        np.sum(np.fft.fft2(x1, axes=(0, 1)) * np.conj(np.fft.fft2(x2, axes=(0, 1))), axis=2),
    ))
    d = (np.sum(x1 ** 2) + np.sum(x2 ** 2) - 2.0 * c) / x1.size
    d = np.maximum(d, 0)
    k = np.exp(-d / (sigma ** 2 + 1e-8))
    return np.fft.fft2(k)


class KCFPredictor(BasePredictor):
    """KCF-based bounding-box predictor.

    Parameters
    ----------
    sigma : float
        Gaussian kernel bandwidth.
    lambda_ : float
        Regularisation parameter for ridge regression.
    interp_factor : float
        Linear interpolation factor for model update.
    cell_size : int
        HOG cell size in pixels.
    """

    def __init__(
        self,
        sigma: float = 0.5,
        lambda_: float = 1e-4,
        interp_factor: float = 0.02,
        cell_size: int = 4,
    ):
        self._sigma = sigma
        self._lambda = lambda_
        self._interp_factor = interp_factor
        self._cell_size = cell_size

        self._bbox = np.zeros(4, dtype=np.float64)
        self._prev_frame: Optional[np.ndarray] = None
        self._curr_frame: Optional[np.ndarray] = None

        self._alpha: Optional[np.ndarray] = None
        self._template: Optional[np.ndarray] = None
        self._label_fft: Optional[np.ndarray] = None
        self._patch_size: tuple = (0, 0)

    @property
    def needs_frame(self) -> bool:
        return True

    def set_frame(self, frame: np.ndarray):
        self._prev_frame = self._curr_frame
        self._curr_frame = frame.copy()

    def initiate(self, bbox: np.ndarray):
        self._bbox = bbox.astype(np.float64).copy()
        if self._curr_frame is not None:
            self._train(self._curr_frame)

    def predict(self) -> np.ndarray:
        if self._curr_frame is None or self._template is None:
            return self._bbox.copy()

        patch = self._extract_patch(self._curr_frame, self._bbox)
        features = _hog_features(patch, self._cell_size)

        fh, fw = self._template.shape[:2]
        if features.shape[0] != fh or features.shape[1] != fw:
            return self._bbox.copy()

        kzx = _gaussian_kernel(features, self._template, self._sigma)
        response = _ifft2(self._alpha * kzx)
        peak = np.unravel_index(response.argmax(), response.shape)

        dy = peak[0] - fh // 2
        dx = peak[1] - fw // 2
        dx *= self._cell_size
        dy *= self._cell_size

        self._bbox[0] += dx
        self._bbox[1] += dy
        self._bbox[2] += dx
        self._bbox[3] += dy
        return self._bbox.copy()

    def update(self, bbox: np.ndarray):
        self._bbox = bbox.astype(np.float64).copy()
        if self._curr_frame is not None:
            self._train(self._curr_frame)

    def _train(self, frame: np.ndarray):
        patch = self._extract_patch(frame, self._bbox)
        features = _hog_features(patch, self._cell_size)
        h, w = features.shape[:2]
        self._patch_size = (h, w)

        y = _gaussian_label((h, w), sigma=max(h, w) * 0.1 + 1.0)
        yf = _fft2(y)

        kxx = _gaussian_kernel(features, features, self._sigma)
        alpha = yf / (kxx + self._lambda)

        if self._alpha is None:
            self._alpha = alpha
            self._template = features.copy()
            self._label_fft = yf
        else:
            lr = self._interp_factor
            self._alpha = (1 - lr) * self._alpha + lr * alpha
            self._template = (1 - lr) * self._template + lr * features

    def _extract_patch(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(w, int(bbox[2]))
        y2 = min(h, int(bbox[3]))
        if x2 <= x1 or y2 <= y1:
            return np.zeros((32, 32, 3), dtype=np.uint8)
        return frame[y1:y2, x1:x2].copy()
