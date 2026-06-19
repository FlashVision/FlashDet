"""
Data augmentation transforms for PPE detection.

Training uses a full homography pipeline (matching official FlashDet):
  - All spatial augmentations (scale, flip, perspective, rotation, shear,
    translate) are composed into a single 3x3 warp matrix and applied in
    one cv2.warpPerspective call.
  - keep_ratio=True letterboxes the image so aspect ratio is preserved.
  - The warp matrix is stored in meta so boxes can be precisely unprojected.

Inference uses the same letterbox logic and stores the warp matrix so that
detected boxes are mapped back to the original image correctly.
"""

import math
import random
from typing import Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Warp-matrix helpers (matching official FlashDet warp.py)
# ---------------------------------------------------------------------------

def _get_flip_matrix(prob: float = 0.5) -> np.ndarray:
    F = np.eye(3)
    if random.random() < prob:
        F[0, 0] = -1
    return F


def _get_perspective_matrix(perspective: float = 0.0) -> np.ndarray:
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)
    P[2, 1] = random.uniform(-perspective, perspective)
    return P


def _get_rotation_matrix(degree: float = 0.0) -> np.ndarray:
    R = np.eye(3)
    a = random.uniform(-degree, degree)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=1)
    return R


def _get_scale_matrix(ratio: Tuple[float, float] = (1, 1)) -> np.ndarray:
    S = np.eye(3)
    S[0, 0] *= random.uniform(*ratio)
    S[1, 1] = S[0, 0]
    return S


def _get_stretch_matrix(
    width_ratio: Tuple[float, float] = (1, 1),
    height_ratio: Tuple[float, float] = (1, 1),
) -> np.ndarray:
    S = np.eye(3)
    S[0, 0] *= random.uniform(*width_ratio)
    S[1, 1] *= random.uniform(*height_ratio)
    return S


def _get_shear_matrix(degree: float = 0.0) -> np.ndarray:
    Sh = np.eye(3)
    Sh[0, 1] = math.tan(random.uniform(-degree, degree) * math.pi / 180)
    Sh[1, 0] = math.tan(random.uniform(-degree, degree) * math.pi / 180)
    return Sh


def _get_translate_matrix(translate: float, width: int, height: int) -> np.ndarray:
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height
    return T


def _get_resize_matrix(
    raw_shape: Tuple[int, int],
    dst_shape: Tuple[int, int],
    keep_ratio: bool,
) -> np.ndarray:
    """Return 3x3 matrix that maps raw_shape → dst_shape.

    When keep_ratio=True the image is letterboxed (centred, padded with grey).
    """
    r_w, r_h = raw_shape
    d_w, d_h = dst_shape
    Rs = np.eye(3)
    if keep_ratio:
        C = np.eye(3)
        C[0, 2] = -r_w / 2
        C[1, 2] = -r_h / 2
        ratio = d_h / r_h if r_w / r_h < d_w / d_h else d_w / r_w
        Rs[0, 0] *= ratio
        Rs[1, 1] *= ratio
        T = np.eye(3)
        T[0, 2] = 0.5 * d_w
        T[1, 2] = 0.5 * d_h
        return T @ Rs @ C
    else:
        Rs[0, 0] *= d_w / r_w
        Rs[1, 1] *= d_h / r_h
        return Rs


def _warp_boxes(
    boxes: np.ndarray, M: np.ndarray, width: int, height: int
) -> np.ndarray:
    """Apply homography M to a set of xyxy boxes."""
    n = len(boxes)
    if n == 0:
        return boxes
    # Represent each box as 4 corner points
    xy = np.ones((n * 4, 3))
    xy[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)
    xy = xy @ M.T
    xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)
    x = xy[:, [0, 2, 4, 6]]
    y = xy[:, [1, 3, 5, 7]]
    warped = np.concatenate(
        [x.min(1), y.min(1), x.max(1), y.max(1)]
    ).reshape(4, n).T
    warped[:, [0, 2]] = warped[:, [0, 2]].clip(0, width)
    warped[:, [1, 3]] = warped[:, [1, 3]].clip(0, height)
    return warped.astype(np.float32)


# ---------------------------------------------------------------------------
# Color augmentation (unchanged)
# ---------------------------------------------------------------------------

def _color_jitter(
    image: np.ndarray,
    brightness: float = 0.2,
    contrast: Tuple[float, float] = (0.6, 1.4),
    saturation: Tuple[float, float] = (0.5, 1.2),
    hue_delta: int = 18,
) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    if hue_delta > 0 and random.random() < 0.5:
        hsv[:, :, 0] += random.uniform(-hue_delta, hue_delta)
        hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 180)
    if random.random() < 0.5:
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(*saturation), 0, 255)
    if brightness > 0 and random.random() < 0.5:
        hsv[:, :, 2] = np.clip(
            hsv[:, :, 2] * random.uniform(1 - brightness, 1 + brightness), 0, 255
        )
    image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    if random.random() < 0.5:
        image = np.clip(
            image.astype(np.float32) * random.uniform(*contrast), 0, 255
        ).astype(np.uint8)
    if random.random() < 0.1:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return image


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

class TrainTransform:
    """
    Training augmentation transform — matches official FlashDet pipeline.

    All spatial ops (scale, flip, perspective, rotation, shear, translate,
    letterbox-resize) are composed into a single 3x3 warp matrix applied via
    cv2.warpPerspective.  keep_ratio=True ensures the image is letterboxed.
    """

    def __init__(
        self,
        input_size: Tuple[int, int] = (320, 320),
        keep_ratio: bool = True,
        scale: Tuple[float, float] = (0.6, 1.4),
        stretch: Tuple = ((0.8, 1.2), (0.8, 1.2)),
        flip_prob: float = 0.5,
        perspective: float = 0.0,
        rotation: float = 0.0,
        shear: float = 0.0,      # Official FlashDet config: shear=0
        translate: float = 0.2,
        brightness: float = 0.2,              # Matches official config
        contrast: Tuple[float, float] = (0.6, 1.4),   # Matches official config
        saturation: Tuple[float, float] = (0.5, 1.2), # Matches official config
        hue_delta: int = 18,
        mean: Tuple[float, ...] = (123.675, 116.28, 103.53),
        std: Tuple[float, ...] = (58.395, 57.12, 57.375),
    ):
        self.input_size = input_size
        self.keep_ratio = keep_ratio
        self.scale = scale
        self.stretch = stretch
        self.flip_prob = flip_prob
        self.perspective = perspective
        self.rotation = rotation
        self.shear = shear
        self.translate = translate
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue_delta = hue_delta
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        labels: np.ndarray,
    ):
        import torch

        h, w = image.shape[:2]
        dst_w, dst_h = self.input_size

        # --- Build composite warp matrix (same order as official FlashDet) ---
        C = np.eye(3)
        C[0, 2] = -w / 2
        C[1, 2] = -h / 2

        if self.perspective > 0:
            C = _get_perspective_matrix(self.perspective) @ C
        C = _get_scale_matrix(self.scale) @ C
        if self.stretch != ((1, 1), (1, 1)):
            C = _get_stretch_matrix(*self.stretch) @ C
        if self.rotation > 0:
            C = _get_rotation_matrix(self.rotation) @ C
        if self.shear > 0:
            C = _get_shear_matrix(self.shear) @ C
        C = _get_flip_matrix(self.flip_prob) @ C
        T = _get_translate_matrix(self.translate, w, h)
        M = T @ C

        ResizeM = _get_resize_matrix((w, h), (dst_w, dst_h), self.keep_ratio)
        M = ResizeM @ M

        # Apply warp (grey border = 114)
        image = cv2.warpPerspective(
            image, M, dsize=(dst_w, dst_h),
            borderValue=(114, 114, 114)
        )

        # Warp boxes
        if len(boxes) > 0:
            boxes = _warp_boxes(boxes, M, dst_w, dst_h)
            # Remove degenerate boxes (< 2px in either dim)
            w_box = boxes[:, 2] - boxes[:, 0]
            h_box = boxes[:, 3] - boxes[:, 1]
            valid = (w_box >= 2) & (h_box >= 2)
            boxes = boxes[valid]
            labels = labels[valid]

        # Color jitter
        image = _color_jitter(
            image, self.brightness, self.contrast, self.saturation, self.hue_delta
        )

        # Normalize and convert to tensor
        image = (image.astype(np.float32) - self.mean) / self.std
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()

        return image, boxes, labels


class ValTransform:
    """Validation transform — letterbox resize, no augmentation."""

    def __init__(
        self,
        input_size: Tuple[int, int] = (320, 320),
        mean: Tuple[float, ...] = (123.675, 116.28, 103.53),
        std: Tuple[float, ...] = (58.395, 57.12, 57.375),
    ):
        self.input_size = input_size
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        labels: np.ndarray,
    ):
        import torch

        h, w = image.shape[:2]
        dst_w, dst_h = self.input_size

        M = _get_resize_matrix((w, h), (dst_w, dst_h), keep_ratio=True)
        image = cv2.warpPerspective(
            image, M, dsize=(dst_w, dst_h), borderValue=(114, 114, 114)
        )

        if len(boxes) > 0:
            boxes = _warp_boxes(boxes, M, dst_w, dst_h)

        image = (image.astype(np.float32) - self.mean) / self.std
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()

        return image, boxes, labels


class InferenceTransform:
    """
    Inference transform — letterbox resize, stores warp matrix for precise
    box remapping back to original image coordinates.
    """

    def __init__(
        self,
        input_size: Tuple[int, int] = (320, 320),
        mean: Tuple[float, ...] = (123.675, 116.28, 103.53),
        std: Tuple[float, ...] = (58.395, 57.12, 57.375),
    ):
        self.input_size = input_size
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image: np.ndarray):
        """
        Returns:
            ndarray [C, H, W] (float32) and meta dict containing the
            warp_matrix needed to map detected boxes back to original
            image space.  Caller is responsible for wrapping in a tensor.
        """
        h, w = image.shape[:2]
        dst_w, dst_h = self.input_size

        M = _get_resize_matrix((w, h), (dst_w, dst_h), keep_ratio=True)
        warped = cv2.warpPerspective(
            image, M, dsize=(dst_w, dst_h), borderValue=(114, 114, 114)
        )

        meta = {
            "original_size": (w, h),
            "input_size": self.input_size,
            "warp_matrix": M,
        }

        warped = (warped.astype(np.float32) - self.mean) / self.std
        return warped.transpose(2, 0, 1), meta
