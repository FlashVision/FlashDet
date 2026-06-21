"""
Advanced detection augmentations: Mosaic, MixUp, and Copy-Paste.

These augmentations operate on multiple images and their annotations
simultaneously, producing richly augmented training samples.

References:
  - Mosaic: Bochkovskiy et al., "YOLOv4", 2020.
  - MixUp: Zhang et al., "mixup: Beyond Empirical Risk Minimization", ICLR 2018.
  - Copy-Paste: Ghiasi et al., "Simple Copy-Paste is a Strong Data
    Augmentation Method for Instance Segmentation", CVPR 2021.
"""

import random
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np


class Mosaic:
    """4-image mosaic augmentation.

    Stitches 4 random images into a 2x2 grid with a random center offset,
    creating a single composite training image with varied spatial context.

    Args:
        img_size: Output image size (w, h).
        center_range: Fractional range for the mosaic center [low, high].
        fill_value: Border fill value.
        extra_images_fn: Callable that returns (image, boxes, labels) for a
            random dataset sample. Required for the extra 3 images.
    """

    def __init__(
        self,
        img_size: Tuple[int, int] = (640, 640),
        center_range: Tuple[float, float] = (0.5, 1.5),
        fill_value: int = 114,
        extra_images_fn: Optional[Callable] = None,
    ):
        self.img_size = img_size
        self.center_range = center_range
        self.fill_value = fill_value
        self.extra_images_fn = extra_images_fn

    def __call__(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.extra_images_fn is None:
            return image, boxes, labels

        w, h = self.img_size
        mosaic_img = np.full((h * 2, w * 2, 3), self.fill_value, dtype=np.uint8)

        cx = int(random.uniform(self.center_range[0] * w, self.center_range[1] * w))
        cy = int(random.uniform(self.center_range[0] * h, self.center_range[1] * h))

        samples = [(image, boxes.copy(), labels.copy())]
        for _ in range(3):
            samples.append(self.extra_images_fn())

        all_boxes = []
        all_labels = []

        placements = [
            (max(cx - samples[0][0].shape[1], 0), max(cy - samples[0][0].shape[0], 0), cx, cy),
            (cx, max(cy - samples[1][0].shape[0], 0), min(cx + samples[1][0].shape[1], w * 2), cy),
            (max(cx - samples[2][0].shape[1], 0), cy, cx, min(cy + samples[2][0].shape[0], h * 2)),
            (cx, cy, min(cx + samples[3][0].shape[1], w * 2), min(cy + samples[3][0].shape[0], h * 2)),
        ]

        for idx, (img, bxs, lbs) in enumerate(samples):
            x1, y1, x2, y2 = placements[idx]
            pw, ph = x2 - x1, y2 - y1
            if pw <= 0 or ph <= 0:
                continue

            img_h, img_w = img.shape[:2]
            # Compute which part of the source image to copy
            if idx == 0:
                src_x1 = max(img_w - pw, 0)
                src_y1 = max(img_h - ph, 0)
            elif idx == 1:
                src_x1 = 0
                src_y1 = max(img_h - ph, 0)
            elif idx == 2:
                src_x1 = max(img_w - pw, 0)
                src_y1 = 0
            else:
                src_x1 = 0
                src_y1 = 0

            src_x2 = min(src_x1 + pw, img_w)
            src_y2 = min(src_y1 + ph, img_h)
            actual_w = src_x2 - src_x1
            actual_h = src_y2 - src_y1

            mosaic_img[y1:y1 + actual_h, x1:x1 + actual_w] = img[src_y1:src_y2, src_x1:src_x2]

            if bxs is not None and len(bxs) > 0:
                offset_x = x1 - src_x1
                offset_y = y1 - src_y1
                shifted = bxs.copy()
                shifted[:, [0, 2]] += offset_x
                shifted[:, [1, 3]] += offset_y

                shifted[:, [0, 2]] = np.clip(shifted[:, [0, 2]], x1, x1 + actual_w)
                shifted[:, [1, 3]] = np.clip(shifted[:, [1, 3]], y1, y1 + actual_h)

                bw = shifted[:, 2] - shifted[:, 0]
                bh = shifted[:, 3] - shifted[:, 1]
                valid = (bw > 2) & (bh > 2)
                all_boxes.append(shifted[valid])
                all_labels.append(lbs[valid])

        # Crop to output size
        crop_x1 = max(cx - w // 2, 0)
        crop_y1 = max(cy - h // 2, 0)
        crop_x2 = crop_x1 + w
        crop_y2 = crop_y1 + h

        if crop_x2 > w * 2:
            crop_x2 = w * 2
            crop_x1 = crop_x2 - w
        if crop_y2 > h * 2:
            crop_y2 = h * 2
            crop_y1 = crop_y2 - h

        result_img = mosaic_img[crop_y1:crop_y2, crop_x1:crop_x2]

        if all_boxes:
            result_boxes = np.concatenate(all_boxes, axis=0)
            result_labels = np.concatenate(all_labels, axis=0)
            result_boxes[:, [0, 2]] -= crop_x1
            result_boxes[:, [1, 3]] -= crop_y1
            result_boxes[:, [0, 2]] = np.clip(result_boxes[:, [0, 2]], 0, w)
            result_boxes[:, [1, 3]] = np.clip(result_boxes[:, [1, 3]], 0, h)

            bw = result_boxes[:, 2] - result_boxes[:, 0]
            bh = result_boxes[:, 3] - result_boxes[:, 1]
            valid = (bw > 2) & (bh > 2)
            result_boxes = result_boxes[valid]
            result_labels = result_labels[valid]
        else:
            result_boxes = np.zeros((0, 4), dtype=np.float32)
            result_labels = np.zeros((0,), dtype=np.int64)

        return result_img, result_boxes, result_labels


class MixUp:
    """MixUp augmentation for object detection.

    Blends two images and their annotations with a random mixing ratio,
    effectively creating a superimposed composite training sample.

    Args:
        alpha: Beta distribution parameter for sampling the mix ratio.
        extra_image_fn: Callable returning (image, boxes, labels) for the
            second sample.
    """

    def __init__(
        self,
        alpha: float = 1.5,
        extra_image_fn: Optional[Callable] = None,
    ):
        self.alpha = alpha
        self.extra_image_fn = extra_image_fn

    def __call__(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.extra_image_fn is None:
            return image, boxes, labels

        image2, boxes2, labels2 = self.extra_image_fn()

        lam = np.random.beta(self.alpha, self.alpha)
        lam = max(lam, 1 - lam)

        h1, w1 = image.shape[:2]
        h2, w2 = image2.shape[:2]
        h_out, w_out = max(h1, h2), max(w1, w2)

        img1_pad = np.full((h_out, w_out, 3), 114, dtype=np.uint8)
        img2_pad = np.full((h_out, w_out, 3), 114, dtype=np.uint8)
        img1_pad[:h1, :w1] = image
        img2_pad[:h2, :w2] = image2

        mixed = (lam * img1_pad.astype(np.float32) + (1 - lam) * img2_pad.astype(np.float32)).astype(np.uint8)

        if boxes is not None and len(boxes) > 0 and boxes2 is not None and len(boxes2) > 0:
            merged_boxes = np.concatenate([boxes, boxes2], axis=0)
            merged_labels = np.concatenate([labels, labels2], axis=0)
        elif boxes is not None and len(boxes) > 0:
            merged_boxes = boxes
            merged_labels = labels
        elif boxes2 is not None and len(boxes2) > 0:
            merged_boxes = boxes2
            merged_labels = labels2
        else:
            merged_boxes = np.zeros((0, 4), dtype=np.float32)
            merged_labels = np.zeros((0,), dtype=np.int64)

        return mixed, merged_boxes, merged_labels


class CopyPaste:
    """Copy-Paste augmentation for detection.

    Copies object instances (crops) from a source image and pastes them
    onto the target image at random locations.

    Args:
        prob: Probability of applying copy-paste per instance.
        extra_image_fn: Callable returning (image, boxes, labels) for the
            source of paste candidates.
    """

    def __init__(
        self,
        prob: float = 0.5,
        extra_image_fn: Optional[Callable] = None,
    ):
        self.prob = prob
        self.extra_image_fn = extra_image_fn

    def __call__(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.extra_image_fn is None:
            return image, boxes, labels

        src_image, src_boxes, src_labels = self.extra_image_fn()

        if src_boxes is None or len(src_boxes) == 0:
            return image, boxes, labels

        h_dst, w_dst = image.shape[:2]
        h_src, w_src = src_image.shape[:2]

        new_boxes = list(boxes) if boxes is not None else []
        new_labels = list(labels) if labels is not None else []
        result = image.copy()

        for i in range(len(src_boxes)):
            if random.random() > self.prob:
                continue

            x1, y1, x2, y2 = src_boxes[i].astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_src, x2), min(h_src, y2)

            crop_w = x2 - x1
            crop_h = y2 - y1
            if crop_w < 4 or crop_h < 4:
                continue

            crop = src_image[y1:y2, x1:x2]

            paste_x = random.randint(0, max(0, w_dst - crop_w))
            paste_y = random.randint(0, max(0, h_dst - crop_h))

            actual_w = min(crop_w, w_dst - paste_x)
            actual_h = min(crop_h, h_dst - paste_y)

            if actual_w < 4 or actual_h < 4:
                continue

            result[paste_y:paste_y + actual_h, paste_x:paste_x + actual_w] = crop[:actual_h, :actual_w]

            new_boxes.append(np.array([paste_x, paste_y, paste_x + actual_w, paste_y + actual_h], dtype=np.float32))
            new_labels.append(src_labels[i])

        if new_boxes:
            result_boxes = np.array(new_boxes, dtype=np.float32) if isinstance(new_boxes[0], np.ndarray) else np.stack(new_boxes)
            result_labels = np.array(new_labels, dtype=np.int64) if isinstance(new_labels[0], (int, np.integer)) else np.concatenate([np.atleast_1d(l) for l in new_labels])
        else:
            result_boxes = np.zeros((0, 4), dtype=np.float32)
            result_labels = np.zeros((0,), dtype=np.int64)

        return result, result_boxes, result_labels
