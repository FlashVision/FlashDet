"""
Visualization utilities for object detection.
"""

import colorsys
import cv2
import numpy as np
from typing import List, Tuple, Dict


# Default class names (PPE dataset)
CLASS_NAMES = [
    "Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest",
    "Person", "Safety Cone", "Safety Vest", "machinery", "vehicle"
]

# High-contrast colors for each class (BGR format)
COLORS = {
    "Hardhat":         (0, 200, 0),
    "Mask":            (0, 200, 0),
    "NO-Hardhat":      (0, 0, 230),
    "NO-Mask":         (0, 0, 230),
    "NO-Safety Vest":  (0, 0, 230),
    "Person":          (230, 180, 0),
    "Safety Cone":     (0, 140, 255),
    "Safety Vest":     (50, 205, 50),
    "machinery":       (180, 105, 30),
    "vehicle":         (200, 0, 200),
}


def make_color_palette(n: int) -> Dict[int, Tuple[int, int, int]]:
    """Generate *n* visually distinct BGR colors using HSV spacing."""
    palette = {}
    for i in range(n):
        hue = i / max(n, 1)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.90)
        palette[i] = (int(b * 255), int(g * 255), int(r * 255))
    return palette


def _font_params(img_h: int):
    """Return (font_scale, thickness, pad) scaled to image height."""
    base = max(img_h / 480.0, 0.4)
    scale = round(base * 0.45, 2)
    thickness = max(1, int(base * 0.9))
    pad = max(2, int(base * 3))
    return scale, thickness, pad


def draw_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray = None,
    class_names: List[str] = None,
    colors: Dict = None,
    line_thickness: int = None,
) -> np.ndarray:
    """Draw bounding boxes with readable, non-overlapping labels.

    Args:
        image: BGR image.
        boxes: [N, 4] ``(x1, y1, x2, y2)`` in pixel coordinates.
        labels: [N] integer class ids.
        scores: [N] confidence scores (optional).
        class_names: List mapping class-id → name.
        colors: Dict mapping class-name (str) or class-id (int) to BGR tuple.
        line_thickness: Box outline thickness (auto-scaled if *None*).
    """
    class_names = class_names or CLASS_NAMES
    colors = colors or COLORS
    output = image.copy()
    h = output.shape[0]
    font_scale, font_thick, pad = _font_params(h)
    lt = line_thickness or max(1, int(h / 300))

    used_regions: list = []

    for i, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = map(int, box)
        cls_id = int(label)
        name = class_names[cls_id] if cls_id < len(class_names) else f"cls_{cls_id}"

        color = colors.get(name) or colors.get(cls_id, (255, 255, 255))

        cv2.rectangle(output, (x1, y1), (x2, y2), color, lt)

        text = f"{name}: {scores[i]:.2f}" if scores is not None and i < len(scores) else name
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)

        # Place label above the box; nudge down if it would overlap a prior label
        lx = x1
        ly = y1 - pad
        label_rect = (lx, ly - th - pad, lx + tw + pad * 2, ly + pad)
        for rx1, ry1, rx2, ry2 in used_regions:
            if lx < rx2 and lx + tw + pad * 2 > rx1 and label_rect[1] < ry2 and label_rect[3] > ry1:
                ly = ry2 + th + pad
                label_rect = (lx, ly - th - pad, lx + tw + pad * 2, ly + pad)

        used_regions.append(label_rect)

        bg_tl = (lx, ly - th - pad)
        bg_br = (lx + tw + pad * 2, ly + pad)
        cv2.rectangle(output, bg_tl, bg_br, color, -1)
        cv2.putText(output, text, (lx + pad, ly), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)

    return output


def draw_detections(
    image: np.ndarray,
    detections: List[Tuple],
    class_names: List[str] = None,
) -> np.ndarray:
    """Draw detections as ``(class_name, score, x1, y1, x2, y2)`` tuples."""
    if not detections:
        return image.copy()

    class_names = class_names or CLASS_NAMES
    output = image.copy()
    h = output.shape[0]
    font_scale, font_thick, pad = _font_params(h)
    lt = max(1, int(h / 300))

    used_regions: list = []

    for det in detections:
        if len(det) == 6:
            name, score, x1, y1, x2, y2 = det
        else:
            x1, y1, x2, y2, score, cls_id = det
            name = class_names[int(cls_id)] if int(cls_id) < len(class_names) else f"cls_{int(cls_id)}"

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        color = COLORS.get(name, (255, 255, 255))

        cv2.rectangle(output, (x1, y1), (x2, y2), color, lt)

        text = f"{name}: {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)

        lx, ly = x1, y1 - pad
        label_rect = (lx, ly - th - pad, lx + tw + pad * 2, ly + pad)
        for rx1, ry1, rx2, ry2 in used_regions:
            if lx < rx2 and lx + tw + pad * 2 > rx1 and label_rect[1] < ry2 and label_rect[3] > ry1:
                ly = ry2 + th + pad
                label_rect = (lx, ly - th - pad, lx + tw + pad * 2, ly + pad)
        used_regions.append(label_rect)

        cv2.rectangle(output, (lx, ly - th - pad), (lx + tw + pad * 2, ly + pad), color, -1)
        cv2.putText(output, text, (lx + pad, ly), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)

    return output


def make_gt_pred_panel(
    image: np.ndarray,
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
    pred_boxes: np.ndarray = None,
    pred_labels: np.ndarray = None,
    pred_scores: np.ndarray = None,
    class_names: List[str] = None,
    colors: Dict = None,
    title_extra: str = "",
) -> np.ndarray:
    """Create a side-by-side Ground Truth | Predictions panel.

    Returns a single BGR image roughly ``(h, 2*w + border, 3)``.
    """
    class_names = class_names or CLASS_NAMES
    colors = colors or COLORS
    h, w = image.shape[:2]
    border = max(4, int(w * 0.015))

    gt_img = image.copy()
    pred_img = image.copy()

    if gt_boxes is not None and len(gt_boxes):
        gt_img = draw_boxes(gt_img, gt_boxes, gt_labels,
                            class_names=class_names, colors=colors)
    if pred_boxes is not None and len(pred_boxes):
        pred_img = draw_boxes(pred_img, pred_boxes, pred_labels,
                              scores=pred_scores, class_names=class_names,
                              colors=colors)

    # Header bar
    header_h = max(28, int(h * 0.06))
    font_scale = max(0.45, header_h / 55.0)
    ft = max(1, int(font_scale * 1.5))

    panel_w = w * 2 + border
    panel = np.zeros((header_h + h, panel_w, 3), dtype=np.uint8)
    panel[:header_h] = (40, 40, 40)

    gt_count = len(gt_boxes) if gt_boxes is not None else 0
    pred_count = len(pred_boxes) if pred_boxes is not None and len(pred_boxes) else 0

    gt_title = f"Ground Truth ({gt_count})"
    pred_title = f"Predictions ({pred_count})"
    if title_extra:
        pred_title += f"  {title_extra}"

    cv2.putText(panel, gt_title, (8, header_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (120, 255, 120), ft, cv2.LINE_AA)
    cv2.putText(panel, pred_title, (w + border + 8, header_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (120, 200, 255), ft, cv2.LINE_AA)

    panel[header_h:, :w] = gt_img
    panel[header_h:, w:w + border] = (80, 80, 80)
    panel[header_h:, w + border:] = pred_img

    return panel


def add_fps_overlay(
    image: np.ndarray,
    fps: float,
    position: Tuple[int, int] = (10, 30)
) -> np.ndarray:
    """Add FPS counter overlay to image."""
    cv2.putText(image, f"FPS: {fps:.1f}", position,
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    return image


def add_violation_warning(
    image: np.ndarray,
    num_violations: int,
    position: Tuple[int, int] = (10, 70)
) -> np.ndarray:
    """Add violation warning overlay to image."""
    if num_violations > 0:
        cv2.putText(image, f"VIOLATIONS: {num_violations}", position,
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    return image


def count_violations(detections: List[Tuple]) -> Tuple[List, List]:
    """Partition detections into (violations, safe)."""
    violations, safe = [], []
    for det in detections:
        if len(det) < 6 and not isinstance(det[0], str):
            continue
        name = det[0] if isinstance(det[0], str) else CLASS_NAMES[int(det[5])]
        if name.startswith("NO-"):
            violations.append(det)
        elif name in ("Hardhat", "Mask", "Safety Vest"):
            safe.append(det)
    return violations, safe
