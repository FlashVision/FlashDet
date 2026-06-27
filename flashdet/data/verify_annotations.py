"""
Annotation verification for FlashDet data pipelines.

Checks raw COCO annotations and dataloader output. Saves reports and
single-image GT box overlays inside the training save directory.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

GT_VERIFY_SUBDIR = "gt_verification"
GT_IMAGES_SUBDIR = "images"
GT_RAW_SUBDIR = "raw"
GT_DATALOADER_SUBDIR = "dataloader"

_OOB_TOL = 2.0
_MIN_BOX_SIDE = 2.0
_MEAN = np.array([123.675, 116.28, 103.53])
_STD = np.array([58.395, 57.12, 57.375])
_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
    (0, 128, 255), (128, 255, 0), (255, 0, 128), (0, 255, 128),
]


def gt_verification_dir(save_dir: str) -> str:
    """Return the GT verification folder inside a training save directory."""
    return os.path.join(save_dir, GT_VERIFY_SUBDIR)


def _as_numpy(arr) -> np.ndarray:
    if isinstance(arr, np.ndarray):
        return arr
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def _denormalize_tensor(img_tensor) -> np.ndarray:
    """Convert CHW float tensor to HWC BGR uint8."""
    img = _as_numpy(img_tensor).transpose(1, 2, 0)
    img = np.clip(img * _STD + _MEAN, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def _draw_gt_boxes(
    img_bgr: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> np.ndarray:
    """Draw GT xyxy boxes on a single BGR image."""
    vis = img_bgr.copy()
    if boxes.size == 0:
        return vis

    boxes = boxes.reshape(-1, 4)
    labels = _as_numpy(labels).astype(int) if labels is not None and len(labels) else np.zeros(len(boxes), dtype=int)

    for box, lbl in zip(boxes, labels):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        lbl_idx = int(lbl)
        name = class_names[lbl_idx] if class_names and lbl_idx < len(class_names) else f"cls_{lbl_idx}"
        color = _COLORS[lbl_idx % len(_COLORS)]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label_text = f"{name}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(vis, (x1, max(y1 - th - 6, 0)), (x1 + tw + 2, max(y1, th + 6)), color, -1)
        cv2.putText(vis, label_text, (x1 + 1, max(y1 - 4, th + 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return vis


def _check_boxes(
    boxes: np.ndarray,
    img_w: int,
    img_h: int,
    num_classes: Optional[int],
    labels: Optional[np.ndarray] = None,
) -> Dict[str, int]:
    """Return counts of annotation issues for one sample."""
    issues = {"oob": 0, "degenerate": 0, "invalid_label": 0}
    if boxes.size == 0:
        return issues

    boxes = boxes.reshape(-1, 4)
    oob = (
        (boxes[:, 0] < -_OOB_TOL)
        | (boxes[:, 1] < -_OOB_TOL)
        | (boxes[:, 2] > img_w + _OOB_TOL)
        | (boxes[:, 3] > img_h + _OOB_TOL)
    )
    issues["oob"] = int(oob.sum())

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    issues["degenerate"] = int(((widths < _MIN_BOX_SIDE) | (heights < _MIN_BOX_SIDE)).sum())

    if labels is not None and num_classes is not None and len(labels) > 0:
        labels = _as_numpy(labels).astype(int)
        issues["invalid_label"] = int(((labels < 0) | (labels >= num_classes)).sum())

    return issues


def verify_coco_annotations(
    ann_file: str,
    img_dir: str,
    num_classes: Optional[int] = None,
    log: Optional[logging.Logger] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """Validate a COCO annotation file and referenced image files."""
    log = log or logger
    summary: Dict[str, Any] = {
        "ann_file": ann_file,
        "img_dir": img_dir,
        "num_images": 0,
        "num_annotations": 0,
        "missing_images": 0,
        "oob_boxes": 0,
        "degenerate_boxes": 0,
        "unknown_categories": 0,
        "images_without_anns": 0,
        "invalid_labels": 0,
    }

    if not os.path.isfile(ann_file):
        log.error("Annotation file not found: %s", ann_file)
        return False, summary

    with open(ann_file, encoding="utf-8") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])
    cat_ids = sorted(c["id"] for c in categories)
    cat_id_to_idx = {cid: idx for idx, cid in enumerate(cat_ids)}
    if num_classes is None:
        num_classes = len(cat_ids)

    summary["num_images"] = len(images)
    summary["num_annotations"] = len(annotations)
    summary["num_classes"] = num_classes

    img_lookup = {img["id"]: img for img in images}
    img_to_anns: Dict[int, list] = {img["id"]: [] for img in images}

    for ann in annotations:
        cat_id = ann.get("category_id")
        if cat_id not in cat_id_to_idx:
            summary["unknown_categories"] += 1
            continue

        x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
        if w < 1 or h < 1:
            summary["degenerate_boxes"] += 1
            continue

        img_id = ann["image_id"]
        if img_id not in img_lookup:
            continue
        img_to_anns.setdefault(img_id, []).append(ann)

        img_info = img_lookup[img_id]
        box = np.array([[x, y, x + w, y + h]], dtype=np.float32)
        label = np.array([cat_id_to_idx[cat_id]], dtype=np.int64)
        issues = _check_boxes(box, img_info.get("width", 0), img_info.get("height", 0), num_classes, label)
        summary["oob_boxes"] += issues["oob"]
        summary["degenerate_boxes"] += issues["degenerate"]
        summary["invalid_labels"] += issues["invalid_label"]

    for img in images:
        if not os.path.isfile(os.path.join(img_dir, img["file_name"])):
            summary["missing_images"] += 1
        if not img_to_anns.get(img["id"]):
            summary["images_without_anns"] += 1

    ok = (
        summary["missing_images"] == 0
        and summary["unknown_categories"] == 0
        and summary["invalid_labels"] == 0
    )

    log.info(
        "COCO check %s: images=%d anns=%d missing_files=%d oob=%d degenerate=%d "
        "unknown_cat=%d empty_images=%d",
        "OK" if ok else "ISSUES",
        summary["num_images"], summary["num_annotations"], summary["missing_images"],
        summary["oob_boxes"], summary["degenerate_boxes"],
        summary["unknown_categories"], summary["images_without_anns"],
    )
    return ok, summary


def verify_dataloader(
    loader: DataLoader,
    num_batches: int = 5,
    num_classes: Optional[int] = None,
    log: Optional[logging.Logger] = None,
    collect_samples: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    """Sample batches from a DataLoader and verify GT boxes/labels."""
    log = log or logger
    summary: Dict[str, Any] = {
        "batches_checked": 0,
        "samples_checked": 0,
        "total_boxes": 0,
        "empty_samples": 0,
        "oob_boxes": 0,
        "degenerate_boxes": 0,
        "invalid_labels": 0,
        "avg_boxes_per_sample": 0.0,
        "samples": [],
    }

    if num_classes is None and hasattr(loader.dataset, "num_classes"):
        num_classes = loader.dataset.num_classes

    for batch_idx, (images, gt_meta) in enumerate(loader):
        if batch_idx >= num_batches:
            break

        summary["batches_checked"] += 1
        img_h, img_w = int(images.shape[2]), int(images.shape[3])

        for i in range(images.shape[0]):
            summary["samples_checked"] += 1
            boxes = _as_numpy(gt_meta["gt_bboxes"][i])
            labels = _as_numpy(gt_meta["gt_labels"][i])
            img_id = gt_meta.get("img_ids", [None] * images.shape[0])[i]

            sample_info: Dict[str, Any] = {
                "batch_idx": batch_idx, "sample_idx": i, "img_id": img_id,
                "img_size": [img_w, img_h], "num_boxes": 0,
                "oob": 0, "degenerate": 0, "invalid_label": 0,
            }

            if boxes.size == 0:
                summary["empty_samples"] += 1
                if collect_samples:
                    summary["samples"].append(sample_info)
                continue

            boxes = boxes.reshape(-1, 4)
            summary["total_boxes"] += len(boxes)
            issues = _check_boxes(boxes, img_w, img_h, num_classes, labels)
            summary["oob_boxes"] += issues["oob"]
            summary["degenerate_boxes"] += issues["degenerate"]
            summary["invalid_labels"] += issues["invalid_label"]

            if collect_samples:
                sample_info.update({
                    "num_boxes": len(boxes),
                    "oob": issues["oob"],
                    "degenerate": issues["degenerate"],
                    "invalid_label": issues["invalid_label"],
                    "box_x_range": [float(boxes[:, 0].min()), float(boxes[:, 2].max())],
                    "box_y_range": [float(boxes[:, 1].min()), float(boxes[:, 3].max())],
                })
                summary["samples"].append(sample_info)

    avg_boxes = summary["total_boxes"] / max(summary["samples_checked"] - summary["empty_samples"], 1)
    summary["avg_boxes_per_sample"] = round(avg_boxes, 2)
    ok = summary["invalid_labels"] == 0

    log.info(
        "Dataloader check %s: batches=%d samples=%d boxes=%d avg_boxes=%.1f "
        "empty=%d oob=%d degenerate=%d invalid_label=%d",
        "OK" if ok else "ISSUES",
        summary["batches_checked"], summary["samples_checked"], summary["total_boxes"],
        avg_boxes, summary["empty_samples"], summary["oob_boxes"],
        summary["degenerate_boxes"], summary["invalid_labels"],
    )
    return ok, summary


def save_gt_verification_images(
    loader: DataLoader,
    images_dir: str,
    split_name: str,
    class_names: Optional[List[str]] = None,
    num_images: int = 8,
    log: Optional[logging.Logger] = None,
) -> List[str]:
    """
    Save GT box overlays from dataloader output (ValTransform, no mosaic).

    Each sample is saved as a separate image; mosaic sources are split into tiles.
    """
    log = log or logger
    os.makedirs(images_dir, exist_ok=True)
    saved_paths: List[str] = []
    count = 0

    for _batch_idx, (images, gt_meta) in enumerate(loader):
        for i in range(images.shape[0]):
            if count >= num_images:
                break

            img_bgr = _denormalize_tensor(images[i])
            h, w = img_bgr.shape[:2]
            boxes = _as_numpy(gt_meta["gt_bboxes"][i])
            labels = _as_numpy(gt_meta["gt_labels"][i])
            boxes_np = boxes.reshape(-1, 4) if boxes.size else np.zeros((0, 4), dtype=np.float32)

            img_infos = gt_meta.get("img_infos", [None] * images.shape[0])
            file_name = ""
            if img_infos[i] and isinstance(img_infos[i], dict):
                file_name = img_infos[i].get("file_name", "")
            stem = os.path.splitext(os.path.basename(file_name))[0] if file_name else f"sample{count}"

            if _is_mosaic_layout(boxes_np, h, w):
                for qi, (tile, t_boxes, t_labels) in enumerate(
                    _split_mosaic_quadrants(img_bgr, boxes_np, labels)
                ):
                    if count >= num_images:
                        break
                    if len(t_boxes) == 0:
                        continue
                    out_name = f"{split_name}_{count:03d}_{stem}_dl_tile{qi}.jpg"
                    out_path = os.path.join(images_dir, out_name)
                    _save_one_gt_image(tile, t_boxes, t_labels, out_path, class_names)
                    saved_paths.append(out_path)
                    count += 1
            else:
                out_name = f"{split_name}_{count:03d}_{stem}_dl.jpg"
                out_path = os.path.join(images_dir, out_name)
                _save_one_gt_image(img_bgr, boxes_np, labels, out_path, class_names)
                saved_paths.append(out_path)
                count += 1

        if count >= num_images:
            break

    log.info(
        "Saved %d dataloader GT images (separate scenes/tiles) to %s (%s split)",
        len(saved_paths), images_dir, split_name,
    )
    return saved_paths


def _clear_image_dir(images_dir: str) -> None:
    """Remove old JPG/PNG files so each run produces a fresh set."""
    if not os.path.isdir(images_dir):
        return
    for name in os.listdir(images_dir):
        if name.lower().endswith((".jpg", ".jpeg", ".png")):
            try:
                os.remove(os.path.join(images_dir, name))
            except OSError:
                pass


def _is_mosaic_layout(boxes: np.ndarray, img_h: int, img_w: int) -> bool:
    """
    Detect Roboflow-style 2x2 mosaic exports using GT box layout.

    Mosaic images have object centers in 2+ quadrants without one box covering
    the whole frame. Single-scene photos are kept intact.
    """
    if boxes.size == 0 or len(boxes) < 2 or img_h != img_w:
        return False

    boxes = boxes.reshape(-1, 4)
    mh, mw = img_h // 2, img_w // 2
    centers_per_quad = [0, 0, 0, 0]
    max_area_ratio = 0.0
    frame_area = float(img_h * img_w)

    for box in boxes:
        x1, y1, x2, y2 = box
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        max_area_ratio = max(max_area_ratio, area / frame_area)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        qi = (1 if cx >= mw else 0) + (2 if cy >= mh else 0)
        centers_per_quad[qi] += 1

    quads_with_objects = sum(1 for c in centers_per_quad if c > 0)
    return quads_with_objects >= 2 and max_area_ratio < 0.55


def _is_mosaic_grid_image(img_bgr: np.ndarray, boxes: Optional[np.ndarray] = None) -> bool:
    """Detect mosaic using GT layout (preferred) with visual fallback."""
    h, w = img_bgr.shape[:2]
    if boxes is not None and boxes.size:
        if _is_mosaic_layout(boxes, h, w):
            return True
    if h < 64 or w < 64 or abs(h - w) > max(h, w) * 0.05:
        return False
    mh, mw = h // 2, w // 2
    row = img_bgr[mh, :, :].astype(np.int16)
    col = img_bgr[:, mw, :].astype(np.int16)
    row_edge = float(np.abs(row[1:] - row[:-1]).mean())
    col_edge = float(np.abs(col[1:] - col[:-1]).mean())
    quads = [
        img_bgr[:mh, :mw].mean(),
        img_bgr[:mh, mw:].mean(),
        img_bgr[mh:, :mw].mean(),
        img_bgr[mh:, mw:].mean(),
    ]
    return row_edge > 8.0 and col_edge > 8.0 and (max(quads) - min(quads)) > 5.0


def _split_mosaic_quadrants(
    img_bgr: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Split a 2x2 mosaic image into 4 separate tiles with remapped boxes."""
    h, w = img_bgr.shape[:2]
    mh, mw = h // 2, w // 2
    quads = [(0, 0, mw, mh), (mw, 0, w, mh), (0, mh, mw, h), (mw, mh, w, h)]
    tiles: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    boxes_xy = boxes.reshape(-1, 4) if boxes.size else np.zeros((0, 4), dtype=np.float32)
    labels_arr = _as_numpy(labels).astype(int) if labels is not None and len(labels) else np.zeros(0, dtype=int)

    for qx1, qy1, qx2, qy2 in quads:
        tile = img_bgr[qy1:qy2, qx1:qx2].copy()
        t_boxes, t_labels = [], []
        for bi, box in enumerate(boxes_xy):
            x1, y1, x2, y2 = box
            ix1, iy1 = max(x1, qx1), max(y1, qy1)
            ix2, iy2 = min(x2, qx2), min(y2, qy2)
            if ix2 - ix1 < _MIN_BOX_SIDE or iy2 - iy1 < _MIN_BOX_SIDE:
                continue
            t_boxes.append([ix1 - qx1, iy1 - qy1, ix2 - qx1, iy2 - qy1])
            t_labels.append(labels_arr[bi] if bi < len(labels_arr) else 0)
        t_boxes_arr = np.array(t_boxes, dtype=np.float32) if t_boxes else np.zeros((0, 4), dtype=np.float32)
        t_labels_arr = np.array(t_labels, dtype=np.int64) if t_labels else np.zeros(0, dtype=np.int64)
        tiles.append((tile, t_boxes_arr, t_labels_arr))
    return tiles


def _save_one_gt_image(
    img_bgr: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    out_path: str,
    class_names: Optional[List[str]],
) -> str:
    vis = _draw_gt_boxes(img_bgr, boxes, labels, class_names)
    cv2.imwrite(out_path, vis)
    return out_path


def save_raw_gt_images(
    img_dir: str,
    ann_file: str,
    images_dir: str,
    split_name: str,
    class_names: Optional[List[str]] = None,
    num_images: int = 8,
    seed: int = 42,
    log: Optional[logging.Logger] = None,
) -> List[str]:
    """
    Save GT boxes on single-scene images only.

    Roboflow 2x2 mosaic exports are split into separate tile files — the
    combined 4-panel image is never saved.
    """
    import random
    from .dataset import FlashDetDataset

    log = log or logger
    os.makedirs(images_dir, exist_ok=True)
    saved_paths: List[str] = []

    dataset = FlashDetDataset(
        img_dir=img_dir,
        ann_file=ann_file,
        transform=None,
        input_size=(320, 320),
    )
    random.seed(seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    count = 0
    mosaic_sources = 0
    single_sources = 0

    for idx in indices:
        if count >= num_images:
            break

        img_rgb, boxes, labels = dataset.get_raw_item(idx)
        img_id = dataset.img_ids[idx]
        img_info = dataset.images[img_id]
        file_name = img_info.get("file_name", f"id{img_id}")
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        h, w = img_bgr.shape[:2]
        boxes_np = boxes.reshape(-1, 4) if boxes.size else np.zeros((0, 4), dtype=np.float32)
        stem = os.path.splitext(os.path.basename(file_name))[0]

        if _is_mosaic_layout(boxes_np, h, w):
            mosaic_sources += 1
            for qi, (tile, t_boxes, t_labels) in enumerate(
                _split_mosaic_quadrants(img_bgr, boxes_np, labels)
            ):
                if count >= num_images:
                    break
                if len(t_boxes) == 0:
                    continue
                out_name = f"{split_name}_{count:03d}_{stem}_tile{qi}.jpg"
                out_path = os.path.join(images_dir, out_name)
                _save_one_gt_image(tile, t_boxes, t_labels, out_path, class_names)
                saved_paths.append(out_path)
                count += 1
        else:
            single_sources += 1
            out_name = f"{split_name}_{count:03d}_{stem}.jpg"
            out_path = os.path.join(images_dir, out_name)
            _save_one_gt_image(img_bgr, boxes_np, labels, out_path, class_names)
            saved_paths.append(out_path)
            count += 1

    # #region agent log
    try:
        import json as _j, time as _t
        shapes = []
        for p in saved_paths[:3]:
            im = cv2.imread(p)
            if im is not None:
                shapes.append(list(im.shape))
        with open("/home/ggoswami/Project/Gaurav/FlashVision/FlashDet/.cursor/debug-8c3ea2.log", "a") as _f:
            _f.write(_j.dumps({"sessionId": "8c3ea2", "hypothesisId": "H9", "location": "verify_annotations.py:save_raw_gt_images", "message": "gt_save_summary", "data": {"split": split_name, "saved": len(saved_paths), "mosaic_sources": mosaic_sources, "single_sources": single_sources, "has_tile_suffix": sum(1 for p in saved_paths if "_tile" in os.path.basename(p)), "sample_files": [os.path.basename(p) for p in saved_paths[:4]], "sample_shapes": shapes}, "timestamp": int(_t.time() * 1000)}) + "\n")
    except Exception:
        pass
    # #endregion

    log.info(
        "Saved %d GT images (separate scenes/tiles, never 4-in-1) to %s (%s split)",
        len(saved_paths), images_dir, split_name,
    )
    if mosaic_sources:
        log.info("  Split %d mosaic source files into individual tile images", mosaic_sources)
    return saved_paths


def _format_split_summary(split: str, coco: Dict[str, Any], loader: Dict[str, Any], coco_ok: bool, loader_ok: bool) -> List[str]:
    return [
        f"[{split.upper()}]",
        f"  COCO GT:     {'PASS' if coco_ok else 'FAIL'}",
        f"    images={coco['num_images']}  annotations={coco['num_annotations']}  "
        f"missing_files={coco['missing_images']}  oob={coco['oob_boxes']}  "
        f"degenerate={coco['degenerate_boxes']}  unknown_cat={coco['unknown_categories']}  "
        f"empty_images={coco['images_without_anns']}",
        f"    ann_file: {coco['ann_file']}",
        f"    img_dir:  {coco['img_dir']}",
        f"  Dataloader:  {'PASS' if loader_ok else 'FAIL'}",
        f"    batches={loader['batches_checked']}  samples={loader['samples_checked']}  "
        f"boxes={loader['total_boxes']}  avg_boxes={loader['avg_boxes_per_sample']}  "
        f"empty={loader['empty_samples']}  oob={loader['oob_boxes']}  "
        f"degenerate={loader['degenerate_boxes']}  invalid_label={loader['invalid_labels']}",
        "",
    ]


def save_verification_report(
    report: Dict[str, Any],
    output_dir: str,
    log: Optional[logging.Logger] = None,
) -> str:
    """Write JSON + text summary to output_dir."""
    log = log or logger
    abs_output_dir = os.path.abspath(output_dir)
    os.makedirs(abs_output_dir, exist_ok=True)

    json_path = os.path.join(abs_output_dir, "verification_report.json")
    txt_path = os.path.join(abs_output_dir, "verification_summary.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    lines = [
        "FlashDet Dataloader & GT Verification",
        "=" * 50,
        f"Timestamp: {report.get('timestamp', '')}",
        f"Overall:   {'PASS' if report.get('passed') else 'FAIL'}",
        f"Classes:   {report.get('num_classes', '?')}",
        f"GT images: {report.get('gt_images_dir', '')}",
        "",
    ]
    for split in ("train", "val"):
        if split not in report.get("splits", {}):
            continue
        sp = report["splits"][split]
        lines.extend(_format_split_summary(
            split, sp["coco"], sp["dataloader"], sp["coco_ok"], sp["dataloader_ok"],
        ))

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # #region agent log
    try:
        import json as _j, time as _t
        gt_images = report.get("gt_images_saved", [])
        with open("/home/ggoswami/Project/Gaurav/FlashVision/FlashDet/.cursor/debug-8c3ea2.log", "a") as _f:
            _f.write(_j.dumps({"sessionId": "8c3ea2", "hypothesisId": "H6", "location": "verify_annotations.py:save_verification_report", "message": "report_and_images_saved", "data": {"abs_output_dir": abs_output_dir, "gt_images_dir": report.get("gt_images_dir"), "num_gt_images": len(gt_images), "sample_images": gt_images[:3], "json_exists": os.path.isfile(json_path)}, "timestamp": int(_t.time() * 1000)}) + "\n")
    except Exception:
        pass
    # #endregion

    log.info("Verification reports saved to: %s", abs_output_dir)
    log.info("  - verification_report.json")
    log.info("  - verification_summary.txt")
    log.info("  - images/raw/        (original COCO GT)")
    log.info("  - images/dataloader/ (ValTransform pipeline GT)")
    return abs_output_dir


def verify_training_data(
    train_ann_file: str,
    train_img_dir: str,
    save_dir: str,
    val_ann_file: Optional[str] = None,
    val_img_dir: Optional[str] = None,
    num_classes: Optional[int] = None,
    class_names: Optional[List[str]] = None,
    input_size: Tuple[int, int] = (320, 320),
    num_batches: int = 5,
    num_gt_images: int = 8,
    log: Optional[logging.Logger] = None,
) -> bool:
    """
    Run COCO + dataloader checks and save GT box images inside save_dir.

    Output layout::
        {save_dir}/gt_verification/
            verification_report.json
            verification_summary.txt
            images/raw/          ← original COCO GT (single scene / split tiles)
            images/dataloader/   ← after ValTransform pipeline (single scene / split tiles)

    Both COCO checks and dataloader numeric checks run. Images are never saved
    as combined 4-panel mosaics.
    """
    from .dataloader import create_dataloader

    log = log or logger
    verify_dir = gt_verification_dir(save_dir)
    raw_images_dir = os.path.join(verify_dir, GT_IMAGES_SUBDIR, GT_RAW_SUBDIR)
    dl_images_dir = os.path.join(verify_dir, GT_IMAGES_SUBDIR, GT_DATALOADER_SUBDIR)
    _clear_image_dir(raw_images_dir)
    _clear_image_dir(dl_images_dir)

    log.info("=" * 60)
    log.info("Verifying annotations -> %s", verify_dir)
    log.info("=" * 60)

    # ValTransform loaders for numeric dataloader checks (deterministic, single image)
    verify_train_loader = create_dataloader(
        img_dir=train_img_dir,
        ann_file=train_ann_file,
        batch_size=4,
        input_size=input_size,
        num_workers=0,
        is_train=False,
        mosaic=False,
        mixup=False,
        copy_paste=False,
    )
    verify_val_loader = None
    if val_ann_file and val_img_dir:
        verify_val_loader = create_dataloader(
            img_dir=val_img_dir,
            ann_file=val_ann_file,
            batch_size=4,
            input_size=input_size,
            num_workers=0,
            is_train=False,
        )

    all_ok = True
    gt_images_saved: List[str] = []
    report: Dict[str, Any] = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "save_dir": os.path.abspath(save_dir),
        "gt_raw_images_dir": os.path.abspath(raw_images_dir),
        "gt_dataloader_images_dir": os.path.abspath(dl_images_dir),
        "num_classes": num_classes,
        "num_batches_sampled": num_batches,
        "num_gt_images_per_split": num_gt_images,
        "passed": False,
        "splits": {},
    }

    train_coco_ok, train_coco = verify_coco_annotations(
        train_ann_file, train_img_dir, num_classes=num_classes, log=log,
    )
    all_ok = all_ok and train_coco_ok

    train_loader_ok, train_loader_summary = verify_dataloader(
        verify_train_loader, num_batches=num_batches, num_classes=num_classes, log=log,
        collect_samples=True,
    )
    all_ok = all_ok and train_loader_ok

    train_raw_images = save_raw_gt_images(
        train_img_dir, train_ann_file, raw_images_dir, "train", class_names, num_gt_images, log=log,
    )
    train_dl_images = save_gt_verification_images(
        verify_train_loader, dl_images_dir, "train", class_names, num_gt_images, log=log,
    )
    gt_images_saved.extend(train_raw_images)
    gt_images_saved.extend(train_dl_images)

    report["splits"]["train"] = {
        "coco_ok": train_coco_ok,
        "dataloader_ok": train_loader_ok,
        "coco": train_coco,
        "dataloader": train_loader_summary,
        "gt_raw_images": [os.path.basename(p) for p in train_raw_images],
        "gt_dataloader_images": [os.path.basename(p) for p in train_dl_images],
    }

    if verify_val_loader is not None:
        val_coco_ok, val_coco = verify_coco_annotations(
            val_ann_file, val_img_dir, num_classes=num_classes, log=log,
        )
        all_ok = all_ok and val_coco_ok

        val_batches = min(num_batches, len(verify_val_loader))
        val_loader_ok, val_loader_summary = verify_dataloader(
            verify_val_loader, num_batches=val_batches, num_classes=num_classes, log=log,
            collect_samples=True,
        )
        all_ok = all_ok and val_loader_ok

        val_raw_images = save_raw_gt_images(
            val_img_dir, val_ann_file, raw_images_dir, "val", class_names, num_gt_images, log=log,
        )
        val_dl_images = save_gt_verification_images(
            verify_val_loader, dl_images_dir, "val", class_names, num_gt_images, log=log,
        )
        gt_images_saved.extend(val_raw_images)
        gt_images_saved.extend(val_dl_images)

        report["splits"]["val"] = {
            "coco_ok": val_coco_ok,
            "dataloader_ok": val_loader_ok,
            "coco": val_coco,
            "dataloader": val_loader_summary,
            "gt_raw_images": [os.path.basename(p) for p in val_raw_images],
            "gt_dataloader_images": [os.path.basename(p) for p in val_dl_images],
        }

    report["passed"] = all_ok
    report["gt_images_saved"] = [os.path.basename(p) for p in gt_images_saved]

    save_verification_report(report, verify_dir, log=log)

    if all_ok:
        log.info("Annotation verification passed.")
    else:
        log.error(
            "Annotation verification found critical issues. "
            "Fix annotations or use --skip-verify-annotations to proceed anyway."
        )

    log.info("=" * 60)
    return all_ok
