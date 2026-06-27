#!/usr/bin/env python3
"""
GT Bounding Box Verification Script.

Draws ground-truth bounding boxes on images at 3 stages to verify
the data pipeline is correct:

  Stage 1 — RAW:       Original image + raw COCO annotations (no transforms)
  Stage 2 — TRANSFORM: After TrainTransform (resize + warp + color jitter)
  Stage 3 — DATALOADER: After full pipeline (mosaic/mixup + transform + collate)

Usage:
    python scripts/verify_gt_boxes.py --train-images data/demo/train \\
        --input-size 320 --n 8 --mosaic --mixup
"""
import argparse
import os
import random
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flashdet.data.dataset import FlashDetDataset
from flashdet.data.dataloader import create_dataloader


COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
    (0, 128, 255), (128, 255, 0), (255, 0, 128), (0, 255, 128),
]

MEAN = np.array([123.675, 116.28, 103.53])
STD = np.array([58.395, 57.12, 57.375])


def draw_boxes(img_bgr, boxes, labels, class_names, title=""):
    """Draw xyxy boxes on a BGR image and return annotated copy."""
    vis = img_bgr.copy()
    h, w = vis.shape[:2]

    for i, (box, lbl) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        lbl_idx = int(lbl)
        name = class_names[lbl_idx] if lbl_idx < len(class_names) else f"cls_{lbl_idx}"
        color = COLORS[lbl_idx % len(COLORS)]
        bw, bh = x2 - x1, y2 - y1

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label_text = f"{name} {bw}x{bh}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(vis, (x1, max(y1 - th - 6, 0)), (x1 + tw + 2, max(y1, th + 6)), color, -1)
        cv2.putText(vis, label_text, (x1 + 1, max(y1 - 4, th + 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    if title:
        cv2.putText(vis, title, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return vis


def denormalize_tensor(img_tensor):
    """Convert CHW float tensor back to HWW BGR uint8 image."""
    img = img_tensor.numpy().transpose(1, 2, 0)  # CHW → HWC
    img = np.clip(img * STD + MEAN, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def main():
    parser = argparse.ArgumentParser(description="Verify GT bounding boxes")
    parser.add_argument("--train-images", required=True, help="Path to training images dir")
    parser.add_argument("--input-size", type=int, default=320)
    parser.add_argument("--n", type=int, default=8, help="Number of samples to visualize")
    parser.add_argument("--mosaic", action="store_true")
    parser.add_argument("--mixup", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="workspace/gt_verify")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ann_file = os.path.join(args.train_images, "_annotations.coco.json")
    if not os.path.exists(ann_file):
        print(f"ERROR: annotation file not found: {ann_file}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    input_size = (args.input_size, args.input_size)

    # --- Load class names ---
    import json
    with open(ann_file) as f:
        coco = json.load(f)
    cat_map = {c["id"]: c["name"] for c in coco["categories"]}
    sorted_ids = sorted(cat_map.keys())
    class_names = [cat_map[cid] for cid in sorted_ids]
    print(f"Classes ({len(class_names)}): {class_names}")

    # ========== STAGE 1: RAW images + raw annotations (no transform) ==========
    print(f"\n{'='*60}")
    print("STAGE 1: RAW images with original COCO annotations")
    print(f"{'='*60}")

    raw_ds = FlashDetDataset(
        img_dir=args.train_images,
        ann_file=ann_file,
        input_size=input_size,
        transform=None,
    )

    indices = random.sample(range(len(raw_ds)), min(args.n, len(raw_ds)))
    for i, idx in enumerate(indices):
        img_rgb, boxes, labels = raw_ds.get_raw_item(idx)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        img_info = raw_ds.images[raw_ds.img_ids[idx]]

        print(f"  [{i}] {img_info['file_name']}  size={img_bgr.shape[1]}x{img_bgr.shape[0]}  "
              f"boxes={len(boxes)}")
        if len(boxes) > 0:
            oob = int(((boxes[:, 2] > img_bgr.shape[1] + 2) |
                        (boxes[:, 3] > img_bgr.shape[0] + 2) |
                        (boxes[:, 0] < -2) | (boxes[:, 1] < -2)).sum())
            print(f"       box_x=[{boxes[:,0].min():.0f}, {boxes[:,2].max():.0f}]  "
                  f"box_y=[{boxes[:,1].min():.0f}, {boxes[:,3].max():.0f}]  oob={oob}")

        vis = draw_boxes(img_bgr, boxes, labels, class_names,
                         title=f"RAW {img_info['file_name']} ({len(boxes)} boxes)")
        path = os.path.join(args.output_dir, f"stage1_raw_{i:02d}.jpg")
        cv2.imwrite(path, vis)
        print(f"       saved: {path}")

    # ========== STAGE 2: After TrainTransform (resize + warp) ==========
    print(f"\n{'='*60}")
    print("STAGE 2: After TrainTransform (resize + spatial aug + color jitter)")
    print(f"{'='*60}")

    from flashdet.data.transforms import TrainTransform
    train_tf = TrainTransform(input_size=input_size)

    for i, idx in enumerate(indices):
        img_rgb, boxes, labels = raw_ds.get_raw_item(idx)
        img_tensor, tf_boxes, tf_labels = train_tf(img_rgb, boxes, labels)

        img_bgr = denormalize_tensor(img_tensor)
        h, w = img_bgr.shape[:2]

        if hasattr(tf_boxes, 'numpy'):
            tf_boxes = tf_boxes.numpy()
        if hasattr(tf_labels, 'numpy'):
            tf_labels = tf_labels.numpy()

        print(f"  [{i}] output={w}x{h}  boxes={len(tf_boxes)}")
        if len(tf_boxes) > 0:
            oob = int(((tf_boxes[:, 2] > w + 2) | (tf_boxes[:, 3] > h + 2) |
                        (tf_boxes[:, 0] < -2) | (tf_boxes[:, 1] < -2)).sum())
            print(f"       box_x=[{tf_boxes[:,0].min():.0f}, {tf_boxes[:,2].max():.0f}]  "
                  f"box_y=[{tf_boxes[:,1].min():.0f}, {tf_boxes[:,3].max():.0f}]  oob={oob}")

        vis = draw_boxes(img_bgr, tf_boxes, tf_labels, class_names,
                         title=f"TRANSFORM ({len(tf_boxes)} boxes, {w}x{h})")
        path = os.path.join(args.output_dir, f"stage2_transform_{i:02d}.jpg")
        cv2.imwrite(path, vis)
        print(f"       saved: {path}")

    # ========== STAGE 3: Full dataloader (mosaic + mixup + transform) ==========
    print(f"\n{'='*60}")
    augs = []
    if args.mosaic:
        augs.append("mosaic")
    if args.mixup:
        augs.append("mixup")
    aug_str = "+".join(augs) if augs else "none"
    print(f"STAGE 3: Full dataloader output (augmentations: {aug_str})")
    print(f"{'='*60}")

    loader = create_dataloader(
        img_dir=args.train_images,
        ann_file=ann_file,
        batch_size=4,
        input_size=input_size,
        num_workers=0,
        is_train=True,
        mosaic=args.mosaic,
        mixup=args.mixup,
    )

    sample_count = 0
    for batch_idx, (images, gt_meta) in enumerate(loader):
        for j in range(images.shape[0]):
            if sample_count >= args.n:
                break

            img_bgr = denormalize_tensor(images[j])
            h, w = img_bgr.shape[:2]
            boxes = gt_meta["gt_bboxes"][j]
            labels = gt_meta["gt_labels"][j]

            if hasattr(boxes, 'numpy'):
                boxes = boxes.numpy()
            if hasattr(labels, 'numpy'):
                labels = labels.numpy()
            boxes = np.array(boxes) if not isinstance(boxes, np.ndarray) else boxes

            print(f"  [{sample_count}] batch={batch_idx} idx={j}  img={w}x{h}  boxes={len(boxes)}")
            if len(boxes) > 0:
                oob = int(((boxes[:, 2] > w + 2) | (boxes[:, 3] > h + 2) |
                            (boxes[:, 0] < -2) | (boxes[:, 1] < -2)).sum())
                ws = boxes[:, 2] - boxes[:, 0]
                hs = boxes[:, 3] - boxes[:, 1]
                degen = int((ws < 2).sum() + (hs < 2).sum())
                print(f"       box_x=[{boxes[:,0].min():.0f}, {boxes[:,2].max():.0f}]  "
                      f"box_y=[{boxes[:,1].min():.0f}, {boxes[:,3].max():.0f}]  "
                      f"oob={oob}  degenerate={degen}")

            vis = draw_boxes(img_bgr, boxes, labels, class_names,
                             title=f"DATALOADER b{batch_idx}s{j} ({len(boxes)} boxes, {aug_str})")
            path = os.path.join(args.output_dir, f"stage3_loader_{sample_count:02d}.jpg")
            cv2.imwrite(path, vis)
            print(f"       saved: {path}")
            sample_count += 1

        if sample_count >= args.n:
            break

    print(f"\n{'='*60}")
    print(f"All {sample_count * 3} images saved to: {args.output_dir}/")
    print(f"  stage1_raw_*.jpg       — raw images with original COCO boxes")
    print(f"  stage2_transform_*.jpg — after TrainTransform")
    print(f"  stage3_loader_*.jpg    — full dataloader output")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
