#!/usr/bin/env python3
"""Create a synthetic detection dataset for training/benchmark smoke tests.

The generator draws class-specific colored shapes, so a model can actually
learn the 10-class task (unlike purely random labels).

Usage:
    python scripts/create_synthetic_dataset.py --output /tmp/synth10
    python scripts/create_synthetic_dataset.py --train 1000 --val 200 --size 192
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

COLORS = [
    (220, 40, 40), (40, 180, 60), (40, 100, 230), (230, 190, 30),
    (180, 50, 200), (30, 200, 200), (240, 120, 30), (120, 80, 40),
    (240, 120, 180), (90, 90, 90),
]


def _draw_shape(draw: ImageDraw.ImageDraw, cls: int, box: list[int]) -> None:
    x1, y1, x2, y2 = box
    color = COLORS[cls % len(COLORS)]
    kind = cls % 5
    if kind == 0:
        draw.ellipse(box, fill=color, outline=(255, 255, 255), width=2)
    elif kind == 1:
        draw.rectangle(box, fill=color, outline=(255, 255, 255), width=2)
    elif kind == 2:
        draw.polygon([(x1, y2), ((x1 + x2) // 2, y1), (x2, y2)],
                     fill=color, outline=(255, 255, 255))
    elif kind == 3:
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        draw.polygon([(cx, y1), (x2, cy), (cx, y2), (x1, cy)],
                     fill=color, outline=(255, 255, 255))
    else:
        width = max(4, (x2 - x1) // 5)
        cy = (y1 + y2) // 2
        draw.line([(x1, cy), (x2, cy)], fill=color, width=width)
        cx = (x1 + x2) // 2
        draw.line([(cx, y1), (cx, y2)], fill=color, width=width)


def _make_split(root: Path, split: str, count: int, size: int,
                num_classes: int, rng: random.Random) -> None:
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    ann_id = 1

    for image_id in range(1, count + 1):
        background = tuple(rng.randint(30, 210) for _ in range(3))
        image = Image.new("RGB", (size, size), background)
        draw = ImageDraw.Draw(image)

        for _ in range(rng.randint(1, 3)):
            cls = rng.randrange(num_classes)
            w = rng.randint(max(16, size // 8), max(20, size // 2))
            h = rng.randint(max(16, size // 8), max(20, size // 2))
            x = rng.randint(0, size - w)
            y = rng.randint(0, size - h)
            box = [x, y, x + w, y + h]
            _draw_shape(draw, cls, box)
            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": cls,
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
            })
            ann_id += 1

        file_name = f"{split}_{image_id:05d}.jpg"
        image.save(split_dir / file_name, quality=88)
        images.append({
            "id": image_id,
            "file_name": file_name,
            "width": size,
            "height": size,
        })

    categories = [
        {"id": i, "name": f"class_{i}", "supercategory": "synthetic"}
        for i in range(num_classes)
    ]
    with open(split_dir / "_annotations.coco.json", "w", encoding="utf-8") as f:
        json.dump(
            {"images": images, "annotations": annotations, "categories": categories},
            f,
        )


def create_synthetic_dataset(
    output_dir: str,
    train_images: int = 1000,
    val_images: int = 200,
    image_size: int = 192,
    num_classes: int = 10,
    seed: int = 42,
) -> str:
    if num_classes < 1:
        raise ValueError("num_classes must be >= 1")
    if train_images < 1 or val_images < 1:
        raise ValueError("train_images and val_images must be >= 1")

    root = Path(output_dir)
    rng = random.Random(seed)
    _make_split(root, "train", train_images, image_size, num_classes, rng)
    _make_split(root, "valid", val_images, image_size, num_classes, rng)
    return str(root)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a synthetic COCO dataset")
    parser.add_argument("--output", default="/tmp/flashdet-synthetic")
    parser.add_argument("--train", type=int, default=1000)
    parser.add_argument("--val", type=int, default=200)
    parser.add_argument("--size", type=int, default=192)
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = create_synthetic_dataset(
        args.output, args.train, args.val, args.size, args.classes, args.seed,
    )
    print(f"Created synthetic dataset: {root}")
    print(f"  images: {args.train} train / {args.val} valid")
    print(f"  classes: {args.classes}, size: {args.size}px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
