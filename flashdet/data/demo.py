"""Generate a tiny, self-contained COCO-format demo dataset.

The repository used to ship ``data/demo`` annotations without the referenced
images (``*.jpg`` is ignored by git), so the examples could not run from a
clean checkout.  This module generates images and annotations together.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List

from PIL import Image, ImageDraw

DEMO_CLASSES = ["cat", "dog", "car"]


def create_demo_dataset(
    output_dir: str = "data/demo",
    train_images: int = 12,
    val_images: int = 4,
    image_size: int = 320,
    seed: int = 0,
) -> str:
    """Create a small synthetic COCO dataset for examples and smoke tests.

    Args:
        output_dir: Root directory (will contain ``train/`` and ``valid/``).
        train_images: Number of training images.
        val_images: Number of validation images.
        image_size: Square image size.
        seed: Deterministic random seed.

    Returns:
        The output directory path.
    """
    if train_images < 1 or val_images < 1:
        raise ValueError("train_images and val_images must be >= 1")
    if image_size < 32:
        raise ValueError("image_size must be >= 32")

    rng = random.Random(seed)
    root = Path(output_dir)
    categories = [
        {"id": i, "name": name, "supercategory": "demo"}
        for i, name in enumerate(DEMO_CLASSES)
    ]

    for split, count in (("train", train_images), ("valid", val_images)):
        split_dir = root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        images: List[dict] = []
        annotations: List[dict] = []
        ann_id = 1

        for image_id in range(1, count + 1):
            background = tuple(rng.randint(20, 220) for _ in range(3))
            image = Image.new("RGB", (image_size, image_size), background)
            draw = ImageDraw.Draw(image)

            for _ in range(rng.randint(1, 4)):
                w = rng.randint(image_size // 8, image_size // 2)
                h = rng.randint(image_size // 8, image_size // 2)
                x = rng.randint(0, image_size - w)
                y = rng.randint(0, image_size - h)
                color = tuple(rng.randint(0, 255) for _ in range(3))
                draw.rectangle([x, y, x + w, y + h], fill=color,
                               outline=(255, 255, 255), width=2)
                annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": rng.randrange(len(DEMO_CLASSES)),
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                })
                ann_id += 1

            file_name = f"{split}_{image_id:04d}.jpg"
            image.save(split_dir / file_name, quality=90)
            images.append({
                "id": image_id,
                "file_name": file_name,
                "width": image_size,
                "height": image_size,
            })

        payload = {
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }
        with open(split_dir / "_annotations.coco.json", "w", encoding="utf-8") as f:
            json.dump(payload, f)

    return str(root)
