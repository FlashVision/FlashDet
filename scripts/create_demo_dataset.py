#!/usr/bin/env python3
"""Create a tiny synthetic COCO dataset for examples and smoke tests.

Usage:
    python scripts/create_demo_dataset.py
    python scripts/create_demo_dataset.py --output data/demo --size 320
"""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from flashdet.data.demo import create_demo_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a synthetic FlashDet demo dataset")
    parser.add_argument("--output", default="data/demo", help="Output directory")
    parser.add_argument("--train", type=int, default=12, help="Training images")
    parser.add_argument("--val", type=int, default=4, help="Validation images")
    parser.add_argument("--size", type=int, default=320, help="Image size")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    root = create_demo_dataset(
        output_dir=args.output,
        train_images=args.train,
        val_images=args.val,
        image_size=args.size,
        seed=args.seed,
    )
    print(f"Demo dataset created: {root}")
    print(f"  train: {args.output}/train")
    print(f"  valid: {args.output}/valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
