"""Tests for the self-contained demo dataset generator."""

import json
from pathlib import Path

import pytest

from flashdet.data.demo import DEMO_CLASSES, create_demo_dataset


def test_demo_dataset_is_complete(tmp_path):
    root = Path(create_demo_dataset(
        str(tmp_path / "demo"), train_images=2, val_images=1,
        image_size=64, seed=7,
    ))

    for split, count in (("train", 2), ("valid", 1)):
        split_dir = root / split
        assert (split_dir / "_annotations.coco.json").is_file()
        payload = json.loads((split_dir / "_annotations.coco.json").read_text())
        assert len(payload["images"]) == count
        assert [c["name"] for c in payload["categories"]] == DEMO_CLASSES
        for image in payload["images"]:
            assert (split_dir / image["file_name"]).is_file()
        for ann in payload["annotations"]:
            x, y, w, h = ann["bbox"]
            assert w > 0 and h > 0
            assert 0 <= x < 64 and 0 <= y < 64


def test_demo_dataset_rejects_bad_sizes(tmp_path):
    with pytest.raises(ValueError, match=">= 1"):
        create_demo_dataset(str(tmp_path / "x"), train_images=0)
    with pytest.raises(ValueError, match=">= 32"):
        create_demo_dataset(str(tmp_path / "x"), image_size=8)
