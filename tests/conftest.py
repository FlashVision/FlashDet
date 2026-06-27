"""Root conftest — shared fixtures and markers for the FlashDet test suite.

Pytest Markers
--------------
- ``@pytest.mark.slow``: Tests that take > 10s (skipped unless ``--runslow``)
- ``@pytest.mark.gpu``: Tests requiring CUDA (skipped on CPU-only CI)
- ``@pytest.mark.integration``: End-to-end pipeline tests
- ``@pytest.mark.smoke``: Fast sanity checks (~5s total)

Fixtures
--------
- ``device``: "cuda" if available, else "cpu"
- ``sample_image``: 640x480 random uint8 numpy image
- ``sample_batch``: (2, 3, 320, 320) float32 tensor
- ``flashdet_model``: FlashDet-N model instance (eval mode)
- ``gt_meta``: Synthetic ground-truth metadata for training tests
- ``tmp_dataset_dir``: Temporary COCO dataset directory
"""

import json
import os
import tempfile
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# CLI hooks
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="Run slow tests")
    parser.addoption("--rungpu", action="store_true", default=False, help="Run GPU-only tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "gpu: marks tests as requiring CUDA GPU")
    config.addinivalue_line("markers", "integration: end-to-end pipeline tests")
    config.addinivalue_line("markers", "smoke: fast sanity checks")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    if not config.getoption("--rungpu") and not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="CUDA not available (use --rungpu to force)")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


# ---------------------------------------------------------------------------
# Device fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def device():
    """Return best available device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Image / Tensor fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_image():
    """Random 640x480 uint8 BGR image (like cv2.imread output)."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_image_320():
    """Random 320x320 uint8 BGR image."""
    return np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)


@pytest.fixture
def sample_batch():
    """(2, 3, 320, 320) float32 tensor — typical training batch."""
    return torch.randn(2, 3, 320, 320)


@pytest.fixture
def sample_batch_640():
    """(2, 3, 640, 640) float32 tensor — YOLO-size batch."""
    return torch.randn(2, 3, 640, 640)


# ---------------------------------------------------------------------------
# Model fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def flashdet_model():
    """FlashDet-N model in eval mode (num_classes=10, input_size=320)."""
    from flashdet.models.detector import FlashDet
    model = FlashDet(num_classes=10, size="n")
    model.eval()
    return model


@pytest.fixture
def flashdet_model_train():
    """FlashDet-N model in train mode."""
    from flashdet.models.detector import FlashDet
    model = FlashDet(num_classes=5, size="n")
    model.train()
    return model


# ---------------------------------------------------------------------------
# Ground-truth fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def gt_meta():
    """Synthetic ground-truth metadata for training (batch_size=2, num_classes=5)."""
    np.random.seed(42)
    gt = {"gt_bboxes": [], "gt_labels": []}
    for _ in range(2):
        n_objs = np.random.randint(2, 6)
        x1y1 = np.random.rand(n_objs, 2).astype(np.float32) * 200
        wh = np.random.rand(n_objs, 2).astype(np.float32) * 80 + 20
        boxes = np.concatenate([x1y1, x1y1 + wh], axis=1)
        boxes = np.clip(boxes, 0, 319)
        labels = np.random.randint(0, 5, size=(n_objs,)).astype(np.int64)
        gt["gt_bboxes"].append(boxes)
        gt["gt_labels"].append(labels)
    return gt


@pytest.fixture
def gt_meta_empty():
    """Empty ground-truth (no objects) for edge-case testing."""
    return {
        "gt_bboxes": [np.zeros((0, 4), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)],
        "gt_labels": [np.array([], dtype=np.int64), np.array([], dtype=np.int64)],
    }


# ---------------------------------------------------------------------------
# Temporary dataset fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dataset_dir():
    """Create a temporary minimal COCO dataset and clean up after test."""
    tmp = tempfile.mkdtemp(prefix="flashdet_test_")

    for split in ["train", "valid"]:
        split_dir = os.path.join(tmp, split)
        os.makedirs(split_dir)

        n_images = 10 if split == "train" else 3
        images = []
        annotations = []
        ann_id = 1

        for i in range(1, n_images + 1):
            images.append({"id": i, "file_name": f"img_{i:04d}.jpg", "width": 640, "height": 480})
            n_objs = np.random.randint(1, 5)
            for _ in range(n_objs):
                x = float(np.random.randint(0, 500))
                y = float(np.random.randint(0, 350))
                w = float(np.random.randint(20, 140))
                h = float(np.random.randint(20, 130))
                annotations.append({
                    "id": ann_id, "image_id": i,
                    "category_id": int(np.random.randint(0, 3)),
                    "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0,
                })
                ann_id += 1

            # Create dummy JPEG
            from PIL import Image
            img = Image.new("RGB", (640, 480), color=(
                np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)
            ))
            img.save(os.path.join(split_dir, f"img_{i:04d}.jpg"))

        coco = {
            "images": images,
            "annotations": annotations,
            "categories": [
                {"id": 0, "name": "cat", "supercategory": "animal"},
                {"id": 1, "name": "dog", "supercategory": "animal"},
                {"id": 2, "name": "car", "supercategory": "vehicle"},
            ],
        }
        with open(os.path.join(split_dir, "_annotations.coco.json"), "w") as f:
            json.dump(coco, f)

    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Determinism helper
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=False)
def seed_everything():
    """Fix random seeds for reproducibility."""
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
