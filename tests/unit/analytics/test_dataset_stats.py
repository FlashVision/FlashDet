"""Tests for DatasetAnalyzer — dataset composition analysis."""

import json
import os
import tempfile
import shutil

import numpy as np
import pytest
from PIL import Image

from flashdet.analytics.dataset_stats import DatasetAnalyzer


@pytest.fixture
def coco_annotations(tmp_path):
    """Create a minimal COCO annotation file."""
    coco = {
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": 640, "height": 480},
            {"id": 2, "file_name": "b.jpg", "width": 800, "height": 600},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 0, "bbox": [10, 10, 100, 80]},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [200, 200, 50, 60]},
            {"id": 3, "image_id": 2, "category_id": 0, "bbox": [50, 50, 200, 150]},
        ],
        "categories": [
            {"id": 0, "name": "cat"},
            {"id": 1, "name": "dog"},
        ],
    }
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(coco))
    return str(path)


@pytest.fixture
def yolo_labels(tmp_path):
    """Create minimal YOLO label files."""
    lbl_dir = tmp_path / "labels"
    lbl_dir.mkdir()
    (lbl_dir / "img1.txt").write_text("0 0.5 0.5 0.3 0.4\n1 0.2 0.3 0.1 0.15\n")
    (lbl_dir / "img2.txt").write_text("0 0.7 0.7 0.2 0.2\n")
    return str(lbl_dir)


class TestDatasetAnalyzerCOCO:
    """COCO-format analysis."""

    def test_basic_stats(self, coco_annotations):
        analyzer = DatasetAnalyzer(annotation_path=coco_annotations, class_names=["cat", "dog"])
        stats = analyzer.analyze()
        assert stats["num_images"] == 2
        assert stats["num_annotations"] == 3
        assert stats["num_classes"] == 2

    def test_class_distribution(self, coco_annotations):
        analyzer = DatasetAnalyzer(annotation_path=coco_annotations, class_names=["cat", "dog"])
        stats = analyzer.analyze()
        dist = dict(stats["class_distribution"])
        assert dist["cat"] == 2
        assert dist["dog"] == 1

    def test_objects_per_image(self, coco_annotations):
        analyzer = DatasetAnalyzer(annotation_path=coco_annotations)
        stats = analyzer.analyze()
        opi = stats["objects_per_image"]
        assert opi["mean"] == 1.5
        assert opi["min"] == 1
        assert opi["max"] == 2

    def test_bbox_sizes(self, coco_annotations):
        analyzer = DatasetAnalyzer(annotation_path=coco_annotations)
        stats = analyzer.analyze()
        bs = stats["bbox_sizes"]
        assert bs["width_mean"] > 0
        assert bs["height_mean"] > 0
        assert bs["area_mean"] > 0

    def test_class_balance(self, coco_annotations):
        analyzer = DatasetAnalyzer(annotation_path=coco_annotations)
        stats = analyzer.analyze()
        cb = stats["class_balance"]
        assert cb["imbalance_ratio"] == 2.0
        assert 0 < cb["balance_score"] <= 1.0

    def test_summary_string(self, coco_annotations):
        analyzer = DatasetAnalyzer(annotation_path=coco_annotations, class_names=["cat", "dog"])
        summary = analyzer.summary()
        assert "FlashDet Dataset Analysis" in summary
        assert "cat" in summary


class TestDatasetAnalyzerYOLO:
    """YOLO TXT-format analysis."""

    def test_basic_stats(self, yolo_labels):
        analyzer = DatasetAnalyzer(label_dir=yolo_labels, class_names=["cat", "dog"])
        stats = analyzer.analyze()
        assert stats["num_images"] == 2
        assert stats["num_annotations"] == 3

    def test_objects_per_image(self, yolo_labels):
        analyzer = DatasetAnalyzer(label_dir=yolo_labels)
        stats = analyzer.analyze()
        assert stats["objects_per_image"]["max"] == 2

    def test_requires_annotation_or_labels(self):
        with pytest.raises(ValueError):
            DatasetAnalyzer(annotation_path=None, label_dir=None)
