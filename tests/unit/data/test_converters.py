"""Tests for dataset format converters — COCO ↔ YOLO TXT ↔ VOC."""

import json
import os
import tempfile
import shutil

import numpy as np
import pytest
from PIL import Image

from flashdet.data.prepare import (
    convert_coco_to_txt,
    convert_coco_to_voc,
    convert_dataset,
    detect_dataset_format,
)


@pytest.fixture
def coco_dataset():
    """Create a small COCO-format dataset in a temp directory."""
    tmp = tempfile.mkdtemp(prefix="flashdet_coco_")
    train_dir = os.path.join(tmp, "train")
    os.makedirs(train_dir)

    coco = {
        "images": [
            {"id": 1, "file_name": "img1.jpg", "width": 640, "height": 480},
            {"id": 2, "file_name": "img2.jpg", "width": 640, "height": 480},
            {"id": 3, "file_name": "img3.jpg", "width": 320, "height": 240},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 0, "bbox": [100, 100, 200, 150], "area": 30000, "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [300, 200, 100, 100], "area": 10000, "iscrowd": 0},
            {"id": 3, "image_id": 2, "category_id": 0, "bbox": [50, 50, 300, 250], "area": 75000, "iscrowd": 0},
            {"id": 4, "image_id": 3, "category_id": 2, "bbox": [10, 10, 100, 80], "area": 8000, "iscrowd": 0},
        ],
        "categories": [
            {"id": 0, "name": "cat", "supercategory": "animal"},
            {"id": 1, "name": "dog", "supercategory": "animal"},
            {"id": 2, "name": "car", "supercategory": "vehicle"},
        ],
    }
    with open(os.path.join(train_dir, "_annotations.coco.json"), "w") as f:
        json.dump(coco, f)

    for info in coco["images"]:
        img = Image.new("RGB", (info["width"], info["height"]), color="blue")
        img.save(os.path.join(train_dir, info["file_name"]))

    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def yolo_dataset():
    """Create a small YOLO TXT-format dataset in a temp directory."""
    tmp = tempfile.mkdtemp(prefix="flashdet_yolo_")
    for split in ["train", "valid"]:
        img_dir = os.path.join(tmp, split, "images")
        lbl_dir = os.path.join(tmp, split, "labels")
        os.makedirs(img_dir)
        os.makedirs(lbl_dir)
        for i in range(3):
            img = Image.new("RGB", (640, 480), color="green")
            img.save(os.path.join(img_dir, f"img_{i}.jpg"))
            with open(os.path.join(lbl_dir, f"img_{i}.txt"), "w") as f:
                f.write(f"0 0.5 0.5 0.3 0.4\n1 0.2 0.3 0.1 0.2\n")

    with open(os.path.join(tmp, "data.yaml"), "w") as f:
        f.write("names: ['cat', 'dog']\nnc: 2\n")

    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


class TestDetectFormat:
    """Test format auto-detection."""

    def test_detect_coco(self, coco_dataset):
        assert detect_dataset_format(coco_dataset) == "coco"

    def test_detect_yolo_txt(self, yolo_dataset):
        assert detect_dataset_format(yolo_dataset) == "txt"

    def test_detect_unknown(self, tmp_path):
        os.makedirs(tmp_path / "empty_dir")
        assert detect_dataset_format(str(tmp_path / "empty_dir")) == "unknown"


class TestConvertCOCOToTXT:
    """COCO → YOLO TXT conversion."""

    def test_creates_labels(self, coco_dataset):
        out = os.path.join(coco_dataset, "output_txt")
        stats = convert_coco_to_txt(coco_dataset, out)
        assert "train" in stats
        assert stats["train"]["images"] == 3
        assert stats["train"]["labels"] > 0

    def test_label_format_valid(self, coco_dataset):
        out = os.path.join(coco_dataset, "output_txt")
        convert_coco_to_txt(coco_dataset, out)
        lbl_dir = os.path.join(out, "train", "labels")
        for lbl_file in os.listdir(lbl_dir):
            with open(os.path.join(lbl_dir, lbl_file)) as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    assert len(parts) == 5
                    cls_id = int(parts[0])
                    assert cls_id >= 0
                    for val in parts[1:]:
                        v = float(val)
                        assert 0.0 <= v <= 1.0

    def test_creates_data_yaml(self, coco_dataset):
        out = os.path.join(coco_dataset, "output_txt")
        convert_coco_to_txt(coco_dataset, out)
        assert os.path.isfile(os.path.join(out, "data.yaml"))


class TestConvertCOCOToVOC:
    """COCO → Pascal VOC conversion."""

    def test_creates_xml_files(self, coco_dataset):
        out = os.path.join(coco_dataset, "output_voc")
        stats = convert_coco_to_voc(coco_dataset, out)
        ann_dir = os.path.join(out, "Annotations")
        assert os.path.isdir(ann_dir)
        xml_files = [f for f in os.listdir(ann_dir) if f.endswith(".xml")]
        assert len(xml_files) == 3

    def test_xml_has_objects(self, coco_dataset):
        import xml.etree.ElementTree as ET

        out = os.path.join(coco_dataset, "output_voc")
        convert_coco_to_voc(coco_dataset, out)
        xml_path = os.path.join(out, "Annotations", "img1.xml")
        tree = ET.parse(xml_path)
        objects = tree.getroot().findall("object")
        assert len(objects) == 2
        names = [obj.find("name").text for obj in objects]
        assert "cat" in names
        assert "dog" in names

    def test_creates_imagesets(self, coco_dataset):
        out = os.path.join(coco_dataset, "output_voc")
        convert_coco_to_voc(coco_dataset, out)
        sets_dir = os.path.join(out, "ImageSets", "Main")
        assert os.path.isdir(sets_dir)


class TestConvertDatasetUniversal:
    """Universal convert_dataset() function."""

    def test_noop_same_format(self, coco_dataset):
        result = convert_dataset(coco_dataset, target_format="coco")
        assert result["status"] == "already_in_target_format"

    def test_coco_to_txt(self, coco_dataset):
        out = os.path.join(coco_dataset, "out_txt")
        result = convert_dataset(coco_dataset, output_dir=out, target_format="txt")
        assert "train" in result
        assert os.path.isdir(os.path.join(out, "train", "labels"))

    def test_coco_to_voc(self, coco_dataset):
        out = os.path.join(coco_dataset, "out_voc")
        result = convert_dataset(coco_dataset, output_dir=out, target_format="voc")
        assert "train" in result
        assert os.path.isdir(os.path.join(out, "Annotations"))

    def test_yolo_to_coco(self, yolo_dataset):
        out = os.path.join(yolo_dataset, "out_coco")
        result = convert_dataset(yolo_dataset, output_dir=out, target_format="coco")
        assert "train" in result

    def test_invalid_target_raises(self, coco_dataset):
        with pytest.raises(ValueError, match="target_format"):
            convert_dataset(coco_dataset, target_format="invalid_format")

    def test_unknown_source_raises(self, tmp_path):
        empty = str(tmp_path / "empty")
        os.makedirs(empty)
        with pytest.raises(ValueError, match="Could not detect"):
            convert_dataset(empty, target_format="coco")
