"""
Dataset download utilities for popular open-source detection datasets.

Supported datasets:
  - COCO 2017 (train/val/test splits)
  - Pascal VOC 2007 / 2012
  - Open Images V7 (subset download)
  - Sample/toy datasets for quick testing

All datasets are downloaded into COCO JSON format so they can be used
directly with FlashDet's training pipeline.
"""

import hashlib
import json
import logging
import os
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional
from urllib.request import urlretrieve
from urllib.error import URLError

from tqdm import tqdm

logger = logging.getLogger(__name__)

DATASET_REGISTRY: Dict[str, dict] = {
    "coco2017": {
        "name": "COCO 2017",
        "description": "Microsoft COCO: 118K train, 5K val images, 80 classes",
        "urls": {
            "train_images": "http://images.cocodataset.org/zips/train2017.zip",
            "val_images": "http://images.cocodataset.org/zips/val2017.zip",
            "train_annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        },
        "format": "coco",
        "classes": 80,
        "post_process": "_setup_coco2017",
    },
    "coco2017-val": {
        "name": "COCO 2017 (val only)",
        "description": "COCO 2017 validation set only — 5K images, 80 classes (quick testing)",
        "urls": {
            "val_images": "http://images.cocodataset.org/zips/val2017.zip",
            "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        },
        "format": "coco",
        "classes": 80,
        "post_process": "_setup_coco2017_val",
    },
    "voc2007": {
        "name": "Pascal VOC 2007",
        "description": "Pascal VOC 2007: 5K train/val + 5K test, 20 classes",
        "urls": {
            "trainval": "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
            "test": "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar",
        },
        "format": "voc",
        "classes": 20,
        "post_process": "_setup_voc",
    },
    "voc2012": {
        "name": "Pascal VOC 2012",
        "description": "Pascal VOC 2012: 11.5K train/val, 20 classes",
        "urls": {
            "trainval": "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
        },
        "format": "voc",
        "classes": 20,
        "post_process": "_setup_voc",
    },
    "sample": {
        "name": "Sample Dataset",
        "description": "Tiny 50-image sample from COCO val for quick smoke tests",
        "urls": {
            "val_images": "http://images.cocodataset.org/zips/val2017.zip",
            "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        },
        "format": "coco",
        "classes": 80,
        "post_process": "_setup_sample",
    },
}


class _DownloadProgress:
    """tqdm-based download progress bar for urlretrieve."""

    def __init__(self, desc: str = "Downloading"):
        self.pbar = None
        self.desc = desc

    def __call__(self, block_num, block_size, total_size):
        if self.pbar is None:
            self.pbar = tqdm(
                total=total_size if total_size > 0 else None,
                unit="B",
                unit_scale=True,
                desc=self.desc,
            )
        downloaded = block_num * block_size
        if self.pbar.total and downloaded > self.pbar.total:
            downloaded = self.pbar.total
        self.pbar.n = downloaded
        self.pbar.refresh()

    def close(self):
        if self.pbar:
            self.pbar.close()


def _download_file(url: str, dest: str, desc: str = "Downloading") -> str:
    """Download a file with progress bar, skipping if it already exists."""
    if os.path.isfile(dest):
        logger.info("Already downloaded: %s", os.path.basename(dest))
        return dest

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    logger.info("Downloading %s ...", url)

    progress = _DownloadProgress(desc=desc)
    try:
        urlretrieve(url, dest, reporthook=progress)
    except (URLError, OSError) as e:
        progress.close()
        if os.path.exists(dest):
            os.remove(dest)
        raise RuntimeError(f"Download failed: {url}\n{e}") from e
    finally:
        progress.close()

    return dest


def _extract(archive_path: str, dest_dir: str) -> None:
    """Extract a .zip or .tar/.tar.gz archive."""
    logger.info("Extracting %s ...", os.path.basename(archive_path))
    os.makedirs(dest_dir, exist_ok=True)

    if archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(dest_dir)
    elif archive_path.endswith((".tar", ".tar.gz", ".tgz")):
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(dest_dir)
    else:
        raise ValueError(f"Unknown archive format: {archive_path}")


def _setup_coco2017(download_dir: str, output_dir: str) -> str:
    """Rearrange downloaded COCO 2017 files into FlashDet layout."""
    ann_dir = os.path.join(download_dir, "annotations")
    train_img_src = os.path.join(download_dir, "train2017")
    val_img_src = os.path.join(download_dir, "val2017")

    train_out = os.path.join(output_dir, "train")
    val_out = os.path.join(output_dir, "valid")
    os.makedirs(train_out, exist_ok=True)
    os.makedirs(val_out, exist_ok=True)

    train_img_dst = os.path.join(train_out, "images")
    val_img_dst = os.path.join(val_out, "images")

    if os.path.isdir(train_img_src) and not os.path.exists(train_img_dst):
        os.symlink(os.path.abspath(train_img_src), train_img_dst)
    if os.path.isdir(val_img_src) and not os.path.exists(val_img_dst):
        os.symlink(os.path.abspath(val_img_src), val_img_dst)

    train_ann_src = os.path.join(ann_dir, "instances_train2017.json")
    val_ann_src = os.path.join(ann_dir, "instances_val2017.json")

    _convert_coco_official_to_flashdet(train_ann_src, train_out, train_img_src)
    _convert_coco_official_to_flashdet(val_ann_src, val_out, val_img_src)

    return output_dir


def _setup_coco2017_val(download_dir: str, output_dir: str) -> str:
    """Set up COCO 2017 val-only as both train and valid (for testing)."""
    ann_dir = os.path.join(download_dir, "annotations")
    val_img_src = os.path.join(download_dir, "val2017")

    for split in ("train", "valid"):
        split_out = os.path.join(output_dir, split)
        os.makedirs(split_out, exist_ok=True)

        img_dst = os.path.join(split_out, "images")
        if os.path.isdir(val_img_src) and not os.path.exists(img_dst):
            os.symlink(os.path.abspath(val_img_src), img_dst)

    val_ann_src = os.path.join(ann_dir, "instances_val2017.json")
    for split in ("train", "valid"):
        _convert_coco_official_to_flashdet(
            val_ann_src, os.path.join(output_dir, split), val_img_src
        )

    return output_dir


def _setup_sample(download_dir: str, output_dir: str) -> str:
    """Create a tiny sample dataset from COCO val (50 images)."""
    ann_dir = os.path.join(download_dir, "annotations")
    val_img_src = os.path.join(download_dir, "val2017")
    val_ann_src = os.path.join(ann_dir, "instances_val2017.json")

    if not os.path.isfile(val_ann_src):
        raise FileNotFoundError(f"Annotations not found: {val_ann_src}")

    with open(val_ann_src) as f:
        coco_data = json.load(f)

    sample_images = sorted(coco_data["images"], key=lambda x: x["id"])[:50]
    sample_ids = {img["id"] for img in sample_images}
    sample_anns = [a for a in coco_data["annotations"] if a["image_id"] in sample_ids]

    split_at = 40
    train_imgs = sample_images[:split_at]
    val_imgs = sample_images[split_at:]
    train_ids = {img["id"] for img in train_imgs}
    val_ids = {img["id"] for img in val_imgs}

    for split_name, split_imgs, split_ids in [
        ("train", train_imgs, train_ids),
        ("valid", val_imgs, val_ids),
    ]:
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        for img_info in split_imgs:
            src = os.path.join(val_img_src, img_info["file_name"])
            dst = os.path.join(split_dir, img_info["file_name"])
            if os.path.isfile(src) and not os.path.exists(dst):
                try:
                    os.symlink(os.path.abspath(src), dst)
                except OSError:
                    shutil.copy2(src, dst)

        split_anns = [a for a in sample_anns if a["image_id"] in split_ids]
        coco_out = {
            "images": split_imgs,
            "annotations": split_anns,
            "categories": coco_data["categories"],
        }
        ann_path = os.path.join(split_dir, "_annotations.coco.json")
        with open(ann_path, "w") as f:
            json.dump(coco_out, f)
        logger.info("Sample %s: %d images, %d annotations", split_name, len(split_imgs), len(split_anns))

    return output_dir


def _setup_voc(download_dir: str, output_dir: str) -> str:
    """Convert downloaded Pascal VOC to COCO format."""
    from flashdet.data.prepare import convert_voc_to_coco

    voc_root = None
    for root, dirs, _ in os.walk(download_dir):
        if "Annotations" in dirs and ("JPEGImages" in dirs or "images" in dirs):
            voc_root = root
            break

    if voc_root is None:
        raise FileNotFoundError(
            f"Could not find VOC dataset layout (Annotations + JPEGImages) under {download_dir}"
        )

    convert_voc_to_coco(voc_root, output_dir)
    return output_dir


def _convert_coco_official_to_flashdet(
    ann_file: str, output_dir: str, img_dir: str
) -> None:
    """Convert official COCO annotation JSON to FlashDet's expected layout.

    FlashDet expects _annotations.coco.json alongside images in the split directory.
    Official COCO has separate annotation files and image directories.
    """
    if not os.path.isfile(ann_file):
        logger.warning("Annotation file not found: %s", ann_file)
        return

    dst_ann = os.path.join(output_dir, "_annotations.coco.json")
    if os.path.isfile(dst_ann):
        return

    with open(ann_file) as f:
        data = json.load(f)

    img_link_dir = output_dir
    for img_info in data.get("images", []):
        src = os.path.join(img_dir, img_info["file_name"])
        dst = os.path.join(img_link_dir, img_info["file_name"])
        if os.path.isfile(src) and not os.path.exists(dst):
            try:
                os.symlink(os.path.abspath(src), dst)
            except OSError:
                pass

    with open(dst_ann, "w") as f:
        json.dump(data, f)

    logger.info("Created %s (%d images, %d annotations)",
                dst_ann, len(data.get("images", [])), len(data.get("annotations", [])))


def list_datasets() -> List[Dict[str, str]]:
    """Return list of available datasets with name and description."""
    return [
        {
            "id": key,
            "name": info["name"],
            "description": info["description"],
            "classes": info["classes"],
            "format": info["format"],
        }
        for key, info in DATASET_REGISTRY.items()
    ]


def download_dataset(
    dataset_id: str,
    output_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> str:
    """Download and prepare a dataset for FlashDet training.

    Args:
        dataset_id: One of the registered dataset IDs (e.g. "coco2017", "voc2007").
        output_dir: Where to place the prepared dataset. Defaults to ``data/<dataset_id>``.
        cache_dir: Where to cache downloaded archives. Defaults to ``~/.cache/flashdet/``.

    Returns:
        Path to the prepared dataset directory (COCO format, ready for training).
    """
    if dataset_id not in DATASET_REGISTRY:
        available = ", ".join(DATASET_REGISTRY.keys())
        raise ValueError(f"Unknown dataset '{dataset_id}'. Available: {available}")

    info = DATASET_REGISTRY[dataset_id]

    if output_dir is None:
        output_dir = os.path.join("data", dataset_id)
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "flashdet")

    download_dir = os.path.join(cache_dir, dataset_id)
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Downloading: {info['name']}")
    print(f"Description: {info['description']}")
    print(f"Output: {os.path.abspath(output_dir)}")
    print(f"Cache:  {os.path.abspath(download_dir)}")
    print(f"{'='*60}\n")

    for key, url in info["urls"].items():
        ext = url.rsplit(".", 1)[-1]
        if ext in ("gz", "tgz"):
            ext = "tar.gz"
        archive_path = os.path.join(download_dir, f"{key}.{ext}")

        _download_file(url, archive_path, desc=f"Downloading {key}")
        _extract(archive_path, download_dir)

    post_fn_name = info["post_process"]
    post_fn = globals().get(post_fn_name)
    if post_fn is None:
        raise RuntimeError(f"Post-processing function not found: {post_fn_name}")

    result_dir = post_fn(download_dir, output_dir)

    from flashdet.data.prepare import verify_dataset
    verify_dataset(result_dir)

    print(f"\nDataset ready at: {os.path.abspath(result_dir)}")
    print(f"Use for training:")
    print(f"  python train.py --train-images {result_dir}/train --val-images {result_dir}/valid")
    print(f"  # or")
    print(f"  flashdet train --train-images {result_dir}/train --val-images {result_dir}/valid")

    return result_dir
