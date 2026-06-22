"""
FlashDet Detection Dataset — generic COCO-format loader.
"""

import json
import logging
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Tuple, Callable, List


class FlashDetDataset(Dataset):
    """
    Generic COCO-format detection dataset for FlashDet.

    Args:
        img_dir: Directory containing images.
        ann_file: Path to COCO annotation JSON.
        transform: Transform function.
        input_size: Input image size (width, height).
    """
    
    def __init__(
        self,
        img_dir: str,
        ann_file: str,
        transform: Callable = None,
        input_size: Tuple[int, int] = (320, 320)
    ):
        self.img_dir = img_dir
        self.ann_file = ann_file
        self.transform = transform
        self.input_size = input_size
        
        # Load annotations
        with open(ann_file, "r") as f:
            self.coco = json.load(f)
        
        # Build image index
        self.images = {img["id"]: img for img in self.coco["images"]}
        
        # Build category mapping using SORTED category IDs (matches official COCO/FlashDet)
        cat_ids = sorted([cat["id"] for cat in self.coco.get("categories", [])])
        self.cat_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(cat_ids)}
        self.num_classes = len(cat_ids)
        
        # Group annotations by image, filtering invalid ones
        self.img_to_anns = {}
        skipped_anns = 0
        for ann in self.coco["annotations"]:
            img_id = ann["image_id"]
            cat_id = ann["category_id"]
            
            # Skip annotations with unknown category_id (matches official FlashDet)
            if cat_id not in self.cat_id_to_idx:
                skipped_anns += 1
                continue
            
            # Skip annotations with invalid bbox (area <= 0, w < 1, h < 1)
            x, y, w, h = ann["bbox"]
            if w < 1 or h < 1 or w * h <= 0:
                skipped_anns += 1
                continue
            
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
        
        # Image IDs - use sorted order for reproducibility (matches official)
        self.img_ids = sorted(list(self.images.keys()))

        total_anns = len(self.coco["annotations"])
        logging.getLogger(__name__).info(
            "Loaded %d images, %d annotations%s",
            len(self.img_ids), total_anns,
            f" ({skipped_anns} skipped)" if skipped_anns else ""
        )
    
    def __len__(self) -> int:
        return len(self.img_ids)
    
    def __getitem__(self, idx: int) -> Dict:
        img_id = self.img_ids[idx]
        img_info = self.images[img_id]
        
        # Load image
        img_path = os.path.join(self.img_dir, img_info["file_name"])
        image = cv2.imread(img_path)
        
        if image is None:
            fallback_idx = (idx + 1) % len(self.img_ids)
            if fallback_idx == idx:
                raise RuntimeError(f"Cannot read the only image in dataset: {img_path}")
            if not hasattr(self, "_retry_count"):
                self._retry_count = 0
            self._retry_count += 1
            if self._retry_count > min(10, len(self.img_ids)):
                self._retry_count = 0
                raise RuntimeError(
                    f"Too many consecutive unreadable images (started at {img_path}). "
                    "Check that img_dir points to a valid image directory."
                )
            logging.getLogger(__name__).warning(
                "Could not read %s, using sample %d", img_path, fallback_idx
            )
            result = self.__getitem__(fallback_idx)
            self._retry_count = 0
            return result
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]
        
        # Get annotations
        anns = self.img_to_anns.get(img_id, [])
        
        boxes = []
        labels = []
        
        for ann in anns:
            x, y, w, h = ann["bbox"]
            cat_id = ann["category_id"]
            
            # Skip unknown categories (already filtered in __init__, but double-check)
            if cat_id not in self.cat_id_to_idx:
                continue
            
            # Convert to xyxy format
            boxes.append([x, y, x + w, y + h])
            # Map category_id to 0-indexed label
            labels.append(self.cat_id_to_idx[cat_id])
        
        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
        labels = np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64)
        
        # Apply transforms
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        else:
            # Default transform
            image, boxes = self._default_transform(image, boxes)
        
        return {
            "image": image,
            "gt_bboxes": boxes,
            "gt_labels": labels,
            "img_id": img_id,
            "img_info": {
                "height": img_info["height"],
                "width": img_info["width"],
                "file_name": img_info["file_name"],
                "orig_h": orig_h,
                "orig_w": orig_w
            }
        }
    
    def get_raw_item(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return raw (image, boxes, labels) without transform — used by Mosaic/MixUp."""
        img_id = self.img_ids[idx]
        img_info = self.images[img_id]
        img_path = os.path.join(self.img_dir, img_info["file_name"])
        image = cv2.imread(img_path)
        if image is None:
            image = np.full((64, 64, 3), 114, dtype=np.uint8)
            return image, np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        anns = self.img_to_anns.get(img_id, [])
        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            cat_id = ann["category_id"]
            if cat_id not in self.cat_id_to_idx:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_id_to_idx[cat_id])
        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
        labels = np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64)
        return image, boxes, labels

    def _default_transform(self, image: np.ndarray, boxes: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        """Default image transform with box rescaling."""
        orig_h, orig_w = image.shape[:2]
        target_w, target_h = self.input_size
        
        # Scale factors
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h
        
        # Resize image
        image = cv2.resize(image, (target_w, target_h))
        
        # Scale boxes
        if len(boxes) > 0:
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
        
        # Normalize (RGB order - ImageNet stats)
        image = image.astype(np.float32)
        mean = np.array([123.675, 116.28, 103.53])  # RGB order
        std = np.array([58.395, 57.12, 57.375])      # RGB order
        image = (image - mean) / std
        
        # To tensor [C, H, W]
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        
        return image, boxes


PPEDataset = FlashDetDataset


def collate_fn(batch: List[Dict]) -> Tuple[torch.Tensor, Dict]:
    """
    Custom collate function for object detection.
    
    Returns:
        Tuple of (images, gt_meta) where gt_meta contains lists of boxes/labels.
    """
    images = torch.stack([item["image"] for item in batch])
    
    gt_meta = {
        "gt_bboxes": [item["gt_bboxes"] for item in batch],
        "gt_labels": [item["gt_labels"] for item in batch],
        "img_ids": [item["img_id"] for item in batch],
        "img_infos": [item["img_info"] for item in batch]
    }
    
    return images, gt_meta
