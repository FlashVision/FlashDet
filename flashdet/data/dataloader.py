"""
DataLoader utilities for FlashDet detection.
"""

import random
import torch
from torch.utils.data import DataLoader
from typing import Optional, Tuple

from .dataset import FlashDetDataset, collate_fn
from .transforms import TrainTransform, ValTransform
from .augmentations import Mosaic, MixUp, CopyPaste


def create_dataloader(
    img_dir: str,
    ann_file: str,
    batch_size: int = 32,
    input_size: Tuple[int, int] = (320, 320),
    num_workers: int = 4,
    is_train: bool = True,
    shuffle: bool = None,
    mosaic: bool = False,
    mixup: bool = False,
    copy_paste: bool = False,
) -> DataLoader:
    """
    Create a DataLoader for FlashDet detection.

    Args:
        img_dir: Directory containing images.
        ann_file: Path to COCO annotation JSON.
        batch_size: Batch size.
        input_size: Input image size (width, height).
        num_workers: Number of data loading workers.
        is_train: Whether this is training data.
        shuffle: Whether to shuffle (defaults to ``is_train``).
        mosaic: Enable 4-image mosaic augmentation (train only).
        mixup: Enable MixUp augmentation (train only).
        copy_paste: Enable Copy-Paste augmentation (train only).

    Returns:
        DataLoader instance.
    """
    if shuffle is None:
        shuffle = is_train

    if is_train:
        transform = TrainTransform(input_size=input_size)
    else:
        transform = ValTransform(input_size=input_size)

    dataset = FlashDetDataset(
        img_dir=img_dir,
        ann_file=ann_file,
        transform=transform,
        input_size=input_size,
    )

    if is_train and (mosaic or mixup or copy_paste):
        _wrap_dataset_with_augmentations(dataset, input_size, mosaic, mixup, copy_paste)

    pin = torch.cuda.is_available()

    effective_workers = num_workers
    if num_workers > 0:
        try:
            import multiprocessing
            _lock = multiprocessing.Lock()
            del _lock
        except (PermissionError, OSError, RuntimeError) as e:
            import logging
            logging.getLogger(__name__).warning(
                "Multiprocessing unavailable (%s), falling back to num_workers=0", e
            )
            effective_workers = 0

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=effective_workers,
        pin_memory=pin and effective_workers > 0,
        collate_fn=collate_fn,
        drop_last=is_train,
    )

    return dataloader


def _wrap_dataset_with_augmentations(
    dataset: FlashDetDataset,
    input_size: Tuple[int, int],
    mosaic: bool,
    mixup: bool,
    copy_paste: bool,
) -> None:
    """Attach multi-image augmentations to the dataset's transform pipeline.

    The augmentations are applied *before* the regular TrainTransform so
    they operate on raw (un-normalised) images and xyxy boxes.
    """
    def _random_sample():
        idx = random.randint(0, len(dataset) - 1)
        return dataset.get_raw_item(idx)

    original_transform = dataset.transform

    augmenters = []
    if mosaic:
        augmenters.append(Mosaic(img_size=input_size, extra_images_fn=_random_sample))
    if mixup:
        augmenters.append(MixUp(extra_image_fn=_random_sample))
    if copy_paste:
        augmenters.append(CopyPaste(extra_image_fn=_random_sample))

    def _augmented_transform(image, boxes, labels):
        for aug in augmenters:
            if random.random() < 0.5:
                image, boxes, labels = aug(image, boxes, labels)
        return original_transform(image, boxes, labels)

    dataset.transform = _augmented_transform


def create_train_val_loaders(
    train_img_dir: str,
    train_ann_file: str,
    val_img_dir: str,
    val_ann_file: str,
    batch_size: int = 32,
    input_size: Tuple[int, int] = (320, 320),
    num_workers: int = 4,
    mosaic: bool = False,
    mixup: bool = False,
    copy_paste: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation DataLoaders."""
    train_loader = create_dataloader(
        img_dir=train_img_dir,
        ann_file=train_ann_file,
        batch_size=batch_size,
        input_size=input_size,
        num_workers=num_workers,
        is_train=True,
        mosaic=mosaic,
        mixup=mixup,
        copy_paste=copy_paste,
    )

    val_loader = create_dataloader(
        img_dir=val_img_dir,
        ann_file=val_ann_file,
        batch_size=batch_size,
        input_size=input_size,
        num_workers=num_workers,
        is_train=False,
    )

    return train_loader, val_loader
