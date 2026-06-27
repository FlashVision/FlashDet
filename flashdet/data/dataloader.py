"""
DataLoader utilities for FlashDet detection.
"""

import random
import torch
from torch.utils.data import DataLoader, DistributedSampler
from typing import Tuple

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
    distributed: bool = False,
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

    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False  # sampler handles shuffling

    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        num_workers=effective_workers,
        pin_memory=pin and effective_workers > 0,
        collate_fn=collate_fn,
        drop_last=is_train,
        sampler=sampler,
    )
    if effective_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    dataloader = DataLoader(dataset, **loader_kwargs)

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

    All images (main + extras for mosaic/mixup/copy-paste) are letterbox-
    resized to ``input_size`` before compositing.  This keeps object scales
    consistent with the validation pipeline and avoids the train-val
    distribution gap that cripples mAP when mosaic operates on un-resized
    raw images.
    """
    import cv2
    from .transforms import _get_resize_matrix, _warp_boxes

    def _resize_to_input(img, boxes, target_size):
        h, w = img.shape[:2]
        tw, th = target_size
        M = _get_resize_matrix((w, h), (tw, th), keep_ratio=True)
        img_out = cv2.warpPerspective(
            img, M, dsize=(tw, th), borderValue=(114, 114, 114),
        )
        if len(boxes) > 0:
            boxes = _warp_boxes(boxes, M, tw, th)
        return img_out, boxes

    def _random_sample():
        idx = random.randint(0, len(dataset) - 1)
        img, boxes, labels = dataset.get_raw_item(idx)
        img, boxes = _resize_to_input(img, boxes, input_size)
        return img, boxes, labels

    original_transform = dataset.transform

    augmenters = []
    if mosaic:
        augmenters.append(Mosaic(img_size=input_size, extra_images_fn=_random_sample))
    if mixup:
        augmenters.append(MixUp(extra_image_fn=_random_sample))
    if copy_paste:
        augmenters.append(CopyPaste(extra_image_fn=_random_sample))

    # #region agent log
    _aug_counter = [0]
    # #endregion

    def _augmented_transform(image, boxes, labels):
        # #region agent log
        _aug_counter[0] += 1
        _pre_shape = image.shape[:2]
        _pre_nbox = len(boxes)
        # #endregion
        image, boxes = _resize_to_input(image, boxes, input_size)
        # #region agent log
        _post_resize_shape = image.shape[:2]
        _post_resize_nbox = len(boxes)
        _augs_applied = []
        # #endregion
        for aug in augmenters:
            if random.random() < 0.5:
                image, boxes, labels = aug(image, boxes, labels)
                # #region agent log
                _augs_applied.append(type(aug).__name__)
                # #endregion
        # #region agent log
        _pre_train_shape = image.shape[:2]
        _pre_train_nbox = len(boxes)
        if _aug_counter[0] <= 5:
            try:
                import json as _json, time as _time
                _dbg_log = "/home/ggoswami/Project/Gaurav/FlashVision/FlashDet/.cursor/debug-387c01.log"
                _box_range = {}
                if len(boxes) > 0:
                    import numpy as _np
                    _bx = _np.array(boxes) if not isinstance(boxes, _np.ndarray) else boxes
                    _box_range = {"box_x":[float(_bx[:,0].min()),float(_bx[:,2].max())],"box_y":[float(_bx[:,1].min()),float(_bx[:,3].max())],"box_w":[float((_bx[:,2]-_bx[:,0]).min()),float((_bx[:,2]-_bx[:,0]).max())],"box_h":[float((_bx[:,3]-_bx[:,1]).min()),float((_bx[:,3]-_bx[:,1]).max())]}
                with open(_dbg_log, "a") as _f:
                    _f.write(_json.dumps({"sessionId":"387c01","hypothesisId":"H2_transform","location":"dataloader.py:_augmented_transform","message":"augment_pipeline","data":{"sample":_aug_counter[0],"pre_shape":list(_pre_shape),"pre_nbox":_pre_nbox,"post_resize_shape":list(_post_resize_shape),"post_resize_nbox":_post_resize_nbox,"pre_train_shape":list(_pre_train_shape),"pre_train_nbox":_pre_train_nbox,"augs":_augs_applied,"input_size":list(input_size),**_box_range},"timestamp":int(_time.time()*1000)}) + "\n")
            except Exception:
                pass
        # #endregion
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
    distributed: bool = False,
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
        distributed=distributed,
    )

    val_loader = create_dataloader(
        img_dir=val_img_dir,
        ann_file=val_ann_file,
        batch_size=batch_size,
        input_size=input_size,
        num_workers=num_workers,
        is_train=False,
        distributed=distributed,
    )

    return train_loader, val_loader
