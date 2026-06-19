"""
DataLoader utilities for PPE detection.
"""

import torch
from torch.utils.data import DataLoader
from typing import Tuple

from .dataset import PPEDataset, collate_fn
from .transforms import TrainTransform, ValTransform


def create_dataloader(
    img_dir: str,
    ann_file: str,
    batch_size: int = 32,
    input_size: Tuple[int, int] = (320, 320),
    num_workers: int = 4,
    is_train: bool = True,
    shuffle: bool = None
) -> DataLoader:
    """
    Create a DataLoader for PPE detection.
    
    Args:
        img_dir: Directory containing images
        ann_file: Path to COCO annotation JSON
        batch_size: Batch size
        input_size: Input image size (width, height)
        num_workers: Number of data loading workers
        is_train: Whether this is training data
        shuffle: Whether to shuffle data (defaults to is_train)
        
    Returns:
        DataLoader instance
    """
    if shuffle is None:
        shuffle = is_train
    
    # Create transform
    if is_train:
        transform = TrainTransform(input_size=input_size)
    else:
        transform = ValTransform(input_size=input_size)
    
    # Create dataset
    dataset = PPEDataset(
        img_dir=img_dir,
        ann_file=ann_file,
        transform=transform,
        input_size=input_size
    )
    
    # Only pin memory when CUDA is actually available
    pin = torch.cuda.is_available()

    # Probe whether multiprocessing works before creating the DataLoader.
    # Some environments (containers, sandboxes) deny /dev/shm access.
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


def create_train_val_loaders(
    train_img_dir: str,
    train_ann_file: str,
    val_img_dir: str,
    val_ann_file: str,
    batch_size: int = 32,
    input_size: Tuple[int, int] = (320, 320),
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation DataLoaders.
    
    Args:
        train_img_dir: Training images directory
        train_ann_file: Training annotations file
        val_img_dir: Validation images directory
        val_ann_file: Validation annotations file
        batch_size: Batch size
        input_size: Input image size
        num_workers: Number of workers
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_loader = create_dataloader(
        img_dir=train_img_dir,
        ann_file=train_ann_file,
        batch_size=batch_size,
        input_size=input_size,
        num_workers=num_workers,
        is_train=True
    )
    
    val_loader = create_dataloader(
        img_dir=val_img_dir,
        ann_file=val_ann_file,
        batch_size=batch_size,
        input_size=input_size,
        num_workers=num_workers,
        is_train=False
    )
    
    return train_loader, val_loader
