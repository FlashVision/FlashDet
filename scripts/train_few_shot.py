#!/usr/bin/env python3
"""
Few-Shot training for FlashDet.

Fine-tunes a pretrained detection model on very few examples per class
(1-30 shots). Freezes the backbone and optionally the neck, training
only the detection head for fast adaptation.

Usage:
    # 5-shot fine-tuning from a base checkpoint
    python scripts/train_few_shot.py \
        --base-checkpoint workspace/base/model_best_inference.pth \
        --train-images data/novel/train \
        --val-images data/novel/val \
        --n-shot 5

    # 10-shot with unfrozen neck
    python scripts/train_few_shot.py \
        --base-checkpoint path/to/base.pth \
        --train-images data/novel/train \
        --val-images data/novel/val \
        --n-shot 10 --no-freeze-neck
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flashdet.engine import FewShotTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Few-shot training for FlashDet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Few-shot specific
    parser.add_argument("--base-checkpoint", required=True,
                        help="Path to pretrained base model checkpoint")
    parser.add_argument("--n-shot", type=int, default=5,
                        help="Number of examples per class")
    parser.add_argument("--no-freeze-backbone", action="store_true",
                        help="Don't freeze backbone (train all layers)")
    parser.add_argument("--no-freeze-neck", action="store_true",
                        help="Don't freeze neck/FPN")
    parser.add_argument("--head-lr-factor", type=float, default=10.0,
                        help="LR multiplier for detection head vs backbone")
    # Data
    parser.add_argument("--train-images", required=True,
                        help="Path to novel-class training images (COCO format)")
    parser.add_argument("--val-images", required=True,
                        help="Path to validation images (COCO format)")
    # Training
    parser.add_argument("--model-size", default="m", help="Model size (n, s, m, l)")
    parser.add_argument("--architecture", default="flashdet", help="Detector architecture")
    parser.add_argument("--epochs", type=int, default=30, help="Fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Base learning rate")
    parser.add_argument("--input-size", type=int, default=320, help="Input image size")
    parser.add_argument("--device", default="cuda", help="Device: cuda or cpu")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision")
    parser.add_argument("--save-dir", default="workspace/few_shot_output", help="Output directory")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    return parser.parse_args()


def main():
    args = parse_args()

    trainer = FewShotTrainer(
        base_checkpoint=args.base_checkpoint,
        n_shot=args.n_shot,
        freeze_backbone=not args.no_freeze_backbone,
        freeze_neck=not args.no_freeze_neck,
        head_lr_factor=args.head_lr_factor,
        # Trainer base args
        train_images=args.train_images,
        val_images=args.val_images,
        model_size=args.model_size,
        architecture=args.architecture,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        input_size=args.input_size,
        device=args.device,
        amp=args.amp,
        save_dir=args.save_dir,
        workers=args.workers,
        patience=args.patience,
    )

    trainer.train()


if __name__ == "__main__":
    main()
