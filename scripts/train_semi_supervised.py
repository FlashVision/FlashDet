#!/usr/bin/env python3
"""
Semi-Supervised training for FlashDet.

Uses a teacher-student framework with EMA pseudo-labeling:
the teacher generates pseudo-labels on unlabeled data, and the
student trains on both labeled ground truth and pseudo-labels.

Usage:
    python scripts/train_semi_supervised.py \
        --train-images data/labeled/train \
        --val-images data/labeled/val \
        --unlabeled-images data/unlabeled/images \
        --epochs 100

    # Custom pseudo-label threshold
    python scripts/train_semi_supervised.py \
        --train-images data/train --val-images data/val \
        --unlabeled-images data/unlabeled \
        --pseudo-threshold 0.8 \
        --teacher-momentum 0.9995
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flashdet.engine import SemiSupervisedTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Semi-supervised training for FlashDet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    parser.add_argument("--train-images", required=True,
                        help="Path to labeled training images (COCO format)")
    parser.add_argument("--val-images", required=True,
                        help="Path to validation images (COCO format)")
    parser.add_argument("--unlabeled-images", required=True,
                        help="Path to unlabeled image directory")
    # Semi-supervised
    parser.add_argument("--pseudo-threshold", type=float, default=0.7,
                        help="Confidence threshold for pseudo-labels")
    parser.add_argument("--unsup-loss-weight", type=float, default=1.0,
                        help="Weight for unsupervised loss")
    parser.add_argument("--teacher-momentum", type=float, default=0.999,
                        help="EMA momentum for teacher model")
    parser.add_argument("--warmup-teacher-epochs", type=int, default=5,
                        help="Epochs of supervised-only before enabling pseudo-labels")
    parser.add_argument("--no-strong-aug", action="store_true",
                        help="Disable strong augmentation on unlabeled data")
    # Training
    parser.add_argument("--model-size", default="m", help="Model size (n, s, m, l)")
    parser.add_argument("--architecture", default="flashdet", help="Detector architecture")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--input-size", type=int, default=320, help="Input image size")
    parser.add_argument("--device", default="cuda", help="Device: cuda or cpu")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision")
    parser.add_argument("--save-dir", default="workspace/semi_sup_output", help="Output directory")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    return parser.parse_args()


def main():
    args = parse_args()

    trainer = SemiSupervisedTrainer(
        unlabeled_images=args.unlabeled_images,
        pseudo_label_threshold=args.pseudo_threshold,
        unsup_loss_weight=args.unsup_loss_weight,
        teacher_momentum=args.teacher_momentum,
        warmup_teacher_epochs=args.warmup_teacher_epochs,
        strong_aug=not args.no_strong_aug,
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
    )

    trainer.train()


if __name__ == "__main__":
    main()
