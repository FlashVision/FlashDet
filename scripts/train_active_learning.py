#!/usr/bin/env python3
"""
Active Learning training for FlashDet.

Iteratively trains a model and queries the most informative unlabeled
samples for annotation, maximizing accuracy while minimizing labeling cost.

Usage:
    # Entropy-based querying, 5 rounds, 50 images per round
    python scripts/train_active_learning.py \
        --train-images data/labeled/train \
        --val-images data/labeled/val \
        --unlabeled-pool data/unlabeled/images \
        --query-strategy entropy \
        --query-budget 50 \
        --al-rounds 5

    # MC-Dropout uncertainty with 20 forward passes
    python scripts/train_active_learning.py \
        --train-images data/train --val-images data/val \
        --unlabeled-pool data/unlabeled \
        --query-strategy mc_dropout \
        --mc-dropout-passes 20
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flashdet.engine import ActiveLearningTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Active learning training for FlashDet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Active learning specific
    parser.add_argument("--unlabeled-pool", required=True,
                        help="Path to unlabeled image pool")
    parser.add_argument("--query-strategy", default="entropy",
                        choices=["entropy", "margin", "least_confidence", "random", "mc_dropout"],
                        help="Uncertainty sampling strategy")
    parser.add_argument("--query-budget", type=int, default=50,
                        help="Number of images to query per round")
    parser.add_argument("--al-rounds", type=int, default=5,
                        help="Number of active learning rounds")
    parser.add_argument("--mc-dropout-passes", type=int, default=10,
                        help="Number of MC-dropout forward passes (mc_dropout strategy)")
    # Data
    parser.add_argument("--train-images", required=True,
                        help="Path to initial labeled training images (COCO format)")
    parser.add_argument("--val-images", required=True,
                        help="Path to validation images (COCO format)")
    # Training
    parser.add_argument("--model-size", default="m", help="Model size (n, s, m, l)")
    parser.add_argument("--architecture", default="flashdet", help="Detector architecture")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs per AL round")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--input-size", type=int, default=320, help="Input image size")
    parser.add_argument("--device", default="cuda", help="Device: cuda or cpu")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision")
    parser.add_argument("--save-dir", default="workspace/active_learning_output",
                        help="Output directory")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    return parser.parse_args()


def main():
    args = parse_args()

    trainer = ActiveLearningTrainer(
        unlabeled_pool=args.unlabeled_pool,
        query_strategy=args.query_strategy,
        query_budget=args.query_budget,
        al_rounds=args.al_rounds,
        mc_dropout_T=args.mc_dropout_passes,
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
