#!/usr/bin/env python3
"""
Self-Supervised Learning (SSL) pretraining for FlashDet.

Pretrain a backbone using BYOL, MoCo, or SimCLR on unlabeled images,
then use the pretrained weights to initialize a detector for fine-tuning.

Usage:
    # BYOL pretraining (default)
    python scripts/train_ssl.py \
        --train-images path/to/unlabeled/images \
        --epochs 100

    # MoCo pretraining with custom backbone
    python scripts/train_ssl.py \
        --ssl-method moco \
        --backbone shufflenetv2 \
        --train-images path/to/unlabeled/images

    # Then fine-tune on labeled data
    python train.py \
        --finetune workspace/ssl_output/backbone_pretrained.pth \
        --train-images data/train --val-images data/val
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flashdet.engine import SSLTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="SSL pretraining for FlashDet backbones",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-images", required=True,
                        help="Path to directory of unlabeled training images")
    parser.add_argument("--ssl-method", default="byol", choices=["byol", "moco", "simclr"],
                        help="SSL method")
    parser.add_argument("--backbone", default="shufflenetv2",
                        help="Backbone architecture to pretrain")
    parser.add_argument("--backbone-size", default="1.0x",
                        help="Backbone size variant")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of pretraining epochs")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=0.03,
                        help="Learning rate")
    parser.add_argument("--input-size", type=int, default=224,
                        help="Input image size")
    parser.add_argument("--proj-dim", type=int, default=256,
                        help="Projection head output dimension")
    parser.add_argument("--momentum", type=float, default=0.996,
                        help="EMA momentum for target network")
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="InfoNCE temperature (MoCo/SimCLR)")
    parser.add_argument("--workers", type=int, default=4,
                        help="DataLoader workers")
    parser.add_argument("--save-dir", default="workspace/ssl_output",
                        help="Output directory")
    parser.add_argument("--device", default="cuda",
                        help="Device: cuda or cpu")
    parser.add_argument("--amp", action="store_true",
                        help="Enable mixed precision")
    return parser.parse_args()


def main():
    args = parse_args()

    trainer = SSLTrainer(
        ssl_method=args.ssl_method,
        backbone_name=args.backbone,
        backbone_size=args.backbone_size,
        proj_dim=args.proj_dim,
        temperature=args.temperature,
        momentum=args.momentum,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        workers=args.workers,
        save_dir=args.save_dir,
        device=args.device,
        amp=args.amp,
        train_images=args.train_images,
        input_size=args.input_size,
    )

    backbone_path = trainer.pretrain()
    print(f"\nPretrained backbone saved to: {backbone_path}")
    print(f"Fine-tune with: python train.py --finetune {backbone_path} --train-images <data>")


if __name__ == "__main__":
    main()
