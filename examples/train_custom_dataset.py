"""
Train FlashDet on a Custom Dataset
===================================

This example shows how to train FlashDet on your own COCO-format dataset.

Requirements:
    pip install flashdet

Dataset format (COCO-style):
    data/
    ├── train/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── _annotations.coco.json
    └── valid/
        ├── image1.jpg
        └── _annotations.coco.json

A small synthetic demo dataset can be generated at data/demo/ for quick testing:

    python scripts/create_demo_dataset.py
"""

from pathlib import Path

from flashdet import Trainer
from flashdet.data import create_demo_dataset

if __name__ == "__main__":
    if not Path("data/demo/train/_annotations.coco.json").exists():
        create_demo_dataset("data/demo")

    trainer = Trainer(
        model_size="n",
        train_images="data/demo/train",
        val_images="data/demo/valid",
        epochs=100,
        batch_size=32,
        device="cuda",
        save_dir="workspace/my_model",
    )

    trainer.train()
