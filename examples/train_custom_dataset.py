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

A small demo dataset is included at data/demo/ for quick testing.
"""

from flashdet import Trainer

if __name__ == "__main__":
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
