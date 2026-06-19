"""
Train FlashDet on a Custom Dataset
===================================

This example shows how to train FlashDet-m on your own COCO-format dataset
with pretrained weights and LoRA fine-tuning.

Requirements:
    pip install flashdet

Dataset format:
    data/
    ├── train/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── _annotations.coco.json
    └── val/
        ├── image1.jpg
        └── _annotations.coco.json
"""

from flashdet import Trainer

trainer = Trainer(
    model_size="m",
    train_images="data/train",
    val_images="data/val",
    epochs=100,
    batch_size=32,
    device="cuda",
    pretrained_coco=True,
    save_dir="workspace/my_model",
)

print("Starting training...")
trainer.train()
print("Training complete! Model saved to workspace/my_model/")
