"""
LoRA Fine-Tuning Example
=========================

Fine-tune FlashDet using LoRA (Low-Rank Adaptation).
Only ~5% of parameters are trained — much faster and uses less memory.

Variants available: standard, dora, lora_plus, adalora, ortho, lora_fa
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
        epochs=50,
        batch_size=32,
        device="cuda",
        # finetune="path/to/checkpoint.pth",  # optional warm start
        lora=True,
        lora_variant="dora",
        lora_rank=8,
        lora_alpha=16.0,
        save_dir="workspace/lora_model",
    )

    trainer.train()
