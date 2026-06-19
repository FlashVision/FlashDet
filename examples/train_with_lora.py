"""
LoRA Fine-Tuning Example
=========================

Fine-tune FlashDet using LoRA (Low-Rank Adaptation).
Only ~5% of parameters are trained — much faster and uses less memory.

Variants available: standard, dora, lora_plus, adalora, ortho, lora_fa
"""

from flashdet import Trainer

trainer = Trainer(
    model_size="m",
    train_images="data/train",
    val_images="data/val",
    epochs=50,
    batch_size=32,
    device="cuda",
    pretrained_coco=True,
    lora=True,
    lora_variant="dora",
    lora_rank=8,
    lora_alpha=16.0,
    save_dir="workspace/lora_model",
)

trainer.train()
