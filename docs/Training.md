# Training

## Standard Training

```bash
flashdet train --model-size m --epochs 100 --batch-size 32 --device cuda --pretrained-coco
```

Or via Python API:

```python
from flashdet import Trainer

trainer = Trainer(
    model_size="m",
    epochs=100,
    batch_size=32,
    device="cuda",
    pretrained_coco=True,
    train_images="data/train",
    val_images="data/val",
)
trainer.train()
```

## Config-driven Training

```bash
flashdet train --config configs/flashdet_m_320_coco.yaml
```

## Training Options

| Flag | Description | Default |
|------|-------------|---------|
| `--model-size` | Model variant (m, m-0.5x, m-1.5x) | m |
| `--epochs` | Training epochs | 100 |
| `--batch-size` | Batch size | 32 |
| `--lr` | Learning rate | 0.001 |
| `--device` | Device (cuda/cpu) | cuda |
| `--amp` | Mixed precision training | false |
| `--multi-gpu` | DataParallel | false |
| `--pretrained-coco` | Load COCO weights | false |
| `--warmup-epochs` | LR warmup epochs | 5 |
| `--patience` | Early stopping patience | 50 |
| `--grad-accum` | Gradient accumulation steps | 1 |

## LoRA / QLoRA Fine-Tuning

Parameter-efficient fine-tuning — freeze backbone, train only low-rank adapters:

```python
trainer = Trainer(
    model_size="m",
    lora=True,
    lora_variant="dora",    # standard, dora, lora_plus, adalora, ortho, lora_fa
    lora_rank=8,
    lora_alpha=16.0,
    lora_targets=["backbone", "fpn"],
    pretrained_coco=True,
)
```

Or via CLI:

```bash
python train.py --lora --lora-variant dora --lora-rank 8 --lora-alpha 16
python train.py --qlora --qlora-dtype nf4 --lora-rank 8
```

## Knowledge Distillation

Train a smaller student model from a larger teacher:

```bash
python train_kd.py \
  --teacher-checkpoint workspace/teacher/best.pth \
  --teacher-size m-1.5x \
  --model-size m-0.5x \
  --kd-temperature 4.0 \
  --kd-logit-weight 1.0 \
  --kd-feature-weight 0.5
```

KD combines:
- **Logit KD**: KL-divergence between teacher/student class distributions
- **Feature KD**: L2 alignment of FPN feature maps via 1x1 conv adapters

## Training Callbacks

Extend the training loop without modifying source code:

```python
from flashdet import Trainer
from flashdet.engine.callbacks import EarlyStopping, CSVLogger, TensorBoardCallback

trainer = Trainer(model_size="m", train_images="data/train", val_images="data/val")

trainer.add_callback(EarlyStopping(patience=20, metric="val_mAP"))
trainer.add_callback(CSVLogger("metrics.csv"))
trainer.add_callback(TensorBoardCallback("runs/exp1"))

trainer.train()
```

Built-in callbacks:
- `EarlyStopping` — Stop when metric plateaus
- `CSVLogger` — Log metrics to CSV
- `TensorBoardCallback` — Log to TensorBoard
- `LRSchedulerCallback` — Step LR scheduler per epoch

## Model EMA

Exponential Moving Average of model weights is used automatically during training for better generalization. The EMA model is used for validation and saved as the inference model.

## Auxiliary Head (AGM)

The Assign Guidance Module provides additional supervision during training (not used at inference). It uses a deep-copy of the FPN with detached backbone features to guide the label assignment in the main head.

## Data Format

FlashDet supports COCO JSON annotation format:

```
data/
├── train/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── _annotations.coco.json
└── val/
    ├── image1.jpg
    ├── image2.jpg
    └── _annotations.coco.json
```

## Performance Options

| Option | Description |
|--------|-------------|
| `--amp` | FP16 mixed precision (2x memory savings) |
| `--multi-gpu` | DataParallel across GPUs |
| `--grad-accum N` | Simulate larger batch sizes |
| `--activation-checkpointing` | Trade compute for memory |
| `--compile` | torch.compile for faster training |
| `--chunked-loss` | Memory-efficient loss computation |
