# Training

## Standard Training

```bash
python train.py --model-size m --epochs 100 --batch-size 32 --device cuda
```

Or via Python API:

```python
from flashdet import Trainer

trainer = Trainer(
    model_size="m",
    epochs=100,
    batch_size=32,
    device="cuda",
    train_images="data/train",
    val_images="data/val",
)
trainer.train()
```

## Config-driven Training

```bash
flashdet train --config configs/flashdet_m_320_coco.yaml
```

## Model Sizes

| Size | Description |
|------|-------------|
| `n`  | Nano — smallest, fastest |
| `s`  | Small |
| `m`  | Medium (default) |
| `l`  | Large — highest accuracy |

## Multi-Architecture Training

FlashDet supports multiple detector architectures via the `--architecture` flag:

```bash
python train.py --architecture yolov9 --epochs 100
python train.py --architecture detr --epochs 100
python train.py --architecture rt-detr --epochs 100
```

Available: `flashdet` (default), `detr`, `rt-detr`, `yolov9`, `yolov10`, `yolov11`, `grounding-dino`

## Training Methods

### Knowledge Distillation

Train a smaller student model from a larger teacher:

```python
from flashdet.engine import KDTrainer

trainer = KDTrainer(
    teacher_checkpoint="workspace/teacher/best.pth",
    teacher_size="l",
    model_size="n",
    kd_temperature=4.0,
    kd_logit_weight=1.0,
    kd_feature_weight=0.5,
    train_images="data/train",
    val_images="data/val",
)
trainer.train()
```

Or via CLI:

```bash
python scripts/train_kd.py \
  --teacher-checkpoint workspace/teacher/best.pth \
  --teacher-size l --model-size n
```

### Self-Supervised Learning (SSL)

Pretrain the backbone with BYOL before supervised detection training:

```python
from flashdet.engine import SSLTrainer

ssl = SSLTrainer(
    ssl_method="byol",
    epochs=50,
    train_images="data/unlabeled/",
    save_dir="workspace/ssl",
    device="cuda",
)
backbone_path = ssl.pretrain()
```

### Semi-Supervised Learning

Use labeled + unlabeled data with teacher-student EMA pseudo-labeling:

```python
from flashdet.engine import SemiSupervisedTrainer

trainer = SemiSupervisedTrainer(
    unlabeled_images="data/unlabeled/",
    ema_momentum=0.999,
    train_images="data/train",
    val_images="data/val",
)
trainer.train()
```

### Few-Shot Learning

Fine-tune from a base checkpoint with frozen backbone:

```python
from flashdet.engine import FewShotTrainer

trainer = FewShotTrainer(
    k_shot=5,
    base_checkpoint="workspace/base/best.pth",
    freeze_backbone=True,
    train_images="data/few_shot/",
    val_images="data/val/",
)
trainer.train()
```

### Active Learning

Iteratively query the most informative samples:

```python
from flashdet.engine import ActiveLearningTrainer

trainer = ActiveLearningTrainer(
    query_strategy="entropy",
    query_budget=50,
    al_rounds=5,
    train_images="data/train",
    val_images="data/val",
)
trainer.train()
```

### MuSGD Optimizer

Hybrid Muon-SGD optimizer for faster convergence:

```bash
python train.py --optimizer musgd --lr 0.02
```

## Training Options

| Flag | Description | Default |
|------|-------------|---------|
| `--model-size` | Model size (n, s, m, l) | m |
| `--architecture` | Detector architecture | flashdet |
| `--epochs` | Training epochs | 100 |
| `--batch-size` | Batch size | 32 |
| `--lr` | Learning rate | 0.001 |
| `--device` | Device (cuda/cpu) | cuda |
| `--amp` | Mixed precision training | false |
| `--multi-gpu` | DataParallel | false |
| `--warmup-epochs` | LR warmup epochs | 5 |
| `--patience` | Early stopping patience | 50 |
| `--grad-accum` | Gradient accumulation steps | 1 |
| `--optimizer` | Optimizer (adam, sgd, musgd) | adam |

## LoRA / QLoRA Fine-Tuning

Parameter-efficient fine-tuning — freeze backbone, train only low-rank adapters:

```python
trainer = Trainer(
    model_size="m",
    lora=True,
    lora_variant="dora",
    lora_rank=8,
    lora_alpha=16.0,
    lora_targets=["backbone", "fpn"],
)
```

See [LoRA Fine-Tuning](LoRA-Fine-Tuning.md) for details.

## Training Callbacks

```python
from flashdet.engine import Trainer, EarlyStopping, CSVLogger, TensorBoardCallback

trainer = Trainer(model_size="m", train_images="data/train", val_images="data/val")
trainer.add_callback(EarlyStopping(patience=20, metric="val_mAP"))
trainer.add_callback(CSVLogger("metrics.csv"))
trainer.add_callback(TensorBoardCallback("runs/exp1"))
trainer.train()
```

## Data Format

FlashDet uses COCO JSON annotation format:

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
