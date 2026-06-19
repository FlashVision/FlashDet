# Training

## Standard Training

```bash
flashdet train --model-size m --epochs 100 --batch-size 32 --device cuda --pretrained-coco
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
| `--lr` | Learning rate | 0.01 |
| `--device` | Device (cuda/cpu) | cuda |
| `--amp` | Mixed precision training | false |
| `--multi-gpu` | DataParallel | false |
| `--pretrained-coco` | Load COCO weights | false |

## Data Format

FlashDet supports COCO JSON annotation format:

```
data/
├── train/
│   ├── images/
│   └── annotations.json
└── val/
    ├── images/
    └── annotations.json
```
