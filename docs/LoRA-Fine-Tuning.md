# LoRA Fine-Tuning

## Overview

LoRA (Low-Rank Adaptation) freezes the backbone and trains only small low-rank adapters, reducing memory and training time significantly.

## Variants

| Variant | Description |
|---------|-------------|
| standard | Classic LoRA |
| dora | Weight-decomposed LoRA |
| lora_plus | Differentiated learning rates |
| adalora | Adaptive rank allocation |
| ortho | Orthogonal regularization |
| lora_fa | Frozen-A LoRA |

## Usage

```bash
python train.py --lora --lora-variant dora --lora-rank 8 --lora-alpha 16
```

## QLoRA

Quantized LoRA reduces memory further by quantizing frozen weights:

```bash
python train.py --qlora --qlora-dtype nf4 --lora-rank 8
```
