#!/bin/bash
# FlashDet-Pico RepNeXt v2 — Improved training recipe
#
# Key improvements over v1:
#   1. Mosaic + MixUp augmentations (biggest mAP gain)
#   2. 300 epochs (from 100) — small models need more training
#   3. Higher LR 0.01 (from 0.002) — better for SGD + cosine schedule
#   4. 10-epoch warmup (from 5)
#   5. Finetune from v1 best checkpoint (warm start)
#
# Expected: 0.21 → 0.35+ mAP@0.5 on COCO val2017

set -euo pipefail

NGPUS=${NGPUS:-8}
MASTER_PORT=${MASTER_PORT:-29510}
SAVE_DIR="workspace/flashdet_pico_repnext_v2"

# Use v1 best weights as warm start (skip if not found)
FINETUNE_FLAG=""
if [ -f "workspace/flashdet_pico_repnext_v1/model_best_inference.pth" ]; then
    FINETUNE_FLAG="--finetune workspace/flashdet_pico_repnext_v1/model_best_inference.pth"
    echo "Warm-starting from v1 best checkpoint"
fi

torchrun \
    --nproc_per_node=$NGPUS \
    --master_port=$MASTER_PORT \
    train.py \
    --model-size p \
    --backbone pico_v2 \
    --input-size 416 \
    --epochs 300 \
    --batch-size 16 \
    --lr 0.01 \
    --warmup-epochs 10 \
    --optimizer sgd \
    --weight-decay 0.05 \
    --amp \
    --multi-gpu \
    --mosaic \
    --mixup \
    --train-images data/coco2017/train \
    --val-images data/coco2017/valid \
    --save-dir "$SAVE_DIR" \
    --workers 4 \
    --patience 100 \
    --val-interval 5 \
    $FINETUNE_FLAG \
    "$@"
