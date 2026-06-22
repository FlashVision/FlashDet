#!/usr/bin/env bash
#
# Train all FlashDet model sizes (N, S, M) on COCO 2017
# and benchmark inference time after training completes.
#
# Usage:
#   bash scripts/train_all_flashdet.sh
#
# Expects COCO data at: data/coco2017/train  and  data/coco2017/valid
#
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

DATA_TRAIN="data/coco2017/train"
DATA_VAL="data/coco2017/valid"

EPOCHS=300
BATCH_N=64
BATCH_S=48
BATCH_M=32
INPUT_SIZE=640
DEVICE="cuda"
WORKERS=8

echo "============================================================"
echo "  FlashDet Full COCO Training — N / S / M"
echo "============================================================"
echo "  Dataset: COCO 2017 (118K train, 5K val, 80 classes)"
echo "  Epochs:  $EPOCHS"
echo "  Input:   ${INPUT_SIZE}x${INPUT_SIZE}"
echo "  Device:  $DEVICE"
echo "============================================================"
echo ""

# ---------- FlashDet-N ----------
echo "[1/3] Training FlashDet-N (width=0.25, depth=0.33, ~1.5M params)"
python train.py \
  --model-size n \
  --architecture flashdet \
  --input-size "$INPUT_SIZE" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_N" \
  --lr 0.01 \
  --optimizer musgd \
  --warmup-epochs 5 \
  --patience 50 \
  --amp \
  --mosaic --mixup \
  --workers "$WORKERS" \
  --device "$DEVICE" \
  --train-images "$DATA_TRAIN" \
  --val-images "$DATA_VAL" \
  --save-dir "workspace/flashdet_n_coco"

echo ""
echo "[1/3] FlashDet-N training complete."
echo ""

# ---------- FlashDet-S ----------
echo "[2/3] Training FlashDet-S (width=0.50, depth=0.33, ~5.4M params)"
python train.py \
  --model-size s \
  --architecture flashdet \
  --input-size "$INPUT_SIZE" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_S" \
  --lr 0.01 \
  --optimizer musgd \
  --warmup-epochs 5 \
  --patience 50 \
  --amp \
  --mosaic --mixup \
  --workers "$WORKERS" \
  --device "$DEVICE" \
  --train-images "$DATA_TRAIN" \
  --val-images "$DATA_VAL" \
  --save-dir "workspace/flashdet_s_coco"

echo ""
echo "[2/3] FlashDet-S training complete."
echo ""

# ---------- FlashDet-M ----------
echo "[3/3] Training FlashDet-M (width=1.00, depth=0.67, ~18M params)"
python train.py \
  --model-size m \
  --architecture flashdet \
  --input-size "$INPUT_SIZE" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_M" \
  --lr 0.01 \
  --optimizer musgd \
  --warmup-epochs 5 \
  --patience 50 \
  --amp \
  --mosaic --mixup \
  --workers "$WORKERS" \
  --device "$DEVICE" \
  --train-images "$DATA_TRAIN" \
  --val-images "$DATA_VAL" \
  --save-dir "workspace/flashdet_m_coco"

echo ""
echo "[3/3] FlashDet-M training complete."
echo ""

echo "============================================================"
echo "  All 3 FlashDet models trained!"
echo "  Weights saved to:"
echo "    workspace/flashdet_n_coco/"
echo "    workspace/flashdet_s_coco/"
echo "    workspace/flashdet_m_coco/"
echo "============================================================"
