#!/bin/bash
# Train a FlashDet-Pico model with any Pico-Zoo backbone/neck.
#
# Examples:
#   BACKBONE=repvit bash scripts/train_pico_zoo.sh
#   BACKBONE=edgenext NECK=attention bash scripts/train_pico_zoo.sh
#   BACKBONE=mambaout NECK=mamba NGPUS=4 bash scripts/train_pico_zoo.sh
#   BACKBONE=pp_lcnet NECK=lcnet INPUT_SIZE=416 EPOCHS=300 bash scripts/train_pico_zoo.sh
#
# Supported backbones: repvit, mobileone, fastvit, repghostnet, edgenext,
# mobilevitv2, efficientformerv2, shvit, efficientvit, swiftformer,
# efficientnetv2, emo, fasternet, starnet, mobilenetv4, ghostnetv2,
# mobilemamba, efficientvmamba, mambavision, mambaout, efficientvim,
# pp_lcnet, lcnet, esnet, pp_lcnetv2, peleenet, shufflenetv2,
# inceptionnext, unireplknet, convnextv2, efficientrep, rexnet,
# mobileformer, edgevit, tinyvit, levit, lowformer, localmamba,
# vmamba, plainmamba.
#
# Supported necks: pan, rep, ghost, attention, mamba, lcnet.
# If NECK is unset, the neck family matches the backbone family.

set -euo pipefail

BACKBONE=${BACKBONE:-repvit}
NECK=${NECK:-}
NGPUS=${NGPUS:-8}
MASTER_PORT=${MASTER_PORT:-29510}
INPUT_SIZE=${INPUT_SIZE:-320}
EPOCHS=${EPOCHS:-300}
BATCH_SIZE=${BATCH_SIZE:-16}
LR=${LR:-0.01}
SAVE_DIR=${SAVE_DIR:-"workspace/flashdet_pico_${BACKBONE}"}

NECK_ARGS=()
if [ -n "$NECK" ]; then
    NECK_ARGS=(--neck "$NECK")
fi

torchrun \
    --nproc_per_node="$NGPUS" \
    --master_port="$MASTER_PORT" \
    train.py \
    --model-size p \
    --backbone "$BACKBONE" \
    "${NECK_ARGS[@]}" \
    --input-size "$INPUT_SIZE" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
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
    "$@"
