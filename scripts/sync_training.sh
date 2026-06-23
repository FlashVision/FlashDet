#!/bin/bash
#
# Auto-sync Pico training outputs from GPU server to local PC.
# Simply rsyncs every 30s — always gets the latest files.
# No complex epoch detection needed.
#
# Usage:  bash scripts/sync_training.sh
# Stop:   Ctrl+C
#

SERVER="${SYNC_SERVER:?Set SYNC_SERVER e.g. user@host}"
REMOTE_DIR="${SYNC_REMOTE_DIR:-~/workspace/flashdet_pico_coco_v4}"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)/workspace/flashdet_pico_coco_v4"

mkdir -p "$LOCAL_DIR"

echo "============================================"
echo "  FlashDet Pico — Training Sync Monitor"
echo "============================================"
echo "  Server:  $SERVER"
echo "  Remote:  $REMOTE_DIR"
echo "  Local:   $LOCAL_DIR"
echo "  Syncing every 30 seconds"
echo "============================================"
echo ""

while true; do
    echo "[$(date '+%H:%M:%S')] Syncing..."

    rsync -az \
        --include='*.log' \
        --include='*.txt' \
        --include='*.jpg' \
        --include='*.png' \
        --include='*.json' \
        --include='*.csv' \
        --include='plots/***' \
        --include='visualizations/***' \
        --include='checkpoint_best.pth' \
        --include='checkpoint_last.pth' \
        --include='model_best_inference.pth' \
        --include='model_best_fp16.pth' \
        --exclude='*.pth' \
        "$SERVER:$REMOTE_DIR/" "$LOCAL_DIR/" 2>/dev/null

    if [ $? -eq 0 ] || [ $? -eq 24 ]; then
        # Show latest metrics
        LOGFILE=$(ls -t "$LOCAL_DIR"/train_*.log 2>/dev/null | head -1)
        if [ -n "$LOGFILE" ]; then
            LATEST_METRIC=$(grep -E 'Validation.*mAP|Saved best' "$LOGFILE" 2>/dev/null | tail -2)
            LATEST_BATCH=$(grep 'Batch' "$LOGFILE" 2>/dev/null | tail -1)
            echo "[$(date '+%H:%M:%S')] $LATEST_BATCH"
            if [ -n "$LATEST_METRIC" ]; then
                echo "$LATEST_METRIC"
            fi
        fi
    else
        echo "[$(date '+%H:%M:%S')] Sync failed (rsync exit $?)"
    fi

    sleep 30
done
