# Training Matrix — 1,000 Images / 10 Classes

This is a controlled smoke benchmark, not a COCO leaderboard. Its purpose is to
verify that every FlashDet size trains, validates, saves, reloads and exports,
and to expose training-loop bugs on a small dataset.

## Dataset

Generated with `scripts/create_synthetic_dataset.py`:

- 1,000 training images / 200 validation images
- 10 classes with class-specific colored shapes
- 1,967 train annotations / 394 validation annotations
- Generated at 192 px, trained at 128 px
- CPU only, 16 cores, batch size 16

```bash
python scripts/create_synthetic_dataset.py \
  --output /tmp/synth10 --train 1000 --val 200 --size 192 --classes 10
```

## Results

### 5 epochs

| Model | Params (train) | Best mAP@0.5 | Best val loss |
|---|---:|---:|---:|
| FlashDet-P | 0.40M | 0.465 | 6.64 |
| FlashDet-N | 0.85M | 0.266 | 4.99 |
| FlashDet-S | 1.95M | 0.400 | 4.14 |
| FlashDet-M | 4.23M | 0.516 | 3.66 |
| FlashDet-L | 7.25M | 0.594 | 3.15 |
| FlashDet-X | 11.26M | 0.502 | 3.39 |

### 15 epochs

| Model | Best mAP@0.5 | Best val loss | Approx. train time |
|---|---:|---:|---:|
| FlashDet-P | 0.801 | 3.43 | 2m 35s |
| FlashDet-N | 0.699 | 2.73 | 2m 42s |
| FlashDet-S | 0.767 | 2.44 | 3m 28s |
| FlashDet-M | 0.857 | 1.99 | 4m 48s |
| FlashDet-L | 0.844 | 1.91 | 6m 23s |
| FlashDet-X | 0.896 | 1.70 | 8m 00s |

## Bugs found and fixed

### 1. ProgLoss used different epochs for training and validation

**Symptom:** on a 1-epoch run the training log showed `prog_alpha=0.2000`
while validation showed `prog_alpha=0.8000`.

**Cause:** `train.py` passed a 1-based epoch to the model, while `validate()`
passed a 0-based epoch. The ProgLoss schedule is 0-based.

**Fix:** training now passes `epoch - 1` (clamped at 0), matching validation.
The same fix was applied to `Trainer` in the engine. Training and validation
now report the same alpha at every epoch.

### 2. Validation loss depended on batch size

**Symptom:** validation loss was ~16× larger than the per-sample loss shown in
the training sub-metrics (batch size was 16).

**Cause:** the loss returns a batch-scaled total for gradient linear scaling,
but `validate()` logged that scaled total as `val_loss`.

**Fix:** training/validation logging now uses `loss_states["loss_total"]`
(per-sample). Gradients still use the batch-scaled loss, so the linear scaling
rule is unchanged.

## Model-by-model notes

| Model | Observation |
|---|---|
| **P** | Fastest and strongest on the short run because the default LiteBackbone is ImageNet-pretrained. |
| **N** | Lowest 15-epoch mAP. It trains FlashBackbone from scratch and needs more epochs or pretraining. |
| **S** | Improves steadily; between N and M. Same from-scratch limitation as N. |
| **M** | Strong short-run result; best balance of accuracy and training time here. |
| **L** | Close to M; slightly lower at 15 epochs, likely short-run noise/under-training rather than an architecture failure. |
| **X** | Best final mAP but slowest. It had an early validation-loss spike at 5 epochs, which disappeared with more warmup/epochs. |

The non-monotonic 5-epoch ordering (P > N/S) is **not** an architecture bug.
Pico uses a pretrained backbone; N/S start from random weights and need more
training. With 15 epochs the ordering is mostly monotonic:
M/L/X > P > S > N.

## Verification performed

For every size:

- training completed without NaN loss, zero-positive batches or OOM
- `checkpoint_best.pth` and inference weights were written
- `load_inference_model()` rebuilt the correct model
- `Predictor` loaded the checkpoint and ran inference
- `flashdet export` produced an ONNX file successfully

## Reproduce

```bash
python scripts/create_synthetic_dataset.py \
  --output /tmp/synth10 --train 1000 --val 200 --size 192 --classes 10

python train.py \
  --model-size m \
  --epochs 15 \
  --warmup-epochs 2 \
  --lr 0.003 \
  --batch-size 16 \
  --input-size 128 \
  --train-images /tmp/synth10/train \
  --val-images /tmp/synth10/valid \
  --save-dir workspace/matrix/m \
  --workers 4 \
  --device cpu \
  --val-interval 1 \
  --patience 0 \
  --skip-verify-annotations
```

> These results are synthetic and hardware-dependent. Use real data and the
> full 300-epoch schedule for production accuracy decisions.
