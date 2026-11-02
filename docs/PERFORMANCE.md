# Training Performance & Stability

FlashDet keeps the default path safe and portable. Faster backend options are
opt-in.

## New flags

```bash
python train.py \
  --channels-last \
  --tf32 \
  --cudnn-benchmark ...
```

| Flag | Effect | Platform |
|---|---|---|
| `--channels-last` | NHWC memory format for CNN activations | CPU + CUDA |
| `--tf32` | TF32 matmul/cuDNN | CUDA (Ampere+) |
| `--cudnn-benchmark` | cuDNN autotuning for fixed shapes | CUDA |

`--channels-last` is measured at ~5% faster for Pico at 128×128 on CPU in a short
benchmark (0.354 s → 0.335 s for 5 steps, batch 8). Gains depend on model/hardware.

## Automatic optimizations

- AdamW uses fused CUDA kernels when available, otherwise `foreach`.
- SGD uses `foreach` when supported.
- Data loading uses pinned memory + non-blocking host-to-device copies.
- `zero_grad(set_to_none=True)` avoids unnecessary zeroing.
- Validation runs the full validation set on rank 0 (not a strided DDP shard).
- Early stopping is broadcast to all DDP ranks.
- EMA decay ramps exponentially toward the target instead of stalling.
- ProgLoss uses the same 0-based epoch in training and validation.
- Validation loss is per-sample, not batch-scaled.

## Recommended recipes

### CPU

```bash
python train.py --channels-last --workers 8 ...
```

### CUDA

```bash
python train.py --amp --channels-last --tf32 --cudnn-benchmark \
  --multi-gpu ...
```

### Large batches

```bash
python train.py --amp --grad-accum 2 --lr 0.002 ...
```

`--optimizer-in-bwd` is automatically disabled when AMP or gradient
accumulation is active because eager optimizer-in-backward is incompatible with
those paths.

## Stability notes

- Use `--warmup-epochs 2` or more for larger models / small datasets.
- Keep the default gradient clipping (`grad_clip_norm`) unless you have a reason.
- Do not use `--channels-last` with custom layers that assume contiguous NCHW.
- TF32 changes numerics slightly; validate accuracy after enabling it.
