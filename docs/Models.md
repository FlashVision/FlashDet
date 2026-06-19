# Models

## Available Variants

| Model | Params | FP16 Size | mAP (COCO) | GPU Latency |
|-------|--------|-----------|------------|-------------|
| FlashDet-m-0.5x | 0.49M | ~1.2 MB | — | 3.8 ms |
| FlashDet-m | 1.17M | ~2.6 MB | 27.0 | 5.3 ms |
| FlashDet-m-1.5x | 2.44M | ~5.2 MB | 29.9 | 7.2 ms |

## Architecture

- **Backbone**: ShuffleNetV2 (width multiplier configurable)
- **Neck**: GhostPAN (lightweight feature pyramid)
- **Head**: NanoDet-Plus head with GFL (Generalized Focal Loss)

## Pretrained Weights

COCO pretrained weights are auto-downloaded on first use:

```python
trainer = Trainer(model_size="m", pretrained_coco=True)
```
