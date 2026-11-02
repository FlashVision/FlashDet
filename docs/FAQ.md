# FAQ

## What is FlashDet?

FlashDet is a lightweight object detection framework with NMS-free inference, dual-head training, and pluggable components via a registry system.

## What datasets are supported?

COCO JSON, TXT labels, and Pascal VOC XML formats for import.

## Can I use my own backbone?

Yes, use the registry system:

```python
from flashdet.registry import BACKBONES

@BACKBONES.register("MyBackbone")
class MyBackbone(nn.Module):
    ...
```

## How to add a new architecture?

1. Create `flashdet/models/architectures/my_model.py`
2. Register it: `@DETECTORS.register("MyModel")`
3. Add to `flashdet/models/architectures/__init__.py`

## What training methods are available?

| Method | Class | Use case |
|--------|-------|----------|
| Standard | `Trainer` | Full supervision |
| Self-Supervised | `SSLTrainer` | Backbone pretraining |
| Semi-Supervised | `SemiSupervisedTrainer` | Limited labels |
| Few-Shot | `FewShotTrainer` | Very few examples |
| Active Learning | `ActiveLearningTrainer` | Annotation budget |

## Why does the loss level off near the end of training?

It happens in most training runs. Two things slow the final epochs down:

1. **The learning rate is small.** The cosine schedule is near its floor at
   epoch 90 of a 100-epoch run, so steps are tiny. This is expected — but you
   can control the floor with `--lrf`:

   ```bash
   # End at 2% of the base LR instead of the default 10% for more final polish
   python train.py --lrf 0.02 ...
   ```

2. **Weight decay on biases and norms.** Decaying normalization and bias
   parameters sets a floor on how low the loss can go. FlashDet now excludes
   them from decay automatically — this was previously applied to every
   parameter.

If the loss stops improving much earlier than the last few epochs (say by
epoch 30 of 100), the LR floor is not the issue — check the dataset size,
`--warmup-epochs`, and the loss weights instead.
