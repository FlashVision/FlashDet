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
