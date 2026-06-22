# FAQ

## What is FlashDet?

FlashDet is a YOLO26-based lightweight object detection framework supporting multiple architectures (FlashDet, DETR, RT-DETR, YOLOv9, YOLOv10, YOLOv11, GroundingDINO) with pluggable components via a registry system.

## What datasets are supported?

COCO JSON, YOLO TXT, and Pascal VOC XML formats for import.

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

## How to export for mobile?

```bash
flashdet export --model best.pth --output model.onnx --simplify
```

Then convert ONNX to TFLite, CoreML, or NCNN as needed.

## What training methods are available?

| Method | Class | Use case |
|--------|-------|----------|
| Standard | `Trainer` | Full supervision |
| Knowledge Distillation | `KDTrainer` | Model compression |
| Self-Supervised | `SSLTrainer` | Backbone pretraining |
| Semi-Supervised | `SemiSupervisedTrainer` | Limited labels |
| Few-Shot | `FewShotTrainer` | Very few examples |
| Active Learning | `ActiveLearningTrainer` | Annotation budget |
