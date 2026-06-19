# FAQ

## How fast is FlashDet?

FlashDet-m achieves 100+ FPS on GPU and 30+ FPS on edge devices with only 1.17M parameters.

## What datasets are supported?

COCO JSON, YOLO TXT, and Pascal VOC XML formats for import.

## Can I use my own backbone?

Yes, use the registry system to register custom backbones:

```python
from flashdet.registry import BACKBONES

@BACKBONES.register("my_backbone")
class MyBackbone(nn.Module):
    ...
```

## How to export for mobile?

```bash
flashdet export --model best.pth --output model.onnx --simplify
```

Then convert ONNX to TFLite, CoreML, or NCNN as needed.

## What's the difference between LoRA variants?

- **standard**: Classic low-rank adapters
- **dora**: Better generalization via weight decomposition
- **adalora**: Automatically adjusts rank per layer
- **lora_plus**: Different LR for A and B matrices
