# Quick Start

## Train a model

```python
from flashdet import Trainer

trainer = Trainer(
    model_size="m",
    train_images="data/train",
    val_images="data/val",
    epochs=100,
    device="cuda",
)
trainer.train()
```

## Run inference

```python
from flashdet import Predictor

predictor = Predictor(model_path="workspace/best.pth", device="cuda")
results = predictor.predict("photo.jpg")
```

## Export to ONNX

```python
from flashdet import Exporter

exporter = Exporter(model_path="workspace/best.pth")
exporter.export(output="model.onnx", simplify=True)
```

## CLI

```bash
flashdet train --model-size m --epochs 100 --device cuda --train-images data/train --val-images data/val
flashdet predict --model best.pth --source image.jpg
flashdet export --model best.pth --output model.onnx
```
