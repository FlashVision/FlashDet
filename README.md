<p align="center">
  <img src="assets/logo.png" width="200" alt="FlashDet Logo">
</p>

<h1 align="center">FlashDet</h1>

<p align="center">
  <a href="https://pypi.org/project/flashdet/"><img src="https://img.shields.io/pypi/v/flashdet?color=blue&logo=pypi&logoColor=white" alt="PyPI"></a>
  <a href="https://github.com/FlashVision/FlashDet/actions"><img src="https://img.shields.io/github/actions/workflow/status/FlashVision/FlashDet/ci.yml?logo=github" alt="CI"></a>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Python-3.8+-3776ab?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/ONNX-Export-005CED?logo=onnx&logoColor=white" alt="ONNX">
  <img src="https://img.shields.io/badge/LoRA-Fine_Tuning-ff6b6b" alt="LoRA">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

<p align="center">
  <b>Ultra-lightweight real-time object detection with LoRA fine-tuning, knowledge distillation, tracking, and analytics</b>
</p>

<p align="center">
  <a href="#installation">Install</a> •
  <a href="#usage">Usage</a> •
  <a href="#models">Models</a> •
  <a href="#examples">Examples</a> •
  <a href="#solutions">Solutions</a> •
  <a href="#trackers">Trackers</a> •
  <a href="#training">Training</a> •
  <a href="#contributing">Contributing</a>
</p>

---

## What is FlashDet?

FlashDet is an end-to-end object detection framework built for **speed and efficiency**. Based on the NanoDet-Plus architecture with a ShuffleNetV2 backbone, it delivers real-time inference with models as small as 0.49M parameters.

It ships as a `pip`-installable Python package with a CLI, a high-level Python API, and built-in solutions for counting, tracking, speed estimation, and more — similar to how you'd use Ultralytics YOLO.

```bash
pip install -e .
flashdet train --model-size m --epochs 100 --device cuda --train-images data/train --val-images data/val
flashdet predict --model best.pth --source video.mp4
```

---

## Installation

### pip (recommended)

```bash
pip install flashdet

# With all extras (tracking, analytics, ONNX export)
pip install "flashdet[all]"
```

### From source (for development)

```bash
git clone https://github.com/FlashVision/FlashDet.git
cd FlashDet
pip install -e ".[all]"
```

### Optional extras

```bash
pip install -e ".[export]"      # ONNX export support
pip install -e ".[tracker]"     # ByteTracker, SORT, BoTSORT
pip install -e ".[solutions]"   # Counting, speed, heatmaps
pip install -e ".[analytics]"   # Benchmarking, plots
pip install -e ".[all]"         # Everything
```

### Verify installation

```bash
flashdet check       # runs full health check
flashdet settings    # shows Python, PyTorch, CUDA, GPU info
flashdet version     # prints version
```

---

## Usage

### Python API

```python
from flashdet import FlashDet, Trainer, Predictor, Exporter

# Train a model
trainer = Trainer(
    model_size="m",
    train_images="data/train",
    val_images="data/val",
    epochs=100,
    device="cuda",
    use_lora=True,
    pretrained_coco=True,
)
trainer.train()

# Run inference
predictor = Predictor(model_path="workspace/best.pth", device="cuda")
results = predictor.predict("photo.jpg")

# Export to ONNX
exporter = Exporter(model_path="workspace/best.pth")
exporter.export(output="model.onnx", simplify=True)
```

### CLI

```bash
# Train
flashdet train --model-size m --epochs 100 --device cuda \
  --train-images data/train --val-images data/val --pretrained-coco

# Predict
flashdet predict --model best.pth --source image.jpg --conf 0.25

# Validate
flashdet val --model best.pth --val-images data/val

# Export
flashdet export --model best.pth --output model.onnx --simplify
```

### Standalone Scripts

```bash
# Full training with LoRA
python train.py --lora --lora-rank 8 --epochs 50 --device cuda

# Knowledge Distillation
python train_kd.py --teacher-checkpoint teacher.pth --teacher-size m-1.5x --model-size m-0.5x

# Inference
python test.py --model best.pth --image photo.jpg
```

---

## Models

| Model | Params | FP16 Size | INT8 Size | mAP (COCO) | GPU (ms) |
|-------|--------|-----------|-----------|------------|----------|
| **FlashDet-m-0.5x** | 0.49M | ~1.2 MB | ~0.6 MB | — | 3.8 |
| **FlashDet-m** | 1.17M | ~2.6 MB | ~1.3 MB | 27.0 | 5.3 |
| **FlashDet-m-1.5x** | 2.44M | ~5.2 MB | ~2.6 MB | 29.9 | 7.2 |

All models auto-download COCO pretrained weights on first use.

### Config-driven Training (Model Zoo)

Pick a config and train — no code changes needed:

```bash
flashdet train --config configs/flashdet_m_320_coco.yaml --device cuda
flashdet train --config configs/flashdet_m_320_lora.yaml --device cuda
flashdet train --config configs/flashdet_m_320_kd.yaml --device cuda
```

Available configs in [`configs/`](configs/):
| Config | Description |
|--------|-------------|
| `flashdet_m_320_coco.yaml` | Standard FlashDet-m training on COCO |
| `flashdet_m_416_coco.yaml` | FlashDet-m at higher resolution |
| `flashdet_m15x_320_coco.yaml` | Larger model for better accuracy |
| `flashdet_m05x_320_coco.yaml` | Ultra-light for edge deployment |
| `flashdet_m_320_lora.yaml` | LoRA fine-tuning on custom data |
| `flashdet_m_320_kd.yaml` | Knowledge distillation |

---

## Solutions

Built-in high-level applications for real-world use cases:

```python
from flashdet import Predictor
from flashdet.solutions import ObjectCounter, SpeedEstimator, Heatmap, ParkingManager
from flashdet.trackers import ByteTracker

predictor = Predictor(model_path="best.pth")
tracker = ByteTracker()

# Count objects crossing a line
counter = ObjectCounter(predictor, tracker, line_points=[(100, 300), (500, 300)])

# Estimate speeds
estimator = SpeedEstimator(predictor, tracker, pixels_per_meter=8.0)

# Generate heatmaps
heatmap = Heatmap(predictor, decay=0.95)

# Monitor parking
parking = ParkingManager(predictor, spots_file="spots.json")
```

| Solution | Description |
|----------|-------------|
| **ObjectCounter** | Count objects crossing lines or entering regions |
| **SpeedEstimator** | Estimate real-world speed from tracked objects |
| **Heatmap** | Visualize detection density over time |
| **RegionCounter** | Count objects in polygon zones |
| **QueueManager** | Monitor queue lengths and wait times |
| **DistanceCalculator** | Measure real-world distances between objects |
| **ParkingManager** | Track parking spot occupancy |
| **SecurityAlarm** | Alert on intrusions into restricted zones |

---

## Trackers

Multi-object tracking with persistent IDs across frames:

```python
from flashdet.trackers import ByteTracker, SORTTracker, BoTSORT

tracker = ByteTracker(max_age=30, min_hits=3, iou_threshold=0.3)
tracks = tracker.update(detections)  # [x1,y1,x2,y2,track_id,score,cls]
```

| Tracker | Method | Best For |
|---------|--------|----------|
| **ByteTracker** | IoU + Kalman filter | General purpose, fast |
| **SORTTracker** | Simple Kalman + Hungarian | Speed-critical applications |
| **BoTSORT** | Appearance + motion | Crowded scenes, re-identification |

---

## Training

### Standard Training

```bash
python train.py --model-size m --epochs 100 --batch-size 32 --device cuda --pretrained-coco
```

### LoRA / QLoRA Fine-Tuning

Parameter-efficient — freeze backbone, train only low-rank adapters:

```bash
# LoRA (6 variants: standard, dora, lora_plus, adalora, ortho, lora_fa)
python train.py --lora --lora-variant dora --lora-rank 8 --lora-alpha 16

# QLoRA (quantized base weights + LoRA)
python train.py --qlora --qlora-dtype nf4 --lora-rank 8
```

### Knowledge Distillation

Train a small student from a larger teacher:

```bash
python train_kd.py \
  --teacher-checkpoint workspace/teacher/best.pth \
  --teacher-size m-1.5x \
  --model-size m-0.5x \
  --kd-temperature 4.0
```

### Mixed Precision & Multi-GPU

```bash
python train.py --amp --multi-gpu --device cuda
```

---

## Analytics

```python
from flashdet.analytics import Benchmark, Profiler

# Benchmark model speed
bench = Benchmark(model_path="best.pth", device="cuda")
results = bench.run()  # {'fps': 142.3, 'latency_ms': 7.0, 'params': 1170000, ...}

# Profile layer-by-layer
profiler = Profiler(model_path="best.pth")
profiler.run()  # prints per-layer timing breakdown
```

---

## Examples

Ready-to-run scripts in the [`examples/`](examples/) folder:

| Script | What it does |
|--------|--------------|
| `train_custom_dataset.py` | Train on your own COCO-format dataset |
| `train_with_lora.py` | LoRA fine-tuning (DoRA variant) |
| `predict_image.py` | Detect objects in a single image |
| `track_objects.py` | Multi-object tracking on video |
| `count_objects.py` | Count objects crossing a line |
| `export_onnx.py` | Export to ONNX for deployment |
| `benchmark_model.py` | Measure FPS and latency |

```bash
cd examples
python train_custom_dataset.py
python predict_image.py
```

---

## Project Structure

```
FlashDet/
├── flashdet/                  # Main package (pip install -e .)
│   ├── __init__.py            # Public API
│   ├── cli.py                 # CLI entry point (flashdet command)
│   ├── registry.py            # Pluggable component registry
│   ├── cfg/                   # Configuration + YAML loading
│   ├── data/                  # Datasets, loaders, transforms
│   ├── engine/                # Trainer, Validator, Predictor, Exporter, Callbacks
│   ├── models/                # ShuffleNetV2, GhostPAN, NanoDet head, LoRA
│   ├── losses/                # QFL, GIoU, DFL, KD losses
│   ├── nn/                    # Neural network building blocks
│   ├── utils/                 # Metrics, visualization, checkpoint
│   ├── trackers/              # ByteTracker, SORT, BoTSORT
│   ├── solutions/             # Counting, speed, heatmaps, parking, security
│   └── analytics/             # Benchmark, profiling, plots
├── configs/                   # YAML configs for model zoo (pick & train)
├── examples/                  # Ready-to-run example scripts
├── tests/                     # Unit tests (pytest)
├── docker/                    # Dockerfile + docker-compose
├── train.py                   # Training script (LoRA, QLoRA, AMP, multi-GPU)
├── train_kd.py                # Knowledge Distillation script
├── test.py                    # Inference script
├── scripts/                   # ONNX export, quantization, data prep
├── pyproject.toml             # Package configuration
├── CONTRIBUTING.md            # How to contribute
├── CHANGELOG.md               # Version history
└── LICENSE                    # MIT
```

---

## Training Callbacks

Extend the training loop without modifying source code:

```python
from flashdet import Trainer
from flashdet.engine.callbacks import EarlyStopping, CSVLogger, TensorBoardCallback

trainer = Trainer(model_size="m", train_images="data/train", val_images="data/val")

trainer.add_callback(EarlyStopping(patience=20, metric="val_mAP"))
trainer.add_callback(CSVLogger("metrics.csv"))
trainer.add_callback(TensorBoardCallback("runs/exp1"))

trainer.train()
```

Built-in callbacks: `EarlyStopping`, `CSVLogger`, `TensorBoardCallback`, `LRSchedulerCallback`.

---

## Docker

```bash
# Build
docker build -t flashdet -f docker/Dockerfile .

# Run inference
docker run --gpus all -v $(pwd)/data:/app/data flashdet predict --model best.pth --source data/test.jpg

# Or use docker-compose
cd docker && docker compose up
```

---

## Supported Formats

| Import | Export |
|--------|--------|
| COCO JSON | ONNX |
| YOLO TXT | FP16 weights |
| Pascal VOC XML | TorchScript |

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
git clone https://github.com/FlashVision/FlashDet.git
cd FlashDet
pip install -e ".[dev,all]"
ruff check flashdet/
flashdet check
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <a href="https://github.com/FlashVision/FlashDet">
    <b>FlashVision</b>
  </a>
  — Open-source lightweight AI
</p>
