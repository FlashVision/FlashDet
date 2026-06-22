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
  <b>YOLO26-based real-time object detection with multi-architecture support, advanced training methods, LoRA fine-tuning, tracking, and analytics</b>
</p>

<p align="center">
  <a href="#installation">Install</a> •
  <a href="#architectures">Architectures</a> •
  <a href="#usage">Usage</a> •
  <a href="#training-methods">Training Methods</a> •
  <a href="#solutions">Solutions</a> •
  <a href="#trackers">Trackers</a> •
  <a href="#project-structure">Structure</a> •
  <a href="#contributing">Contributing</a>
</p>

---

## What is FlashDet?

FlashDet is an end-to-end object detection framework built for **speed, accuracy, and extensibility**. The core FlashDet model is built on **YOLO26** principles with a dual detection head (NMS-free one-to-one + dense one-to-many), **STAL** (Small-Target-Aware Label Assignment), **ProgLoss** (Progressive Loss Balancing), and the **MuSGD** (Muon+SGD hybrid) optimizer.

Beyond the FlashDet architecture, the framework supports **7 detector architectures** and **6 training methods** — all through a unified, registry-based, pluggable design.

```
Training Pipeline:
  Dataset → Augmentation → YOLO26 Model
    ├── Classification Loss (BCE)
    ├── Box Loss (CIoU + L1, ProgLoss weighted)
    └── STAL Assignment
        → MuSGD → Updated Weights
```
---

## Architectures

FlashDet ships with 7 detector architectures, all accessible via a unified API:

| Architecture | Description | Key Features |
|---|---|---|
| **FlashDet** | YOLO26-based lightweight detector | Dual-head, STAL, ProgLoss, DFL-free |
| **DETR** | DEtection TRansformer | End-to-end, no NMS, Hungarian matching |
| **RT-DETR** | Real-Time DETR | HybridEncoder, efficient transformer |
| **YOLOv9** | PGI + GELAN architecture | Programmable Gradient Information |
| **YOLOv10** | NMS-free real-time YOLO | Dual assignment, efficiency-accuracy |
| **YOLOv11** | Latest YOLO with C3k2 + C2PSA | PSA attention, state-of-the-art speed |
| **GroundingDINO** | Open-vocabulary detector | Text-conditioned detection |

### FlashDet Model Sizes

| Model | Backbone | Params (Inference) | FP16 Size | Notes |
|---|---|---|---|---|
| **FlashDet-P** (Pico) | ShuffleNetV2-0.5x + GhostPAN | ~298K | **0.57 MB** | Sub-1MB, depthwise heads |
| **FlashDet-N** (Nano) | YOLOv11 (w=0.25, d=0.33) | ~1.06M | 2.01 MB | Lightweight |
| **FlashDet-S** (Small) | YOLOv11 (w=0.50, d=0.33) | ~5.4M | 10.3 MB | Balanced |
| **FlashDet-M** (Medium) | YOLOv11 (w=1.00, d=0.67) | ~18M | 34.3 MB | High accuracy |

**FlashDet-P (Pico)** is designed for extreme edge deployment (microcontrollers, mobile, browser). It uses:
- **ShuffleNetV2-0.5x** backbone with ImageNet pretraining (channel shuffle + depthwise convolutions)
- **GhostPAN** neck with 64-ch output (Ghost modules for cheap feature generation)
- **Depthwise-separable E2E dual head** (DW-conv + pointwise instead of full convolutions)
- Same STAL + ProgLoss training recipe as larger variants
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
from flashdet.models.detector import build_model
from flashdet.cfg import get_config

# Build any architecture
config = get_config(num_classes=80)
model = build_model(config, architecture="flashdet")  # or "detr", "yolov11", etc.

# Build sub-1MB Pico model for edge deployment
pico = FlashDet(num_classes=80, size="p")
print(pico.get_model_info())  # inference_fp16_mb: 0.57

# Train
trainer = Trainer(
    model_size="p",   # "p" (Pico), "n", "s", "m", "l", "x"
    train_images="data/train",
    val_images="data/val",
    epochs=100,
    device="cuda",
)
trainer.train()

# Inference
predictor = Predictor(model_path="workspace/best.pth", device="cuda")
results = predictor.predict("photo.jpg")

# Export to ONNX
exporter = Exporter(model_path="workspace/best.pth")
exporter.export(output="model.onnx", simplify=True)
```

### CLI

```bash
# Train (use --model-size p for Pico, n for Nano, s for Small, etc.)
flashdet train --model-size p --epochs 100 --device cuda \
  --train-images data/train --val-images data/val

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

# Inference
python test.py --model best.pth --image photo.jpg
```

---

## Training Methods

FlashDet supports 6 training paradigms, each with a dedicated trainer class and CLI script:

| Method | Trainer Class | CLI Script | Description |
|---|---|---|---|
| **Standard** | `Trainer` | `train.py` | Full supervised training with all augmentations |
| **Knowledge Distillation** | `KDTrainer` | `scripts/train_kd.py` | Teach a small student from a larger teacher |
| **Self-Supervised (SSL)** | `SSLTrainer` | `scripts/train_ssl.py` | BYOL pretraining on unlabeled data |
| **Semi-Supervised** | `SemiSupervisedTrainer` | `scripts/train_semi_supervised.py` | Teacher-student with pseudo-labels |
| **Few-Shot** | `FewShotTrainer` | `scripts/train_few_shot.py` | Learn from very few labeled examples |
| **Active Learning** | `ActiveLearningTrainer` | `scripts/train_active_learning.py` | Intelligently select samples for labeling |

### Knowledge Distillation

```bash
python scripts/train_kd.py \
  --teacher-checkpoint path/to/teacher.pth \
  --teacher-size n \
  --model-size n \
  --kd-temperature 4.0
```

### Self-Supervised Pretraining

```bash
python scripts/train_ssl.py \
  --method byol \
  --data-dir path/to/unlabeled/images \
  --epochs 100 --backbone-size n
```

### Semi-Supervised Learning

```bash
python scripts/train_semi_supervised.py \
  --train-images data/train \
  --unlabeled-dir path/to/unlabeled/images \
  --pseudo-threshold 0.7
```

### Few-Shot Learning

```bash
python scripts/train_few_shot.py \
  --base-checkpoint path/to/base.pth \
  --n-shot 10 --freeze-backbone
```

### Active Learning

```bash
python scripts/train_active_learning.py \
  --train-images data/train \
  --unlabeled-pool path/to/unlabeled/images \
  --query-strategy entropy --budget 50 --rounds 5
```

### LoRA / QLoRA Fine-Tuning

Parameter-efficient — freeze backbone, train only low-rank adapters:

```bash
# LoRA (6 variants: standard, dora, lora_plus, adalora, ortho, lora_fa)
python train.py --lora --lora-variant dora --lora-rank 8 --lora-alpha 16

# QLoRA (quantized base weights + LoRA)
python train.py --qlora --qlora-dtype nf4 --lora-rank 8
```

### Mixed Precision & Multi-GPU

```bash
python train.py --amp --multi-gpu --device cuda
```

---

## Core Components

### STAL (Small-Target-Aware Label Assignment)

Task-Aligned Assignment with small-target protection — temporarily expands tiny GT boxes during candidate selection so small objects always get positive anchor supervision.

### ProgLoss (Progressive Loss Balancing)

Linearly shifts training emphasis from the dense one-to-many head (exploration) to the NMS-free one-to-one head (refinement) over the course of training: `alpha(t): 1.0 → 0.0`.

### MuSGD (Muon + SGD Hybrid Optimizer)

Applies Muon-style orthogonal updates to multi-dimensional parameters (conv weights, attention) while using standard SGD for 1D parameters (biases, norms), combining faster convergence with training stability.

### E2E Detection Loss

Combines CIoU box loss, BCE classification loss, and L1 regression loss across both dual heads, weighted by the ProgLoss schedule.

---

## Solutions

Built-in high-level applications for real-world use cases:

```python
from flashdet import Predictor
from flashdet.solutions import ObjectCounter, SpeedEstimator, Heatmap
from flashdet.trackers import ByteTracker

predictor = Predictor(model_path="best.pth")
tracker = ByteTracker()

counter = ObjectCounter(predictor, tracker, line_points=[(100, 300), (500, 300)])
estimator = SpeedEstimator(predictor, tracker, pixels_per_meter=8.0)
heatmap = Heatmap(predictor, decay=0.95)
```

| Solution | Description |
|---|---|
| **ObjectCounter** | Count objects crossing lines or entering regions |
| **SpeedEstimator** | Estimate real-world speed from tracked objects |
| **Heatmap** | Visualize detection density over time |
| **RegionCounter** | Count objects in polygon zones |
| **QueueManager** | Monitor queue lengths and wait times |
| **DistanceCalculator** | Measure real-world distances between objects |
| **ParkingManager** | Track parking spot occupancy |
| **SecurityAlarm** | Alert on intrusions into restricted zones |
| **WorkoutMonitor** | Track exercise repetitions and form |
| **LiveInference** | Real-time webcam/stream detection |
| **AnalyticsDashboard** | Aggregated detection statistics and visualization |
---

## Trackers

Multi-object tracking with persistent IDs across frames:

```python
from flashdet.trackers import ByteTracker, SORTTracker, BoTSORT

tracker = ByteTracker(max_age=30, min_hits=3, iou_threshold=0.3)
tracks = tracker.update(detections)  # [x1,y1,x2,y2,track_id,score,cls]
```

| Tracker | Method | Best For |
|---|---|---|
| **ByteTracker** | IoU + Kalman filter | General purpose, fast |
| **SORTTracker** | Simple Kalman + Hungarian | Speed-critical applications |
| **BoTSORT** | Appearance + motion | Crowded scenes, re-identification |

---

## Analytics

```python
from flashdet.analytics import Benchmark, Profiler

bench = Benchmark(model_path="best.pth", device="cuda")
results = bench.run()  # {'fps': ..., 'latency_ms': ..., 'params': ..., ...}

profiler = Profiler(model_path="best.pth")
profiler.run()  # prints per-layer timing breakdown
```

---

## Training Callbacks

Extend the training loop without modifying source code:

```python
from flashdet import Trainer
from flashdet.engine.core.callbacks import EarlyStopping, CSVLogger, TensorBoardCallback

trainer = Trainer(model_size="n", train_images="data/train", val_images="data/val")

trainer.add_callback(EarlyStopping(patience=20, metric="val_mAP"))
trainer.add_callback(CSVLogger("metrics.csv"))
trainer.add_callback(TensorBoardCallback("runs/exp1"))

trainer.train()
```

Built-in callbacks: `EarlyStopping`, `CSVLogger`, `TensorBoardCallback`, `LRSchedulerCallback`.

---

## Registry System

FlashDet uses a pluggable registry for all major components. Adding a new architecture, backbone, head, or loss is as simple as decorating your class:

```python
from flashdet.registry import DETECTORS, BACKBONES, HEADS

@DETECTORS.register("MyDetector")
class MyDetector(nn.Module):
    ...

# Later, build from config
model = DETECTORS.build("MyDetector", num_classes=80)
```

Available registries: `DETECTORS`, `BACKBONES`, `NECKS`, `HEADS`, `LOSSES`, `DATASETS`, `TRANSFORMS`, `TRACKERS`.
---

## Examples

Ready-to-run scripts in [`examples/`](examples/):

| Script | What it does |
|---|---|
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
├── flashdet/                        # Main package
│   ├── __init__.py                  # Public API
│   ├── cli.py                       # CLI entry point (flashdet command)
│   ├── registry.py                  # Pluggable component registry
│   ├── cfg/                         # Configuration + YAML loading
│   │   └── config.py
│   ├── data/                        # Datasets, loaders, transforms, download
│   │   ├── dataset.py
│   │   ├── dataloader.py
│   │   ├── augmentations.py
│   │   ├── transforms.py
│   │   ├── download.py
│   │   └── prepare.py
│   ├── engine/                      # Training, evaluation, inference, export
│   │   ├── core/                    # Callbacks, EMA, MuSGD optimizer
│   │   │   ├── callbacks.py
│   │   │   ├── ema.py
│   │   │   └── musgd.py
│   │   ├── training/                # All training paradigms
│   │   │   ├── trainer.py           # Standard Trainer
│   │   │   ├── kd_trainer.py        # Knowledge Distillation
│   │   │   ├── ssl_trainer.py       # Self-Supervised Learning
│   │   │   ├── semi_supervised_trainer.py
│   │   │   ├── few_shot_trainer.py
│   │   │   └── active_learning_trainer.py
│   │   ├── evaluation/              # Validator
│   │   ├── inference/               # Predictor, postprocessing
│   │   └── export/                  # ONNX exporter
│   ├── models/                      # Model components
│   │   ├── architectures/           # Full detector architectures
│   │   │   ├── flashdet.py          # YOLO26-based FlashDet
│   │   │   ├── detr.py
│   │   │   ├── rt_detr.py
│   │   │   ├── yolov9.py
│   │   │   ├── yolov10.py
│   │   │   ├── yolov11.py
│   │   │   └── grounding_dino.py
│   │   ├── backbone/                # ShuffleNetV2, ResNet, YOLOv9-11
│   │   ├── neck/                    # GhostPAN, HybridEncoder, YOLO necks
│   │   ├── head/                    # Detection heads (NanoDet, DETR, E2E, OBB)
│   │   ├── layers/                  # ConvBNSiLU, C2f, C3k2, RepVGG, PSA, SPPF
│   │   ├── transformer/             # DETR transformer, positional encoding
│   │   ├── assignment/              # STAL, DSL, Hungarian matcher
│   │   ├── detector.py              # build_model() factory
│   │   └── lora.py                  # LoRA / QLoRA (6 variants)
│   ├── losses/                      # Loss functions
│   │   ├── e2e_loss.py              # E2E dual-head loss + ProgLoss
│   │   ├── focal_loss.py            # QFL, DFL
│   │   ├── iou_loss.py              # GIoU, CIoU
│   │   ├── kd_loss.py               # Knowledge distillation losses
│   │   ├── detr_loss.py
│   │   ├── rt_detr_loss.py
│   │   ├── yolo_loss.py
│   │   └── varifocal_loss.py
│   ├── utils/                       # Metrics, visualization, checkpoints
│   ├── trackers/                    # ByteTracker, SORT, BoTSORT
│   ├── solutions/                   # 11 ready-to-use vision solutions
│   ├── analytics/                   # Benchmark, profiling, plots
│   └── nn/                          # Additional neural network blocks
├── scripts/                         # Specialized training & utility scripts
│   ├── train_kd.py                  # Knowledge Distillation CLI
│   ├── train_ssl.py                 # SSL pretraining CLI
│   ├── train_semi_supervised.py     # Semi-Supervised CLI
│   ├── train_few_shot.py            # Few-Shot CLI
│   ├── train_active_learning.py     # Active Learning CLI
│   ├── convert_pth_to_onnx.py
│   ├── fp16_to_int8_quantize.py
│   └── prepare_data.py
├── configs/                         # YAML configs for model zoo
├── examples/                        # Ready-to-run example scripts
├── tests/                           # Unit & integration tests (pytest)
├── docs/                            # Documentation (Training, Models, FAQ, etc.)
├── docker/                          # Dockerfile + docker-compose
├── assets/                          # Diagrams and images
├── train.py                         # Main training entry point
├── test.py                          # Main inference entry point
├── pyproject.toml                   # Package configuration
└── LICENSE                          # MIT
```

---

## Docker

```bash
# Build
docker build -t flashdet -f docker/Dockerfile .

# Run inference
docker run --gpus all -v $(pwd)/data:/app/data flashdet \
  predict --model best.pth --source data/test.jpg

# Or use docker-compose
cd docker && docker compose up
```

---

## Supported Formats

| Import | Export |
|---|---|
| COCO JSON | ONNX |
| YOLO TXT | FP16 weights |
| Pascal VOC XML | TorchScript |

---

## Documentation

Full documentation is in the [`docs/`](docs/) folder:

| Document | Description |
|---|---|
| [Home](docs/Home.md) | Overview and getting started |
| [Installation](docs/Installation.md) | Detailed installation guide |
| [Quick-Start](docs/Quick-Start.md) | Quick examples for training and inference |
| [Training](docs/Training.md) | All training methods and hyperparameters |
| [Models](docs/Models.md) | Architecture details and model zoo |
| [LoRA Fine-Tuning](docs/LoRA-Fine-Tuning.md) | LoRA/QLoRA variants and usage |
| [Solutions](docs/Solutions.md) | Vision solutions reference |
| [Trackers](docs/Trackers.md) | Multi-object tracking guide |
| [FAQ](docs/FAQ.md) | Frequently asked questions |
| [Contributing](docs/CONTRIBUTING.md) | How to contribute |
| [Changelog](docs/CHANGELOG.md) | Version history |

---

## Contributing

We welcome contributions! See [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines on adding new architectures, training methods, loss functions, layers, and solutions.

```bash
git clone https://github.com/FlashVision/FlashDet.git
cd FlashDet
pip install -e ".[dev,all]"
pytest tests/
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
