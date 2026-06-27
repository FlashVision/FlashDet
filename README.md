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
  <b>Ultra-lightweight real-time object detection with advanced training methods, LoRA fine-tuning, tracking, and analytics</b>
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

FlashDet is an end-to-end object detection framework built for **speed, accuracy, and extensibility**. The core FlashDet model features a dual detection head (NMS-free one-to-one + dense one-to-many), **STAL** (Small-Target-Aware Label Assignment), **ProgLoss** (Progressive Loss Balancing), and the **MuSGD** (Muon+SGD hybrid) optimizer.

The framework supports **6 training methods** — all through a unified, registry-based, pluggable design.

```
Training Pipeline:
  Dataset → Augmentation → FlashDet Model
    ├── Classification Loss (BCE)
    ├── Box Loss (CIoU + L1, ProgLoss weighted)
    └── STAL Assignment
        → MuSGD → Updated Weights
```
---

## Model Sizes

| Model | Backbone | Params (Inference) | FP16 Size | Notes |
|---|---|---|---|---|
| **FlashDet-P** (Pico) | LiteBackbone-0.5x / PicoBackbone + PicoNeck | ~298K | **0.57 MB** | Sub-1MB, depthwise heads |
| **FlashDet-N** (Nano) | FlashBackbone (w=0.25, d=0.33) | ~1.06M | 2.01 MB | Lightweight |
| **FlashDet-S** (Small) | FlashBackbone (w=0.50, d=0.33) | ~5.4M | 10.3 MB | Balanced |
| **FlashDet-M** (Medium) | FlashBackbone (w=1.00, d=0.67) | ~18M | 34.3 MB | High accuracy |

**FlashDet-P (Pico)** is designed for extreme edge deployment (microcontrollers, mobile, browser). It uses:
- **LiteBackbone-0.5x** with pretrained weights (channel mixing + depthwise convolutions)
- **PicoNeck** with 64-ch output (lightweight modules for efficient feature generation)
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
pip install -e ".[tracker]"     # FlashTracker, MotionTracker, AppearanceTracker
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
from flashdet import FlashDet, Trainer

# Build sub-1MB Pico model for edge deployment
pico = FlashDet(num_classes=80, size="p")
print(pico.get_model_info())  # inference_fp16_mb: 0.57

# Build with reparameterizable backbone
pico_v2 = FlashDet(num_classes=80, size="p", backbone_type="repnext")

# Build larger model
model_n = FlashDet(num_classes=80, size="n")

# Train
trainer = Trainer(
    model_size="p",   # "p" (Pico), "n", "s", "m", "l", "x"
    train_images="data/train",
    val_images="data/val",
    epochs=100,
    device="cuda",
)
trainer.train()
```

### CLI

```bash
# Train (use --model-size p for Pico, n for Nano, s for Small, etc.)
flashdet train --model-size p --epochs 100 --device cuda \
  --train-images data/train --val-images data/val

# Validate
flashdet val --model best.pth --val-images data/val
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

FlashDet supports 5 training paradigms, each with a dedicated trainer class and CLI script:

| Method | Trainer Class | CLI Script | Description |
|---|---|---|---|
| **Standard** | `Trainer` | `train.py` | Full supervised training with all augmentations |
| **Self-Supervised (SSL)** | `SSLTrainer` | `scripts/train_ssl.py` | BYOL pretraining on unlabeled data |
| **Semi-Supervised** | `SemiSupervisedTrainer` | `scripts/train_semi_supervised.py` | Teacher-student with pseudo-labels |
| **Few-Shot** | `FewShotTrainer` | `scripts/train_few_shot.py` | Learn from very few labeled examples |
| **Active Learning** | `ActiveLearningTrainer` | `scripts/train_active_learning.py` | Intelligently select samples for labeling |

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
from flashdet.solutions import ObjectCounter, SpeedEstimator, Heatmap
from flashdet.trackers import FlashTracker

tracker = FlashTracker()
# Solutions integrate with any detection model for real-world applications
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
from flashdet.trackers import FlashTracker, MotionTracker, AppearanceTracker

tracker = FlashTracker(max_age=30, min_hits=3, iou_threshold=0.3)
tracks = tracker.update(detections)  # [x1,y1,x2,y2,track_id,score,cls]
```

| Tracker | Method | Best For |
|---|---|---|
| **FlashTracker** | IoU + Kalman filter | General purpose, fast |
| **MotionTracker** | Kalman + Hungarian matching | Speed-critical applications |
| **AppearanceTracker** | Appearance + motion fusion | Crowded scenes, re-identification |

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

```bash
cd examples
python train_custom_dataset.py
```

---

## Project Structure

```
FlashDet/
├── flashdet/                        # Main package
│   ├── __init__.py                  # Public API
│   ├── cli.py                       # CLI entry point
│   ├── registry.py                  # Pluggable component registry
│   ├── cfg/                         # Configuration
│   ├── data/                        # Datasets, loaders, transforms, download
│   ├── engine/
│   │   ├── core/                    # Callbacks, EMA, MuSGD optimizer
│   │   ├── training/                # All training paradigms
│   │   │   ├── trainer.py           # Standard Trainer
│   │   │   ├── kd_trainer.py        # Knowledge Distillation
│   │   │   ├── ssl_trainer.py       # Self-Supervised Learning
│   │   │   ├── semi_supervised_trainer.py
│   │   │   ├── few_shot_trainer.py
│   │   │   └── active_learning_trainer.py
│   │   └── evaluation/              # Validator
│   ├── models/
│   │   ├── architectures/
│   │   │   └── flashdet.py          # FlashDet + FlashDetPico
│   │   ├── backbone/                # LiteBackbone, PicoBackbone, FlashBackbone
│   │   ├── neck/                    # PicoNeck, YOLO necks
│   │   ├── head/                    # E2E dual detection head
│   │   ├── layers/                  # ConvBlock, PicoBlock, SpatialPool, RepNeXt blocks
│   │   ├── assignment/              # STAL
│   │   ├── detector.py              # build_model() factory
│   │   └── lora.py                  # LoRA / QLoRA (6 variants)
│   ├── losses/
│   │   ├── e2e_loss.py              # E2E dual-head loss + ProgLoss
│   │   └── kd_loss.py               # Knowledge distillation losses
│   ├── utils/                       # Metrics, visualization, checkpoints
│   ├── trackers/                    # SORT, ByteTrack, BoT-SORT, DeepSORT, OC-SORT, StrongSORT
│   ├── solutions/                   # 17 ready-to-use vision solutions
│   └── analytics/                   # Benchmark, profiling, plots
├── scripts/                         # Training scripts (SSL, few-shot, etc.)
├── examples/                        # Ready-to-run example scripts
├── tests/                           # Unit & integration tests (pytest)
├── docs/                            # Documentation
├── docker/                          # Dockerfile + docker-compose
├── train.py                         # Main training entry point
├── test.py                          # Main inference entry point
└── pyproject.toml                   # Package configuration
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
| TXT labels | FP16 weights |
| Pascal VOC XML | TorchScript |

---

## Documentation

Full documentation is in the [`docs/`](docs/) folder:

| Document | Description |
|---|---|
| [Installation](docs/Installation.md) | Detailed installation guide |
| [LoRA Fine-Tuning](docs/LoRA-Fine-Tuning.md) | LoRA/QLoRA variants and usage |
| [Trackers](docs/Trackers.md) | Multi-object tracking guide |
| [FAQ](docs/FAQ.md) | Frequently asked questions |
| [Changelog](docs/CHANGELOG.md) | Version history |

---

## Contributing

We welcome contributions!

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
