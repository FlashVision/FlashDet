<p align="center">
  <img src="https://raw.githubusercontent.com/FlashVision/FlashDet/main/assets/logo.png" width="180" alt="FlashDet Logo">
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
  <b>Ultra-lightweight real-time object detection for edge devices, with a simple training API, NMS-free inference, LoRA fine-tuning, tracking and analytics.</b>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#model-sizes">Models</a> •
  <a href="#pico-zoo">Pico-Zoo</a> •
  <a href="#training">Training</a> •
  <a href="#inference">Inference</a> •
  <a href="#testing">Testing</a> •
  <a href="#documentation">Docs</a>
</p>

---

## What is FlashDet?

FlashDet is an end-to-end object detection framework. You give it a
COCO-format dataset; it gives you a trained detector, validation metrics, and a
deployable FP16 or ONNX model.

It is designed to be:

- **Small** — the default Pico model is ~370K parameters (~0.71 MB FP16).
- **Fast** — NMS-free one-to-one inference; no post-processing NMS step.
- **Practical** — one Python API and one CLI for training, validation,
  inference, export, tracking and analytics.
- **Hackable** — backbones, necks, heads, losses and trackers are pluggable.

No YOLO config files are required. The dataset loader reads COCO JSON directly
and discovers class names automatically.

---

## Quick Start

### 1. Install

```bash
pip install flashdet
```

For tracking, analytics and ONNX export:

```bash
pip install "flashdet[all]"
```

From source:

```bash
git clone https://github.com/FlashVision/FlashDet.git
cd FlashDet
pip install -e ".[all]"
```

Check that everything works:

```bash
flashdet check
```

### 2. Create a tiny demo dataset

No dataset? Generate a small synthetic one:

```bash
python scripts/create_demo_dataset.py
```

This creates `data/demo/train` and `data/demo/valid` with images and COCO
annotations.

### 3. Train

```bash
python train.py \
  --model-size p \
  --backbone repvit \
  --epochs 2 \
  --batch-size 4 \
  --input-size 320 \
  --train-images data/demo/train \
  --val-images data/demo/valid \
  --save-dir workspace/demo \
  --val-interval 1 \
  --workers 0
```

The best checkpoint is written to `workspace/demo/model_best_inference.pth`.
If you do not have CUDA, add `--device cpu` to the command.

### 4. Run inference

```bash
python test.py \
  --model workspace/demo/model_best_inference.pth \
  --image data/demo/valid/valid_0001.jpg \
  --device cpu
```

Or use the CLI:

```bash
flashdet predict \
  --model workspace/demo/model_best_inference.pth \
  --source data/demo/valid/valid_0001.jpg
```

That is the full loop: data → training → checkpoint → predictions.

---

## Why FlashDet?

| Feature | What it means for you |
|---|---|
| **NMS-free inference** | Faster deployment and simpler post-processing |
| **STAL assignment** | Better supervision for small objects |
| **ProgLoss** | Trains the dense head first, then refines the NMS-free head |
| **Pico-Zoo** | Choose from 40 lightweight backbone names under one detector |
| **LoRA / QLoRA** | Fine-tune with a small fraction of trainable parameters |
| **FP16 + ONNX** | Export for CPU, GPU, mobile and edge runtimes |
| **Tracking + solutions** | Counting, speed, heatmaps, queues, alarms and more |
| **Tests + security checks** | Unit, integration, security and load/chaos tests in CI |

---

## Model Sizes

All sizes use the same FlashDet detection head, loss and NMS-free inference
path. Parameter counts below are for 80 classes with the default configuration.

| Model | Size | Inference Params | FP16 Weights | Good for |
|---|---:|---:|---:|---|
| **FlashDet-P** (Pico) | `p` | ~370K | ~0.71 MB | Microcontrollers, browser, mobile |
| **FlashDet-N** (Nano) | `n` | ~804K | ~1.53 MB | Edge devices, real-time CPU |
| **FlashDet-S** (Small) | `s` | ~1.86M | ~3.55 MB | Balanced edge accuracy |
| **FlashDet-M** (Medium) | `m` | ~4.03M | ~7.68 MB | GPU / desktop accuracy |
| **FlashDet-L** (Large) | `l` | ~6.88M | ~13.1 MB | Higher accuracy |
| **FlashDet-X** (XLarge) | `x` | ~10.7M | ~20.4 MB | Maximum accuracy |

```python
from flashdet import FlashDet

model = FlashDet(num_classes=80, size="p")
print(model.get_model_info())
```

> The Pico-Zoo variants below can be smaller or larger than the default Pico,
> depending on the backbone family and width.

---

## Training Benchmark

All six sizes were trained and validated on a reproducible **1,000-image,
10-class** synthetic dataset. This is a smoke benchmark for correctness and
convergence, not a COCO accuracy claim.

**15 epochs, 128×128 input, CPU:**

| Model | Params | mAP@0.5 | Training time |
|---|---:|---:|---:|
| FlashDet-P | 0.40M | 0.788 | 2m 35s |
| FlashDet-N | 0.85M | 0.687 | 2m 42s |
| FlashDet-S | 1.95M | 0.756 | 3m 28s |
| FlashDet-M | 4.23M | 0.851 | 4m 48s |
| FlashDet-L | 7.25M | 0.830 | 6m 23s |
| FlashDet-X | 11.26M | 0.883 | 8m 00s |

What the run showed:

- Every size trained, validated, saved, reloaded and exported to ONNX.
- **P** is fastest on short runs because its default backbone is pretrained.
- **N/S** train from scratch and need more epochs than P.
- **M** is the best accuracy/time balance in this test.
- **X** gives the highest final mAP but is the slowest.

The benchmark also exposed and fixed two training bugs: a ProgLoss epoch
mismatch between training and validation, and a batch-size-dependent validation
loss.

Reproduce it with:

```bash
python scripts/create_synthetic_dataset.py \
  --output /tmp/synth10 --train 1000 --val 200 --size 192 --classes 10
```

Full numbers, commands and per-model notes:
**[docs/TRAINING-MATRIX.md](docs/TRAINING-MATRIX.md)**.

---

## Pico-Zoo

`FlashDet-P` supports **40 lightweight backbone names** from the Pico architecture
table. Each backbone has a matching neck family, and the neck is selected
automatically unless you override it.

| Family | Backbone examples | Matching neck |
|---|---|---|
| Reparameterized CNN | `repvit`, `mobileone`, `fastvit`, `repghostnet` | `rep` |
| Lightweight attention | `edgenext`, `mobilevitv2`, `efficientformerv2`, `shvit`, `efficientvit`, `swiftformer` | `attention` |
| Lightweight CNN | `efficientnetv2`, `emo`, `fasternet`, `starnet`, `mobilenetv4`, `ghostnetv2` | `ghost` |
| Visual Mamba / SSM | `mobilemamba`, `efficientvmamba`, `mambavision`, `mambaout`, `efficientvim` | `mamba` |
| Detection-oriented CNN | `pp_lcnet`, `lcnet`, `esnet`, `pp_lcnetv2`, `peleenet`, `shufflenetv2` | `lcnet` |
| Modern efficient CNN | `inceptionnext`, `unireplknet`, `convnextv2`, `efficientrep`, `rexnet` | `rep` |
| Lightweight transformers | `mobileformer`, `edgevit`, `tinyvit`, `levit`, `lowformer` | `attention` |
| Additional SSM | `localmamba`, `vmamba`, `plainmamba` | `mamba` |

```bash
# RepViT-style Pico
python train.py --model-size p --backbone repvit ...

# EdgeNeXt-style Pico with attention neck
python train.py --model-size p --backbone edgenext --neck attention ...

# MambaOut-style Pico
python train.py --model-size p --backbone mambaout --neck mamba ...

# PP-LCNet-style Pico
python train.py --model-size p --backbone pp_lcnet --neck lcnet ...
```

Or use the helper script:

```bash
BACKBONE=repvit NECK=rep bash scripts/train_pico_zoo.sh
```

Full list, parameter counts and family mapping:
**[docs/Pico-Zoo.md](docs/Pico-Zoo.md)**.

---

## Training

### Dataset format

FlashDet expects COCO JSON next to each split's images:

```text
data/my_dataset/
├── train/
│   ├── image_0001.jpg
│   ├── image_0002.jpg
│   └── _annotations.coco.json
└── valid/
    ├── image_0001.jpg
    └── _annotations.coco.json
```

Class names are read from the annotation file. You can override them with
`--class-file classes.txt`.

### Train from the command line

```bash
python train.py \
  --model-size n \
  --epochs 100 \
  --batch-size 32 \
  --input-size 320 \
  --train-images data/my_dataset/train \
  --val-images data/my_dataset/valid \
  --save-dir workspace/my_model \
  --amp \
  --device cuda
```

Common options:

| Option | Purpose |
|---|---|
| `--model-size p/n/s/m/l/x` | Choose the detector size |
| `--backbone ...` | Pico backbone name (Pico only) |
| `--neck ...` | Pico neck family (Pico only) |
| `--amp` | Mixed precision training |
| `--multi-gpu` | DDP / DataParallel training |
| `--mosaic`, `--mixup` | Stronger augmentation |
| `--multi-scale` | Random input sizes during training |
| `--resume checkpoint.pth` | Resume training |
| `--finetune checkpoint.pth` | Warm-start from weights |
| `--config config.yaml` | Load model/data/train settings from YAML |

### Train from Python

```python
from flashdet import Trainer

trainer = Trainer(
    model_size="p",
    train_images="data/my_dataset/train",
    val_images="data/my_dataset/valid",
    epochs=100,
    batch_size=32,
    device="cuda",
)
trainer.train()
```

### Training methods

| Method | Class | Script |
|---|---|---|
| Standard supervised | `Trainer` | `train.py` |
| Self-supervised pretraining | `SSLTrainer` | `scripts/train_ssl.py` |
| Semi-supervised | `SemiSupervisedTrainer` | `scripts/train_semi_supervised.py` |
| Few-shot | `FewShotTrainer` | `scripts/train_few_shot.py` |
| Active learning | `ActiveLearningTrainer` | `scripts/train_active_learning.py` |

Examples:

```bash
# Self-supervised backbone pretraining
python scripts/train_ssl.py --method byol --data-dir data/unlabeled --epochs 100

# Semi-supervised training
python scripts/train_semi_supervised.py \
  --train-images data/train \
  --unlabeled-dir data/unlabeled \
  --pseudo-threshold 0.7

# Few-shot fine-tuning
python scripts/train_few_shot.py \
  --base-checkpoint workspace/base.pth \
  --n-shot 10 --freeze-backbone

# Active learning
python scripts/train_active_learning.py \
  --train-images data/train \
  --unlabeled-pool data/unlabeled \
  --query-strategy entropy --budget 50 --rounds 5
```

### LoRA / QLoRA fine-tuning

Train only a small adapter instead of the full model:

```bash
# LoRA variants: standard, dora, lora_plus, adalora, ortho, lora_fa
python train.py --lora --lora-variant dora --lora-rank 8 --lora-alpha 16

# QLoRA: quantized base weights + LoRA
python train.py --qlora --qlora-dtype nf4 --lora-rank 8
```

See **[docs/LoRA-Fine-Tuning.md](docs/LoRA-Fine-Tuning.md)**.

---

## Inference

> Inference uses the same RGB + ImageNet mean/std + centred letterbox pipeline
> as validation. Do not add a separate `/255` normalization when using the
> built-in `Predictor`.

### Python

```python
from flashdet import Predictor

predictor = Predictor(
    model_path="workspace/my_model/model_best_inference.pth",
    device="cuda",
    conf_thresh=0.25,
)

results = predictor("photo.jpg")
for bbox, score, class_id in results:
    print(bbox, score, class_id)
```

### CLI

```bash
# Single image
flashdet predict --model best.pth --source photo.jpg

# Save an annotated image
flashdet predict --model best.pth --source photo.jpg --output results/
```

`test.py` also supports video, webcam and dataset evaluation:

```bash
# Image
python test.py --model best.pth --image photo.jpg --device cuda

# Video
python test.py --model best.pth --video input.mp4 --output results/

# Webcam
python test.py --model best.pth --camera 0 --show

# Dataset evaluation
python test.py --model best.pth --eval --image data/valid --output results/
```

### Export

```bash
# ONNX
flashdet export --model best.pth --output model.onnx

# FP16 weights are written automatically during training:
#   model_best_inference.pth
#   model_best_fp16.pth
```

---

## Evaluation

```bash
flashdet val --model best.pth --val-images data/my_dataset/valid
```

The validator reports mAP@0.5 and per-class AP. The training loop saves the best
checkpoint by mAP, not just by loss.

---

## Tracking

Multi-object tracking keeps IDs stable across frames:

```python
from flashdet.trackers import FlashTracker

tracker = FlashTracker(max_age=30, min_hits=3, iou_threshold=0.3)
tracks = tracker.update(detections)
```

| Tracker | Method | Best for |
|---|---|---|
| `FlashTracker` / `SortTracker` | IoU + Kalman | General purpose, fast |
| `ByteTracker` | Two-stage association | Low-confidence detections |
| `OCSortTracker` | Observation-centric SORT | Occlusion handling |
| `DeepSortTracker` | Mahalanobis + appearance | Crowded scenes |
| `BoTSortTracker` / `AppearanceTracker` | ReID + camera motion | Re-identification |
| `StrongSortTracker` | Enhanced DeepSORT | High-accuracy offline |

See **[docs/Trackers.md](docs/Trackers.md)**.

---

## Solutions

Ready-made building blocks on top of detection + tracking:

```python
from flashdet import Predictor
from flashdet.trackers import FlashTracker
from flashdet.solutions import ObjectCounter

predictor = Predictor(model_path="best.pth", device="cuda")
counter = ObjectCounter(
    predictor=predictor,
    tracker=FlashTracker(),
    line_points=[(100, 400), (900, 400)],
)

annotated_frame, results = counter.process_frame(frame)
```

| Solution | Use case |
|---|---|
| `ObjectCounter` | Count objects crossing a line |
| `RegionCounter` | Count objects inside polygon zones |
| `QueueManager` | Queue length and wait time |
| `SpeedEstimator` | Real-world speed from track history |
| `DistanceCalculator` | Distance between objects |
| `ParkingManager` | Parking spot occupancy |
| `SecurityAlarm` | Restricted-zone intrusion alerts |
| `WorkoutMonitor` | Exercise repetitions |
| `Heatmap` | Detection density over time |
| `LiveInference` | Webcam / stream inference |
| `AnalyticsDashboard` | Aggregated statistics dashboard |

---

## Analytics

```python
from flashdet.analytics import Benchmark, Profiler, FLOPsCounter

Benchmark(model_path="best.pth", device="cuda").run()
Profiler(model_path="best.pth").run()
FLOPsCounter(model=model, input_size=320).count()
```

---

## Testing

The project is tested at four levels:

```bash
make test-unit           # fast unit + smoke tests
make test-integration    # CLI, training, inference, checkpoint round-trip
make test-security       # checkpoint/archive/input security regressions
make test-load           # 1,000 concurrent inference + malformed inputs
make coverage            # coverage report
make lint                # ruff
```

Details:

- **[tests/TESTING.md](tests/TESTING.md)** — test structure and TDD workflow
- **[docs/SECURITY-AUDIT.md](docs/SECURITY-AUDIT.md)** — pen-test findings and fixes
- **[docs/LOAD-TEST.md](docs/LOAD-TEST.md)** — 1,000 / 5,000 user simulation results
- **[docs/HARDCODING-AUDIT.md](docs/HARDCODING-AUDIT.md)** — what moved into config

---

## Docker

```bash
# Build
docker build -t flashdet -f docker/Dockerfile .

# Run inference
docker run --gpus all -v $(pwd)/data:/app/data flashdet \
  predict --model best.pth --source data/test.jpg
```

Or:

```bash
cd docker
docker compose up
```

---

## Project Structure

<details>
<summary>Click to expand the full tree</summary>

```text
FlashDet/
├── flashdet/                        # Main package
│   ├── __init__.py                  # Public API + version
│   ├── cli.py                       # CLI entry point
│   ├── registry.py                  # Pluggable component registry
│   ├── cfg/                         # Model / data / train configuration
│   ├── data/                        # Datasets, loaders, transforms, demo generator
│   ├── engine/
│   │   ├── core/                    # Callbacks, EMA, MuSGD optimizer
│   │   ├── training/                # Trainer, SSL, semi-supervised, few-shot, active learning
│   │   ├── evaluation/              # Validator
│   │   └── inference/               # Predictor + post-processing
│   ├── models/
│   │   ├── architectures/           # FlashDet, Pico, Micro, YOLO variants
│   │   ├── backbone/                # Lite, Pico, Pico-Zoo, Flash, YOLO backbones
│   │   ├── neck/                    # Pico, Pico-Zoo, YOLO necks
│   │   ├── head/                    # E2E dual head, YOLO/YOLOX heads
│   │   ├── layers/                  # Conv, reparameterization, micro/YOLO blocks
│   │   ├── assignment/              # STAL
│   │   ├── detector.py              # build_model() factory
│   │   └── lora.py                  # LoRA / QLoRA variants
│   ├── losses/                      # E2E, YOLO and Varifocal losses
│   ├── utils/                       # Checkpoints, metrics, visualization, optimizations
│   ├── trackers/                    # SORT, ByteTrack, BoT-SORT, DeepSORT, OC-SORT, StrongSORT
│   ├── solutions/                   # Ready-to-use vision solutions
│   └── analytics/                   # Benchmark, profiling, plots, reports
├── configs/                         # Example YAML training configs
├── scripts/                         # Training, export, benchmark, security, load-test scripts
├── examples/                        # Ready-to-run Python examples
├── tests/                           # Unit, integration, smoke and security tests
├── docs/                            # Guides, audits, changelog, archived training logs
├── docker/                          # Dockerfile + docker-compose
├── train.py                         # Main training entry point
├── test.py                          # Main inference / evaluation entry point
├── Makefile                         # test / coverage / lint / security targets
└── pyproject.toml                   # Package configuration
```

</details>

---

## Documentation

| Document | What you will find |
|---|---|
| [Installation](docs/Installation.md) | Detailed install and environment setup |
| [Pico-Zoo](docs/Pico-Zoo.md) | All 40 backbones, necks and selection guide |
| [LoRA Fine-Tuning](docs/LoRA-Fine-Tuning.md) | LoRA / QLoRA variants and usage |
| [Trackers](docs/Trackers.md) | Multi-object tracking guide |
| [Testing](tests/TESTING.md) | Unit, integration and TDD workflow |
| [Security Audit](docs/SECURITY-AUDIT.md) | Pen-test findings and hardening notes |
| [Load Test](docs/LOAD-TEST.md) | Concurrency results and serving recommendations |
| [Training Matrix](docs/TRAINING-MATRIX.md) | 1,000-image / 10-class run across all model sizes |
| [Hardcoding Audit](docs/HARDCODING-AUDIT.md) | Configuration cleanup notes |
| [FAQ](docs/FAQ.md) | Common questions |
| [Publishing](docs/PUBLISHING.md) | Auto-publish GitHub → PyPI |
| [Changelog](docs/CHANGELOG.md) | Version history |

---

## Releasing

PyPI publishing is automatic after CI passes:

1. Bump `project.version` in `pyproject.toml`.
2. Bump the fallback `__version__` in `flashdet/__init__.py`.
3. Update `docs/CHANGELOG.md`.
4. Push to `main`.

The publish workflow checks whether that version already exists on PyPI. If it
is new, it builds and uploads it; otherwise it skips. See
**[docs/PUBLISHING.md](docs/PUBLISHING.md)** for the one-time PyPI trusted
publisher setup.

---

## Contributing

```bash
git clone https://github.com/FlashVision/FlashDet.git
cd FlashDet
pip install -e ".[dev,all]"
make test
make lint
```

Before opening a pull request:

1. Add or update a test for your change.
2. Run `make test` and `make lint`.
3. Keep changes focused; one feature per PR.
4. Update the relevant doc in `docs/` if behaviour changes.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <a href="https://github.com/FlashVision/FlashDet"><b>FlashVision</b></a>
  — Open-source lightweight AI
</p>
