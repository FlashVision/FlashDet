# Changelog

All notable changes to FlashDet will be documented in this file.

## [1.1.0] — 2026-06-22

### Architecture Overhaul
- **YOLO26-based FlashDet** — Complete rewrite of FlashDet as a YOLO26-based detector with dual-head design (one-to-one NMS-free + one-to-many dense), DFL-free box regression via softplus decoding, and configurable model sizes (n, s, m, l)
- **STAL** — Small-Target-Aware Label Assignment replacing the old DSL assigner
- **ProgLoss** — Progressive Loss Balancing that ramps one-to-one branch weight during training
- **E2EDetectionLoss** — End-to-end loss combining classification (BCE), box (CIoU), and L1 regression

### New Architectures
- **DETR** — DEtection TRansformer with Hungarian matching
- **RT-DETR** — Real-Time DETR with hybrid encoder
- **YOLOv9** — With PGI and GELAN blocks
- **YOLOv10** — NMS-free with PSA attention
- **YOLOv11** — With C3k2 and C2PSA blocks
- **GroundingDINO** — Open-vocabulary detection with vision-language fusion

### Training Methods
- **MuSGD** — Hybrid Muon-SGD optimizer
- **Knowledge Distillation** — Teacher-student training with logit + feature KD
- **Self-Supervised Learning** — BYOL pretraining for backbone initialization
- **Semi-Supervised Learning** — Teacher-student with EMA pseudo-labels
- **Few-Shot Learning** — Frozen backbone fine-tuning from limited examples
- **Active Learning** — Entropy-based uncertainty querying

### Added
- **Registry system** — Pluggable backbones, necks, heads, detectors, trackers via `flashdet.registry`
- **OBB Head** — Oriented Bounding Box detection
- **Varifocal Loss** — Improved focal loss weighting
- **Data Augmentations** — Mosaic, MixUp, Copy-Paste
- **Dataset Download** — `flashdet.data.download_dataset()` for common datasets
- **Comprehensive Test Suite** — Tests covering all components

### Fixed
- **CIoU gradient flow** — Removed clamping in STAL that blocked gradients for non-overlapping boxes
- **Box decoding** — Replaced `exp()` with `F.softplus()` for stable, non-negative regression
- **GroundingDINO** — Fixed per-class logit training and GIoU loss in `_compute_loss`

## [1.0.0] — 2026-06-19

### Added
- **Package structure** — `pip install` from GitHub or PyPI
- **CLI** — `flashdet train`, `predict`, `val`, `export`, `check`, `settings`, `version`
- **Python API** — `Trainer`, `Predictor`, `Exporter`, `Validator`
- **LoRA fine-tuning** — 6 variants (standard, dora, lora_plus, adalora, ortho, lora_fa)
- **QLoRA** — INT8/NF4 quantized base weights + LoRA
- **Trackers** — ByteTracker, SORTTracker, BoTSORT
- **Solutions** — ObjectCounter, SpeedEstimator, Heatmap, RegionCounter, QueueManager, DistanceCalculator, ParkingManager, SecurityAlarm, WorkoutMonitor, LiveInference, AnalyticsDashboard
- **Analytics** — Benchmark, Profiler, training curve plots
- **ONNX export** — with simplification support
- **Mixed precision** — AMP (FP16) training
- **Multi-GPU** — DataParallel support
