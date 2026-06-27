# Changelog

All notable changes to FlashDet will be documented in this file.

## [1.1.0] — 2026-06-22

### Architecture Overhaul
- **FlashDet v2** — Complete rewrite with dual-head design (one-to-one NMS-free + one-to-many dense), DFL-free box regression, and configurable model sizes (p, n, s, m, l, x)
- **STAL** — Small-Target-Aware Label Assignment replacing the old DSL assigner
- **ProgLoss** — Progressive Loss Balancing that ramps one-to-one branch weight during training
- **E2EDetectionLoss** — End-to-end loss combining classification (BCE), box (CIoU), and L1 regression
- **PicoBackbone** — New reparameterizable backbone option for FlashDet-Pico

### Training Methods
- **MuSGD** — Hybrid Muon-SGD optimizer
- **Knowledge Distillation** — Teacher-student training with logit + feature KD
- **Self-Supervised Learning** — BYOL pretraining for backbone initialization
- **Semi-Supervised Learning** — Teacher-student with EMA pseudo-labels
- **Few-Shot Learning** — Frozen backbone fine-tuning from limited examples
- **Active Learning** — Entropy-based uncertainty querying

### Added
- **Registry system** — Pluggable backbones, necks, heads, detectors, trackers via `flashdet.registry`
- **Data Augmentations** — Mosaic, MixUp, Copy-Paste
- **Dataset Download** — `flashdet.data.download_dataset()` for common datasets

### Fixed
- **CIoU gradient flow** — Removed clamping in STAL that blocked gradients for non-overlapping boxes
- **Box decoding** — Replaced `exp()` with `F.softplus()` for stable, non-negative regression

## [1.0.0] — 2026-06-19

### Added
- **Package structure** — `pip install` from GitHub or PyPI
- **CLI** — `flashdet train`, `val`, `check`, `settings`, `version`
- **Python API** — `Trainer`, `Validator`
- **LoRA fine-tuning** — 6 variants (standard, dora, lora_plus, adalora, ortho, lora_fa)
- **QLoRA** — INT8/NF4 quantized base weights + LoRA
- **Trackers** — FlashTracker, MotionTracker, AppearanceTracker
- **Solutions** — ObjectCounter, SpeedEstimator, Heatmap, RegionCounter, QueueManager, DistanceCalculator, ParkingManager, SecurityAlarm, WorkoutMonitor, LiveInference, AnalyticsDashboard
- **Analytics** — Benchmark, Profiler, training curve plots
- **Mixed precision** — AMP (FP16) training
- **Multi-GPU** — DataParallel support
