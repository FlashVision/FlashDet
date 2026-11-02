# Changelog

All notable changes to FlashDet will be documented in this file.

## [Unreleased]

### Added
- **Pico-Zoo backbones** — `FlashDet-P` now accepts all 40 architecture names from the Pico model table via `--backbone` / `backbone_type` (`repvit`, `mobileone`, `edgenext`, `mambaout`, `pp_lcnet`, `convnextv2`, ...).
- **Pico-Zoo necks** — matching lightweight PAN neck families (`rep`, `ghost`, `attention`, `mamba`, `lcnet`) selected automatically or with `--neck`.
- `scripts/train_pico_zoo.sh` and `docs/Pico-Zoo.md`.
- `flashdet.data.demo.create_demo_dataset()` + `scripts/create_demo_dataset.py`
  so examples work from a clean checkout (the old `data/demo` had annotations
  but no images).
- Archived the old `currDATA/` training logs under `docs/training-logs/`.

### Changed
- **Config-driven defaults** — Pico stem/depths/neck width, FPN strides, LR scaling
  (`nbs`, `lrf`), gradient clipping, EMA decay/warmup, multi-scale sizes and
  validation score/detection limits now come from `ModelConfig` / `TrainConfig`
  instead of being hardcoded in `train.py` and the detector.
- **YAML configs are actually applied** — `train.py --config` now starts from the
  loaded `Config` object instead of ignoring most YAML fields.
- **Multi-scale training** — `--multi-scale` is now wired to
  `config.train.multi_scale` (previously parsed but never enabled).
- **Few-shot model size** — uses `resolve_flashdet_size()` instead of a
  three-entry hardcoded mapping.

### Fixed
- **Pico training CSV** — validation loss/mAP columns were aligned by list index instead of epoch, so logging stopped writing validation metrics after the first few validation epochs.
- **Pico checkpoint loading** — `backbone_type` and `neck_type` are now saved in checkpoints and restored by `test.py`, `flashdet predict` and `flashdet export`.
- **Pico backbone sniffing** — old checkpoints still infer `repnext` from state-dict keys, while new zoo checkpoints no longer get misdetected as RepNeXt.

## [1.1.0] — 2026-06-22

### Architecture Overhaul
- **FlashDet v2** — Complete rewrite with dual-head design (one-to-one NMS-free + one-to-many dense), DFL-free box regression, and configurable model sizes (p, n, s, m, l, x)
- **STAL** — Small-Target-Aware Label Assignment replacing the old DSL assigner
- **ProgLoss** — Progressive Loss Balancing that ramps one-to-one branch weight during training
- **E2EDetectionLoss** — End-to-end loss combining classification (BCE), box (CIoU), and L1 regression
- **PicoBackbone** — New reparameterizable backbone option for FlashDet-Pico

### Training Methods
- **MuSGD** — Hybrid Muon-SGD optimizer
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
