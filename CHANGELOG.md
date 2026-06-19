# Changelog

All notable changes to FlashDet will be documented in this file.

## [1.0.0] — 2026-06-19

### Added
- **Package structure** — `pip install` from GitHub or PyPI
- **CLI** — `flashdet train`, `predict`, `val`, `export`, `check`, `settings`, `version`
- **Python API** — `Trainer`, `Predictor`, `Exporter`, `Validator`
- **Models** — FlashDet-m-0.5x, FlashDet-m, FlashDet-m-1.5x
- **LoRA fine-tuning** — 6 variants (standard, dora, lora_plus, adalora, ortho, lora_fa)
- **QLoRA** — INT8/NF4 quantized base weights + LoRA
- **Knowledge Distillation** — teacher-student training with configurable temperature
- **Trackers** — ByteTracker, SORTTracker, BoTSORT
- **Solutions** — ObjectCounter, SpeedEstimator, Heatmap, RegionCounter, QueueManager, DistanceCalculator, ParkingManager, SecurityAlarm, WorkoutMonitor, LiveInference, AnalyticsDashboard
- **Analytics** — Benchmark, Profiler, training curve plots
- **COCO pretrained weights** — auto-download on first use
- **ONNX export** — with simplification support
- **Mixed precision** — AMP (FP16) training
- **Multi-GPU** — DataParallel support
- **CI/CD** — GitHub Actions (lint + test on Python 3.9-3.12, auto-publish to PyPI)
- **Examples** — 7 runnable example scripts

### Architecture
- ShuffleNetV2 backbone (0.5x, 1.0x, 1.5x)
- GhostPAN neck
- NanoDet-Plus detection head with DFL regression
- Dynamic Soft Label Assignment
