# Changelog

All notable changes to FlashDet will be documented in this file.

## [1.3.0] — 2026-11-02

### Added
- **Fusion toolkit** — `flashdet/fusing/` with composable layer, kernel and
  graph passes: structural reparameterization, conv-BN-activation triples
  collapsed into one module, BatchNorm folded into conv/Linear weights, and
  stacked Linear merging. `flashdet fuse` reports the change and verifies
  numeric parity; fused models are saved as plain, replayable state dicts
  (inference-only, optionally FP16 via `flashdet fuse --half`).
- `scripts/benchmark_pipeline.py` — measures the whole FP32 → FP16 → fused →
  ONNX → INT8 pipeline on one checkpoint (params, file size, latency, FPS,
  mAP@0.5, box IoU) and emits a markdown table plus JSON.
- `scripts/plot_pipeline.py` — renders that report as a six-panel deployment
  dashboard (speed, size, accuracy, box IoU, trade-off, scorecard); the README
  now embeds it alongside the demo run's training curves.
- **Pico-Zoo backbones** — `FlashDet-P` now accepts all 40 architecture names from the Pico model table via `--backbone` / `backbone_type` (`repvit`, `mobileone`, `edgenext`, `mambaout`, `pp_lcnet`, `convnextv2`, ...).
- **Pico-Zoo necks** — matching lightweight PAN neck families (`rep`, `ghost`, `attention`, `mamba`, `lcnet`) selected automatically or with `--neck`.
- `scripts/train_pico_zoo.sh` for Pico backbone/neck experiments.
- **Quantization toolkit** — `flashdet/quantization/` with static (MinMax,
  percentile, histogram/MSE, SmoothQuant), AWQ, GPTQ, dynamic/weight-only
  INT8, QAT (fake quant, LSQ, PACT) and `QuantFlashDet` layer-aware policies.
- `flashdet.data.demo.create_demo_dataset()` + `scripts/create_demo_dataset.py`
  so examples work from a clean checkout (the old `data/demo` had annotations
  but no images).
- `scripts/create_synthetic_dataset.py` — reproducible synthetic dataset for
  training and smoke checks.
- Repository-level GitHub Copilot instructions and `copilot-setup-steps` workflow.
- **Automatic PyPI publishing** — after CI on `main`, the publish workflow
  checks the version and uploads new releases via PyPI trusted publishing or an
  API token.

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
- **ONNX export dependencies** — the `[export]` extra (and `[dev]`) now
  include `onnxscript`, which torch's dynamo ONNX exporter imports at runtime;
  a missing install now raises a message naming the extra.

### Fixed
- **mAP matching** — duplicate detections could steal a second, unmatched GT;
  matching now follows COCO semantics and accepts plain Python lists.
- **Validator** — rebuilds the checkpoint's actual architecture/backbone and
  warns on missing/unexpected weights instead of silently scoring a FlashDet
  model against YOLO weights.
- **Semi-supervised training** — fixed the epoch override signature and
  initialise the frozen teacher from the student.
- **YOLO training** — YOLOv10 now trains both o2o and o2m heads; YOLOv8 accepts
  a training loss. YOLOX training raises a clear NotImplementedError instead of
  failing with a TypeError.
- **EMA warmup** — exponential decay ramp now approaches the target instead of
  reaching only ~0.7 in realistic runs.
- **Tracking** — fixed DeepSORT cascade thresholds, KCF peak displacement,
  track-ID reset, ByteTrack low-score association, and repeated linear
  predictor calls.
- **Counting/heatmap** — fixed traffic direction label indexing, frame-delta
  speed, crowd grid negative indices, object-counter line hysteresis, heatmap
  decay/resolution changes, and parking spot ID gaps.
- **DDP validation** — validation now runs on the full validation set on rank 0,
  and early stopping is broadcast to all ranks.
- **Checkpoints** — strip `torch.compile` prefixes and training-only heads from
  inference weights.
- **Training performance** — opt-in channels-last, TF32 and cuDNN autotuning;
  fused/foreach optimizers, non-blocking copies and `set_to_none`; full-set
  validation on DDP rank 0.
- **Benchmark CLI** — `flashdet benchmark` reports parameters, FP16 weight
  size, median latency, throughput and optional mAP in one command; the
  validator now returns a COCO-style mAP@0.5 and mAP@0.5:0.95 summary.
- **Deployment package** — `flashdet export` now fuses Rep blocks, runs a
  PyTorch↔ONNXRuntime parity check automatically, honors `--simplify`, and can
  produce a checksum-verified INT8 ONNX with decoded-box IoU reporting
  (`--int8 --calib-images`).
- **Model zoo** — `flashdet weights` downloads checksum-verified demo
  checkpoints from GitHub Releases.
- **Decoupled weight decay** — AdamW/SGD no longer decay biases and
  normalization parameters, which removes a hard floor on late-training loss.
- **CLI validation** — `flashdet val` now derives `_annotations.coco.json`,
  prints mAP/per-class AP, and defaults to the same 0.01 score threshold used
  during training validation.
- **DySample robustness** — use `reshape` for non-contiguous tensors (fixes
  quantized/hooked neck execution).
- **Inference preprocessing mismatch (critical)** — `Predictor` used BGR,
  `/255`, top-left padding and scale-only box mapping. It now uses the same
  RGB + ImageNet mean/std + centred letterbox and inverse-warp box mapping as
  validation. On the training-matrix checkpoint this raised measured mAP from
  0.0425 to 0.8573.
- **ProgLoss epoch alignment** — training used a 1-based epoch while validation
  used 0-based, so the two passes applied different ProgLoss weights. Both now
  use the same 0-based schedule.
- **Batch-size-dependent validation loss** — validation logged the batch-scaled
  gradient loss instead of the per-sample loss. Logging now uses
  `loss_states["loss_total"]`; gradients are unchanged.
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
