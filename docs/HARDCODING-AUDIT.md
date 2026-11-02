# Hardcoding Audit

This pass moved values that change per run/model out of Python control flow and
into `ModelConfig` / `TrainConfig`.  Pure architecture *defaults* (e.g. ImageNet
mean/std, 80 COCO classes) remain as API defaults and are documented as such.

## Fixed

| Location | Hardcoded before | Now |
|---|---|---|
| `cfg/config.py` | `strides = [8, 16, 32, 64]` while models use 3 levels | `strides = [8, 16, 32]` |
| `architectures/flashdet.py` | Pico-v2 stem 28, channels 56/112/224, depths 2/4/2 inline | `PICO_V2_CONFIG` + constructor overrides |
| `architectures/flashdet.py` | `E2EDualHead(..., num_levels=3)` | `num_levels=len(strides)` with validation |
| `architectures/flashdet.py` | Pico neck width 80 in the class | `neck_channels` default resolved from config |
| `train.py` | `nbs = 64`, `lrf = 0.1`, clip 10.0, EMA 0.9998/2000 | `config.train.{nbs,lrf,grad_clip_norm,ema_decay,ema_warmup}` |
| `train.py` | multi-scale sizes `[256..416]` | `config.train.multi_scale_sizes` + `--multi-scale-sizes` |
| `train.py` | `--multi-scale` parsed but never enabled | wired to `config.train.multi_scale` |
| `train.py` | validation `score_thr=0.01`, fallback classes 10 | `config.train.val_score_thr` / `max_det`, model classes |
| `train.py` | `--config` ignored most YAML fields | starts from the loaded `Config` object |
| `engine/training/trainer.py` | lrf 0.01, EMA 0.9998/2000, clip 10.0 | config-driven |
| `engine/training/few_shot_trainer.py` | EMA/clip/eta hardcoded, size map `{m: n}` | config-driven + `resolve_flashdet_size()` |
| `engine/inference/predictor.py` | default input 640 | checkpoint/config, fallback 320 |
| `cli.py` | export fallback 640, check 320, `.replace(".pth", ".onnx")` | config default, `os.path.splitext` |
| `cli.py` (predict/val) | input size 640/320 | checkpoint/config |
| `engine/evaluation/validator.py` | input size 320 only | checkpoint/config, fallback 320 |

## Intentionally still defaults

These are part of the public API or a data specification, not accidental magic
numbers. They are all overridable:

- `num_classes=80`, `input_size=(320, 320)` — API defaults.
- `SIZE_CONFIGS` / `PICO_BACKBONE_PRESETS` — architecture definitions.
- Dataset registry URLs/classes — dataset metadata.
- `scripts/train_*.sh` — explicit training recipes (environment variables override).
- `ModelEMA` constructor defaults — used only when a caller does not pass config.

## Regression tests

`tests/unit/models/test_config_defaults.py` checks:

- default FPN strides are exactly 3,
- head levels follow `strides`,
- 4-level strides fail loudly,
- Pico stem/depth/neck overrides reach the backbone and neck,
- `build_model()` forwards config selectors and overrides.
