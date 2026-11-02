# Copilot instructions for FlashDet

## Repository summary

FlashDet is an ultra-lightweight object detection framework for edge devices.
It contains:

- FlashDet detector family: Pico (`p`), Nano (`n`), Small (`s`), Medium (`m`),
  Large (`l`), XLarge (`x`)
- Pico-Zoo: 40 optional lightweight backbone names with matching neck families
- NMS-free one-to-one inference path plus a one-to-many training head
- Training methods: supervised, SSL, semi-supervised, few-shot and active learning
- LoRA / QLoRA fine-tuning
- Tracking, vision solutions, analytics and ONNX/FP16 export

Primary stack: Python 3.9–3.12, PyTorch 2.0+, NumPy, OpenCV, Pillow.
The package is `flashdet`; the main entry points are `train.py`, `test.py` and
`flashdet.cli`.

## Environment setup

Use Python 3.11 for parity with CI unless the task explicitly requires another
version.

```bash
python -m pip install --upgrade pip
pip install -e ".[dev,tracker,solutions,analytics]"
```

Optional extras:

- `.[all]` — ONNX, trackers, analytics
- `.[export]` — ONNX export only
- `.[dev]` — pytest, coverage, ruff

Always install the package in editable mode before running tests so imports use
the checkout, not an installed PyPI version.

## Validation commands

Run the smallest relevant checks first, then the full gate before finishing:

```bash
# Fast checks
make lint
make test-unit

# Focused tests (replace path/test name)
pytest tests/unit/models/test_pico_zoo.py -q
pytest tests/integration/test_pico_zoo_pipeline.py -q

# Full gate
make test
make test-security
make coverage
```

CI runs these jobs: `lint`, `smoke`, `security`, `load-simulation`,
`unit-tests`, `integration-tests`, `model-tests`, `build`. A change is not ready
if `make lint` or any test job fails.

Lint is intentionally pinned to Ruff 0.15.x and selects only `E4`, `E7`, `E9`
and `F`. Do not expand lint rules in an unrelated PR.

## Project layout

```text
flashdet/
├── cfg/config.py              # ModelConfig / TrainConfig / DataConfig
├── cli.py                     # flashdet train/val/predict/export/check
├── data/                      # dataset, dataloader, transforms, download, demo
├── engine/
│   ├── core/                  # EMA, callbacks, MuSGD
│   ├── training/              # Trainer + SSL/semi/few-shot/active trainers
│   ├── evaluation/            # Validator
│   └── inference/             # Predictor + postprocess
├── models/
│   ├── architectures/         # flashdet.py, flashdet_micro.py, YOLO variants
│   ├── backbone/              # lite, pico, pico_zoo, flash, YOLO
│   ├── neck/                  # pico, pico_zoo, YOLO
│   ├── head/                  # E2E dual head, YOLO/YOLOX heads
│   ├── layers/                # conv, reparam, micro, YOLO blocks
│   ├── assignment/stal.py     # STAL
│   ├── detector.py            # build_model() factory
│   └── lora.py                # LoRA / QLoRA
├── losses/                    # E2E, YOLO, Varifocal
├── trackers/                  # SORT/ByteTrack/BoT-SORT/DeepSORT/OC-SORT/StrongSORT
├── solutions/                 # counting, speed, heatmap, queue, alarm, etc.
├── analytics/                 # benchmark, profiler, FLOPs, reports
└── utils/checkpoint.py        # save/load + safe_torch_load
```

Key files:

- `flashdet/models/architectures/flashdet.py` — FlashDet, Pico and size configs
- `flashdet/models/backbone/pico_zoo_backbone.py` — 40 Pico backbones
- `flashdet/models/neck/pico_zoo_neck.py` — matching neck families
- `flashdet/models/layers/reparam.py` — reparameterizable Pico blocks
- `flashdet/engine/inference/predictor.py` — inference + input validation
- `flashdet/utils/checkpoint.py` — checkpoint safety and model rebuild
- `train.py` — CLI training loop and YAML/config resolution
- `test.py` — image/video/camera/eval entry point

## Conventions and constraints

- Prefer configuration (`ModelConfig` / `TrainConfig` / CLI flags) over new
  hardcoded training or architecture values.
- Pico uses exactly 3 FPN levels at strides `(8, 16, 32)`. Do not add a fourth
  level without updating the neck, head and anchor decoding together.
- Pico-Zoo names map to family presets; keep `PICO_BACKBONE_PRESETS`,
  `PICO_BACKBONE_ALIASES` and the neck mapping in sync.
- Keep inference NMS-free. `predict()` uses the one-to-one head only.
- Use `safe_torch_load()` for checkpoints. Never introduce
  `torch.load(..., weights_only=False)` in runtime code.
- Archive extraction must remain path-traversal safe (`_safe_target` in
  `flashdet/data/download.py`).
- Validate inference inputs in `Predictor.preprocess`; do not remove size/type
  checks.
- New behavior needs a test. Security fixes need a negative assertion (for
  example, the malicious payload did not execute).
- Do not commit datasets, checkpoints, `workspace/`, `data/`, caches,
  `.coverage`, `__pycache__/`, `.pytest_cache/` or `.ruff_cache/`.
- Keep public exports updated in the relevant `__init__.py` and `__all__`.
- Update `docs/` when changing user-facing behavior.

## Common tasks

Build a Pico model:

```python
from flashdet import FlashDet
model = FlashDet(num_classes=80, size="p", backbone_type="repvit")
```

Train with the synthetic demo:

```bash
python scripts/create_demo_dataset.py
python train.py --model-size p --backbone repvit --epochs 2 \
  --batch-size 4 --input-size 320 \
  --train-images data/demo/train --val-images data/demo/valid \
  --save-dir workspace/demo --val-interval 1 --workers 0
```

Run inference:

```bash
python test.py --model workspace/demo/model_best_inference.pth \
  --image data/demo/valid/valid_0001.jpg --device cpu
```

## Pull request expectations

- Keep changes focused; one concern per PR.
- Add or update tests for behavior changes.
- Run `make lint` and the relevant test subset before opening a PR.
- Do not rewrite published git history or force-push unless explicitly asked.
- Prefer clear commit messages that describe the behavior change.
