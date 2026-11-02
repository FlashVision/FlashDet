<p align="center">
  <img src="https://raw.githubusercontent.com/FlashVision/FlashDet/main/assets/logo.png" width="160" alt="FlashDet">
</p>

<h1 align="center">FlashDet</h1>

<p align="center">
  <a href="https://pypi.org/project/flashdet/"><img src="https://img.shields.io/pypi/v/flashdet?color=blue&logo=pypi&logoColor=white" alt="PyPI"></a>
  <a href="https://github.com/FlashVision/FlashDet/actions"><img src="https://img.shields.io/github/actions/workflow/status/FlashVision/FlashDet/ci.yml?logo=github" alt="CI"></a>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Python-3.9+-3776ab?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

FlashDet is a lightweight object detection toolkit for edge and embedded
devices. It includes a small NMS-free detector family, a Pico backend with
selectable backbones, LoRA fine-tuning, tracking, counting utilities, ONNX
export, and a quantization package.

It is not a SOTA COCO model. The design target is a small, fast detector that
can be trained and deployed with limited hardware.

---

## Install

```bash
pip install flashdet
```

With trackers, analytics, and ONNX export:

```bash
pip install "flashdet[all]"
```

From source:

```bash
git clone https://github.com/FlashVision/FlashDet.git
cd FlashDet
pip install -e ".[all]"
```

Verify the installation:

```bash
flashdet check
```

Skip training entirely by grabbing the demo checkpoint:

```bash
flashdet weights --name p-demo-synth10          # ~0.75 MB, checksum-verified
```

---

## Quick start

Generate a small synthetic dataset:

```bash
python scripts/create_demo_dataset.py
```

Train a Pico model for a few epochs:

```bash
python train.py \
  --model-size p \
  --backbone repvit \
  --epochs 5 \
  --batch-size 16 \
  --input-size 128 \
  --train-images data/demo/train \
  --val-images data/demo/valid \
  --save-dir workspace/demo \
  --val-interval 1 \
  --workers 0
```

Run inference:

```bash
python test.py \
  --model workspace/demo/model_best_inference.pth \
  --image data/demo/valid/valid_0001.jpg \
  --device cpu
```

Or validate the checkpoint:

```bash
flashdet val \
  --model workspace/demo/model_best_inference.pth \
  --val-images data/demo/valid
```

---

## Model sizes

The numbers below are for 80 classes with the default configuration.

| Model | Size | Inference params | FP16 weights |
|---|---:|---:|---:|
| FlashDet-P | `p` | ~370K | ~0.71 MB |
| FlashDet-N | `n` | ~804K | ~1.53 MB |
| FlashDet-S | `s` | ~1.86M | ~3.55 MB |
| FlashDet-M | `m` | ~4.03M | ~7.68 MB |
| FlashDet-L | `l` | ~6.88M | ~13.1 MB |
| FlashDet-X | `x` | ~10.7M | ~20.4 MB |

```python
from flashdet import FlashDet

model = FlashDet(num_classes=80, size="p")
print(model.get_model_info())
```

---

## Pico backbones and necks

`FlashDet-P` accepts a set of lightweight backbone names through `--backbone`.
Each name selects a block family and a matching neck family.

| Family | Examples | Default neck |
|---|---|---|
| Reparameterized CNN | `repvit`, `mobileone`, `fastvit`, `repghostnet` | `rep` |
| Lightweight attention | `edgenext`, `mobilevitv2`, `efficientformerv2`, `shvit`, `efficientvit`, `swiftformer` | `attention` |
| Lightweight CNN | `efficientnetv2`, `emo`, `fasternet`, `starnet`, `mobilenetv4`, `ghostnetv2` | `ghost` |
| Mamba-style | `mobilemamba`, `efficientvmamba`, `mambavision`, `mambaout`, `efficientvim` | `mamba` |
| Detection CNN | `pp_lcnet`, `lcnet`, `esnet`, `pp_lcnetv2`, `peleenet`, `shufflenetv2` | `lcnet` |
| Modern CNN | `inceptionnext`, `unireplknet`, `convnextv2`, `efficientrep`, `rexnet` | `rep` |
| Lightweight transformer | `mobileformer`, `edgevit`, `tinyvit`, `levit`, `lowformer` | `attention` |
| Additional SSM | `localmamba`, `vmamba`, `plainmamba` | `mamba` |

```bash
python train.py --model-size p --backbone edgenext --neck attention ...
python train.py --model-size p --backbone mambaout --neck mamba ...
python train.py --model-size p --backbone pp_lcnet --neck lcnet ...
```

The attention/Mamba names are lightweight family implementations, not full
reproductions of every paper.

---

## Training

### Dataset layout

FlashDet expects COCO JSON next to each split:

```text
data/my_dataset/
├── train/
│   ├── image_0001.jpg
│   └── _annotations.coco.json
└── valid/
    ├── image_0001.jpg
    └── _annotations.coco.json
```

Class names are read from the annotation file. Use `--class-file` to override
them.

### CLI

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

Useful options:

| Option | Purpose |
|---|---|
| `--model-size p/n/s/m/l/x` | Select model size |
| `--backbone ...` | Pico backbone name |
| `--neck ...` | Pico neck family |
| `--amp` | Mixed precision |
| `--multi-gpu` | DDP/DataParallel training |
| `--mosaic`, `--mixup` | Additional augmentation |
| `--multi-scale` | Random input sizes |
| `--channels-last` | NHWC memory format |
| `--tf32` | TF32 on CUDA |
| `--cudnn-benchmark` | cuDNN autotuning |
| `--resume` | Resume from checkpoint |
| `--finetune` | Warm-start from weights |
| `--config` | YAML configuration |

Run `flashdet --help` for the full list — `predict`, `val`, `export`,
`benchmark`, `weights` and `download` are all in there.

### Python

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

### Other training modes

```bash
python scripts/train_ssl.py --method byol --data-dir data/unlabeled --epochs 100
python scripts/train_semi_supervised.py --train-images data/train --unlabeled-dir data/unlabeled
python scripts/train_few_shot.py --base-checkpoint workspace/base.pth --n-shot 10
python scripts/train_active_learning.py --train-images data/train --unlabeled-pool data/unlabeled
```

LoRA and QLoRA:

```bash
python train.py --lora --lora-variant dora --lora-rank 8
python train.py --qlora --qlora-dtype nf4 --lora-rank 8
```

---

## Validation, export and benchmarking

```bash
flashdet val --model best.pth --val-images data/valid
flashdet export --model best.pth --simplify          # ONNX, parity-checked, Rep blocks fused
flashdet export --model best.pth --int8 --calib-images data/valid   # + INT8 ONNX
flashdet benchmark --model best.pth --val-images data/valid        # FPS + mAP in one go
```

The export step runs a PyTorch↔ONNXRuntime parity check automatically and
reports a decoded-box IoU for the INT8 model, so the exported file is verified
before you ship it. On a 3-epoch sample this means: FP32/FP16 ONNX matches
PyTorch at max|Δ| ≈ 1e-4, and INT8 typically costs a few percent of box IoU
(depending on how much calibration data you give it).

Training writes these files:

```text
checkpoint_best.pth
checkpoint_last.pth
model_best_inference.pth
model_best_fp16.pth
model_last_inference.pth
model_final_inference.pth
results.json
training_log.csv
```

---

## Inference

Python:

```python
from flashdet import Predictor

predictor = Predictor(
    model_path="workspace/my_model/model_best_inference.pth",
    device="cuda",
    conf_thresh=0.25,
)

for bbox, score, class_id in predictor("photo.jpg"):
    print(bbox, score, class_id)
```

CLI:

```bash
flashdet predict --model best.pth --source photo.jpg
flashdet predict --model best.pth --source photo.jpg --output results/
```

`test.py` supports images, video, webcam, and dataset evaluation:

```bash
python test.py --model best.pth --image photo.jpg --device cuda
python test.py --model best.pth --video input.mp4 --output results/
python test.py --model best.pth --camera 0 --show
python test.py --model best.pth --eval --image data/valid --output results/
```

Inference uses the same RGB, ImageNet mean/std, and centred letterbox pipeline
as validation. Do not add another normalization step.

---

## Quantization

The quantization package is under `flashdet/quantization`.

```python
from flashdet.quantization import (
    quantize_static,
    quantize_awq,
    quantize_gptq,
    QuantFlashDet,
    QuantLayerPolicy,
)
```

Static calibration:

```python
qmodel = quantize_static(
    model,
    calib_loader=calib_loader,
    layer_prefixes=("backbone", "neck"),
    keep_float_prefixes=("head",),
    num_batches=32,
)
```

AWQ and GPTQ:

```python
qmodel = quantize_awq(model, calib_loader, bits=4, group_size=128)
qmodel = quantize_gptq(model, calib_loader, bits=8, group_size=-1)
```

Layer-aware policy:

```python
quantizer = QuantFlashDet(
    model,
    policy=QuantLayerPolicy(backbone="static", neck="static", head="none"),
)
qmodel = quantizer.quantize("static", calib_loader=calib_loader)
print(quantizer.report(qmodel))
```

QAT helpers and fake-quant modules are in
`flashdet.quantization.training_aware_quant`.

Static quantization in this package is calibration-based fake quantization so it
works with FlashDet's custom multi-output head. For hardware INT8, export to
ONNX, TensorRT or OpenVINO.

---

## Tests

```bash
make test-unit
make test-integration
make test-security
make test-load
make coverage
make lint
```

The suite covers models, losses, data transforms, inference, checkpoint
loading, trackers, solutions, quantization, security regressions, and a
1,000-user load/chaos simulation.

---

## Project layout

```text
flashdet/
├── cfg/                 # model/data/train configuration
├── data/                # datasets, transforms, download, demo generator
├── engine/              # training, evaluation, inference, EMA, callbacks
├── models/
│   ├── architectures/   # FlashDet, Pico, Micro, YOLO variants
│   ├── backbone/        # Lite, Pico, Pico-Zoo, Flash, YOLO
│   ├── neck/            # Pico, Pico-Zoo, YOLO
│   ├── head/            # E2E dual head, YOLO/YOLOX heads
│   ├── layers/          # conv, reparam, micro, YOLO blocks
│   ├── assignment/      # STAL
│   └── lora.py          # LoRA / QLoRA
├── losses/
├── quantization/        # static, AWQ, GPTQ, dynamic, QAT
├── trackers/
├── solutions/
├── analytics/
└── utils/
```

Top-level scripts:

```text
train.py                 training entry point
test.py                  inference/evaluation entry point
scripts/                 dataset, training, benchmark, load-test tools
configs/                 example YAML configs
docs/                    installation, FAQ, LoRA, trackers, changelog
```

---

## Documentation

| Document | Content |
|---|---|
| [Installation](docs/Installation.md) | Installation and environment setup |
| [LoRA Fine-Tuning](docs/LoRA-Fine-Tuning.md) | LoRA and QLoRA usage |
| [Trackers](docs/Trackers.md) | Multi-object tracking |
| [FAQ](docs/FAQ.md) | Common questions |
| [Changelog](docs/CHANGELOG.md) | Version history |

---

## Limitations

- This is a lightweight detector, not a SOTA COCO model.
- YOLOX training is not implemented; YOLOX inference/export works.
- Static quantization is fake-quant simulation for the custom FlashDet graph.
- The included benchmark uses synthetic data and is not a real-world accuracy
  result.
- CUDA and multi-GPU paths are implemented but the checks in this environment
  were run on CPU.

---

## Contributing

```bash
git clone https://github.com/FlashVision/FlashDet.git
cd FlashDet
pip install -e ".[dev,all]"
make test
make lint
```

Please add or update a test for behavior changes and keep pull requests focused.

---

## License

MIT License. See [LICENSE](LICENSE).
