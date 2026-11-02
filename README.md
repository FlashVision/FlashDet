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

For measured FP32 → FP16 → fused → INT8 numbers on one checkpoint, see
[Optimization pipeline](#optimization-pipeline).

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

## Optimization pipeline

The optimization steps are independent but designed to stack. Each stage is
checked against the previous one, so you can see exactly what it buys and what
it costs:

```text
checkpoint.pth (FP32) ...................... 1.37 MB   mAP 0.601
   │
   ├─ flashdet fuse ........................ 1.21 MB   mAP 0.601
   │     ├─ flashdet fuse --half ........... 0.63 MB   mAP 0.601
   │     └─ flashdet export --simplify ..... 1.22 MB   mAP 0.601
   │            └─ export --int8 --calib-images ...
   │                                       .. 0.52 MB   mAP 0.589
   │
   └─ flashdet export --simplify ........... 1.28 MB   mAP 0.601
```

The first row is the PyTorch checkpoint; the right column is the on-disk size
and validation accuracy of each artifact. Fusion and export are alternatives
that compose: fusing first gives the smallest ONNX and INT8 files.

Measured on the `p-demo-synth10` checkpoint (369K params, 10 classes, 128×128
input) on a 16-thread CPU with PyTorch 2.13 and onnxruntime 1.21. Latency is
the raw forward pass (decode/NMS excluded), median of 100 runs after 20
warmups. The validation split is the 200-image synthetic set.

| Stage | Params | File size | Latency | FPS | mAP@0.5 | Box IoU vs FP32 |
|---|---:|---:|---:|---:|---:|---:|
| FP32 PyTorch | 369,384 | 1.37 MB | 4.80 ms | 208 | 0.601 | 1.000 |
| FP32 PyTorch, fused | 356,432 | 1.21 MB | 2.98 ms | 335 | 0.601 | 1.000 |
| FP16 PyTorch | 369,384 | 0.72 MB | 35.9 ms* | 28 | 0.601 | 0.999 |
| FP16 PyTorch, fused | 356,432 | 0.63 MB | 26.8 ms* | 37 | 0.601 | 0.999 |
| FP32 ONNX | 369,384 | 1.28 MB | 1.10 ms | 906 | 0.601 | 1.000 |
| FP32 ONNX, fused | 356,432 | 1.22 MB | 0.81 ms | 1,231 | 0.601 | 1.000 |
| INT8 ONNX | 369,384 | 0.61 MB | 0.83 ms | 1,210 | 0.594 | 0.900 |
| INT8 ONNX, fused | 356,432 | 0.52 MB | 0.64 ms | 1,551 | 0.589 | 0.921 |

\* General-purpose CPUs do not accelerate FP16 arithmetic, so the FP16 latency
is *worse* than FP32. Treat FP16 as a storage/GPU format; the file size is the
point. On a GPU, FP16 also reduces memory traffic.

End to end, the demo model goes from **4.80 ms / 1.37 MB / mAP 0.601** to
**0.64 ms / 0.52 MB / mAP 0.589**: 7.5× faster, 2.6× smaller, and 1.2 points
(2.0% relative) of mAP. Fusion alone makes eager PyTorch 1.6× faster; the
ONNX export and INT8 quantization then remove most of the remaining runtime.

Reproduce the table on any checkpoint:

```bash
flashdet weights --name p-demo-synth10
python scripts/create_synthetic_dataset.py --output data/synth10
python scripts/benchmark_pipeline.py \
    --model weights/flashdet_p-demo-synth10.pth \
    --val-images data/synth10/valid \
    --input-size 128 \
    --output workspace/pipeline/report.json
```

The released demo checkpoint is FP16, so the script upcasts it before
measuring the FP32 baseline. It exports every artifact, measures each stage,
and writes the table plus a JSON report (`--output`). The dataset generator is
seeded, so the demo numbers are reproducible; your own checkpoint and
validation split will give different absolute values but the same relative
ordering.

### What each stage changes

| Step | What it does | Measured effect | What it costs |
|---|---|---|---|
| **FP16 weights** | Stores parameters as 2-byte floats | 1.37 → 0.72 MB (−47%) | No accuracy change; slower on CPU, so use it for GPU or as a download format |
| **Layer + kernel fusion** | Folds BatchNorm into conv/Linear weights, collapses reparameterizable branches, turns `conv → BN → act` into one module call | PyTorch 4.80 → 2.98 ms (1.6×); 13K fewer params (−3.5%); 1.37 → 1.21 MB | None measurable (parity ≈ 3e-5); shared modules are skipped, so a few BN layers can remain |
| **ONNX export + simplify** | Removes the Python dispatcher, folds constants and dead branches | PyTorch 4.80 → 1.10 ms (4.4×) | Adds onnxruntime as a runtime dependency |
| **INT8 QDQ** | Statically calibrated 8-bit weights *and* activations | FP32 ONNX 1.10 → 0.83 ms; fused 0.81 → 0.64 ms; fused file 1.22 → 0.52 MB | mAP 0.601 → 0.589, box IoU 0.90–0.92; needs representative calibration images |

A few things worth knowing when reading the table:

- **Fusion matters more in eager PyTorch than in ONNX.** Folding BN removes a
  kernel launch per block; ONNX Runtime and TensorRT already fuse some of that,
  so the remaining gain there is smaller (1.10 → 0.81 ms).
- **INT8 is where most of the size win is.** Quantizing the fused graph keeps
  the folded weights and drops the redundant BN scales, which is why the fused
  INT8 file is the smallest artifact (0.52 MB).
- **mAP and box IoU answer different questions.** mAP tracks detection quality;
  box IoU compares the boxes each stage produces against FP32 on the same
  images. INT8 loses about 2% relative mAP while boxes stay at ~0.91 mean IoU.
  Check both if your application reuses fixed confidence thresholds.
- **Sizes are files on disk.** Inference checkpoints omit the auxiliary
  training head, and ONNX stores folded constants instead of BN buffers, so
  "params × 4 bytes" is only a rough guide.
- **Fuse at export time, not before training.** Fusion rewrites module paths
  (a fused triple becomes one module), so fused checkpoints are for inference
  and benchmarking, not for resuming training.

### Recommended combinations

| Deployment | Artifact | Why |
|---|---|---|
| CUDA server, accuracy first | `flashdet fuse --half` (0.63 MB) | FP16 compute on GPU, fused convs, no quantization loss |
| CPU server / edge | `flashdet export --int8` from the fused model | 0.52 MB, ~1.5k FPS on 16 threads, mAP within 2% of FP32 |
| Portable runtime (no PyTorch) | `flashdet export --simplify` (FP32 ONNX) | Matches PyTorch at ≈1e-4, no calibration data needed |
| Fine-tuning base | Original checkpoint | Training needs the unfused graph |

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

## Fusion

The fusing package is under `flashdet/fusing`. Passes fold redundant work out
of a trained model before deployment and report what changed:

```python
import torch
from flashdet.fusing import run_fusions, default_passes

probe = torch.randn(1, 3, 320, 320)
fused, report = run_fusions(model, default_passes(example_input=probe),
                            example_input=probe)
print(report)
```

| Pass | What it does |
|---|---|
| `rep_fusion` | Collapses multi-branch reparameterizable blocks into single convolutions |
| `conv_bn_act` | Replaces `conv -> BN -> activation` triples with one module call |
| `conv_bn`, `linear_bn` | Folds BatchNorm into the preceding conv / Linear weights |
| `linear_pair` | Merges stacked Linear projections into one matrix multiply |

`run_fusions` deep-copies the model, applies the passes, and compares the fused
forward against the original. A pass only rewrites a module attribute when the
traced graph uses it at exactly one call site, so shared modules (for example
one activation applied in two branches) are left untouched.

```bash
flashdet fuse --model best.pth                      # report + parity check
flashdet fuse --model best.pth --output fused.pth   # save a replayable checkpoint
```

Fused checkpoints are ordinary state dicts (no pickled modules), so they load
through the normal safe checkpoint path and work with `export`, `benchmark`,
`val` and `predict`. On the demo checkpoint, fusion collapses 20
reparameterizable blocks and 47 conv-BN-activation triples with max|Δ| ≈ 3e-5
against the unfused model.

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
├── fusing/              # layer, kernel and graph fusion passes
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
