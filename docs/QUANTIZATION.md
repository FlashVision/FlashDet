# Quantization

FlashDet ships a self-contained quantization toolkit under `flashdet/quantization`.

## Layout

```text
flashdet/quantization/
├── base.py
│   ├── QuantConfig
│   ├── MinMaxObserver
│   ├── PercentileObserver
│   ├── HistogramObserver / MSEObserver / EntropyObserver
│   ├── fake_quantize + straight-through estimator
│   ├── FakeQuantize
│   ├── LSQFakeQuantize
│   └── PACTActivation
├── static_quantization/
│   ├── ptq.py                  # calibration + static INT8 simulation
│   ├── smoothquant.py          # activation/weight channel balancing
│   └── observers are re-exported from base
├── dynamic_quantization/
│   ├── dynamic.py              # dynamic Linear + INT8 weight-only Conv
│   └── weight_only.py          # Int8WeightOnlyConv2d / Linear
├── training_aware_quant/
│   ├── qat.py                  # prepare_qat / convert_qat / qat_finetune
│   ├── fake_quant.py
│   ├── lsq.py
│   └── pact.py
└── quantflashdet/
    ├── layer_config.py         # per-group policy
    ├── quantizer.py            # QuantFlashDet
    └── report.py               # quantized layer report
```

## Algorithms included

| Family | Algorithms |
|---|---|
| Observers | MinMax, Percentile, Histogram/MSE, Entropy/KL-style clipping |
| Static PTQ | MinMax/Percentile/Histogram calibration, per-channel weights, SmoothQuant, eager `torch.ao` path |
| Weight-only PTQ | AWQ (activation-aware scaling), GPTQ (Hessian error compensation), group-wise 2/4/8-bit |
| Dynamic | Dynamic INT8 Linear, INT8 weight-only Conv2d/Linear |
| QAT | Observer-based fake quant, LSQ learned step size, PACT activation clipping, prepare/convert |
| FlashDet | Per-group backbone/neck/head policies, layer reports |

## Usage

### Layer-aware FlashDet quantization

```python
from flashdet.quantization import QuantFlashDet, QuantLayerPolicy

policy = QuantLayerPolicy(backbone="static", neck="static", head="none")
quantizer = QuantFlashDet(model, policy=policy)
qmodel = quantizer.quantize("static", calib_loader=calib_loader, num_batches=32)
print(quantizer.report(qmodel))
```

The default policy keeps the sensitive detection head in float.

### Dynamic / weight-only

```python
from flashdet.quantization import quantize_dynamic, quantize_weight_only

qmodel = quantize_dynamic(model, layer_prefixes=("backbone", "neck"))
qmodel = quantize_weight_only(model, layer_prefixes=("backbone", "neck", "head"))
```

### AWQ

```python
from flashdet.quantization import quantize_awq

qmodel = quantize_awq(
    model,
    calib_loader=calib_loader,
    bits=4,
    group_size=128,
    alpha=0.5,
    layer_prefixes=("backbone", "neck"),
    num_batches=32,
)
```

AWQ computes per-input-channel activation scales, folds them into the weights,
then applies group-wise weight-only quantization. The wrapper divides incoming
activations by the same scale.

### GPTQ

```python
from flashdet.quantization import quantize_gptq

qmodel = quantize_gptq(
    model,
    calib_loader=calib_loader,
    bits=8,
    group_size=-1,       # -1 = per-output-channel
    damp=0.01,
    layer_prefixes=("backbone", "neck"),
    num_batches=32,
)
```

GPTQ collects an input Hessian per layer and quantizes column-by-column with
error compensation. It supports Linear and regular Conv2d (`groups=1`);
depthwise Conv2d is left for AWQ/weight-only quantization.

### Static calibration

```python
from flashdet.quantization import quantize_static

qmodel = quantize_static(
    model,
    calib_loader=calib_loader,       # yields images or (images, targets)
    layer_prefixes=("backbone", "neck"),
    keep_float_prefixes=("head",),
    observer="percentile",           # minmax / percentile / histogram / mse / entropy
    per_channel=True,
    num_batches=32,
)
```

Static quantization in this package is **calibration-based INT8 fake quant**:
weights are rounded to INT8 and activations are fake-quantized at runtime, so it
works with FlashDet's custom multi-output head. For hardware INT8, export to
ONNX/TensorRT/OpenVINO or use `quantize_static_ao` with a graph that has
QuantStub/DeQuantStub boundaries.

### SmoothQuant

```python
from flashdet.quantization import compute_smooth_scale, apply_smooth_scale

scale = compute_smooth_scale(conv.weight, activation_max)
apply_smooth_scale(conv, scale)
# scale the matching activation by 1 / scale before the layer
```

### QAT

```python
from flashdet.quantization import prepare_qat, convert_qat

qat_model = prepare_qat(model, layer_prefixes=("backbone", "neck"))
# train qat_model normally for a few epochs
final_model = convert_qat(qat_model)          # trained fake-quant model
# or if your graph supports eager boundaries:
final_model = convert_qat(qat_model, example_input=sample)
```

## Class-based interfaces

```python
from flashdet.quantization import (
    PTQQuantizer, AWQQuantizer, GPTQQuantizer, QATQuantizer,
)

qmodel = PTQQuantizer(model).quantize(calib_loader)
qmodel = AWQQuantizer(model, bits=4).quantize(calib_loader)
qmodel = GPTQQuantizer(model, bits=8).quantize(calib_loader)

qat = QATQuantizer(model)
qat.prepare(layer_prefixes=("backbone", "neck"))
# train qat.model for a few epochs
qmodel = qat.convert()
```

## Notes

- `head` is kept float by default because classification/regression output
  layers are highly sensitive to INT8.
- Dynamic Linear uses PyTorch's native dynamic kernel. Conv2d uses an INT8
  weight-only fallback, since portable dynamic Conv2d kernels are not available.
- Always recalibrate on data from the deployment domain.
- Quantization is not a substitute for training: a poorly trained checkpoint
  will remain poor after quantization.
