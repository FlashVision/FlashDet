"""FlashDet quantization toolkit.

Layout::

    quantization/
    ├── base.py                       # observers, fake quant, qparams
    ├── static_quantization/          # MinMax/Percentile/Histogram/MSE + SmoothQuant
    ├── dynamic_quantization/         # dynamic Linear + weight-only INT8 Conv
    ├── training_aware_quant/         # fake quant, LSQ, PACT, prepare/convert
    └── quantflashdet/                # layer-aware FlashDet quantizer

Example::

    from flashdet.quantization import QuantFlashDet
    quantizer = QuantFlashDet(model)
    qmodel = quantizer.quantize("static", calib_loader=loader)
"""

from .base import (
    BaseObserver,
    EntropyObserver,
    FakeQuantize,
    HistogramObserver,
    LSQFakeQuantize,
    MSEObserver,
    MinMaxObserver,
    OBSERVERS,
    PACTActivation,
    PercentileObserver,
    QuantConfig,
    build_observer,
    dequantize_tensor,
    fake_quantize,
    quant_error,
    quantize_tensor,
)
from .dynamic_quantization import (
    Int8WeightOnlyConv2d,
    Int8WeightOnlyLinear,
    quantize_dynamic,
    quantize_dynamic_int8,
    quantize_weight_only,
)
from .quantflashdet import (
    DEFAULT_POLICY,
    QuantFlashDet,
    QuantLayerPolicy,
    quantization_report,
)
from .static_quantization import (
    AWQConfig,
    AWQQuantizer,
    GPTQQuantizer,
    PTQQuantizer,
    AWQConv2d,
    AWQLinear,
    GPTQConfig,
    GPTQConv2d,
    GPTQLinear,
    apply_smooth_scale,
    awq_quantize_weight,
    calibrate,
    compute_smooth_scale,
    gptq_quantize_weight,
    quantize_awq,
    quantize_gptq,
    quantize_static,
    quantize_static_ao,
    quantize_static_layers,
    set_layer_qconfig,
    smoothquant_layer,
)
from .training_aware_quant import (
    QATQuantizer,
    convert_qat,
    prepare_qat,
    qat_finetune,
)

__all__ = [
    # base
    "QuantConfig", "BaseObserver", "MinMaxObserver", "PercentileObserver",
    "HistogramObserver", "MSEObserver", "EntropyObserver", "OBSERVERS",
    "build_observer", "quantize_tensor", "dequantize_tensor", "fake_quantize",
    "FakeQuantize", "LSQFakeQuantize", "PACTActivation", "quant_error",
    # static
    "calibrate", "quantize_static", "quantize_static_ao", "quantize_static_layers",
    "set_layer_qconfig", "compute_smooth_scale", "apply_smooth_scale",
    "smoothquant_layer", "quantize_awq", "awq_quantize_weight",
    "AWQConfig", "AWQLinear", "AWQConv2d",
    "quantize_gptq", "gptq_quantize_weight",
    "GPTQConfig", "GPTQLinear", "GPTQConv2d",
    "PTQQuantizer", "AWQQuantizer", "GPTQQuantizer",
    # dynamic
    "quantize_dynamic", "quantize_dynamic_int8", "quantize_weight_only",
    "Int8WeightOnlyConv2d", "Int8WeightOnlyLinear",
    # QAT
    "prepare_qat", "convert_qat", "qat_finetune", "QATQuantizer",
    # FlashDet layer-aware
    "QuantFlashDet", "QuantLayerPolicy", "DEFAULT_POLICY", "quantization_report",
]
