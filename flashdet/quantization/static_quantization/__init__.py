"""Static post-training quantization algorithms."""

from flashdet.quantization.base import (
    EntropyObserver,
    HistogramObserver,
    MSEObserver,
    MinMaxObserver,
    PercentileObserver,
    QuantConfig,
)
from .awq import AWQConfig, AWQConv2d, AWQLinear, awq_quantize_weight, quantize_awq
from .gptq import GPTQConfig, GPTQConv2d, GPTQLinear, gptq_quantize_weight, quantize_gptq
from .quantizers import AWQQuantizer, GPTQQuantizer, PTQQuantizer
from .ptq import (
    calibrate,
    quantize_static,
    quantize_static_ao,
    quantize_static_layers,
    set_layer_qconfig,
)
from .smoothquant import (
    apply_smooth_scale,
    compute_smooth_scale,
    smoothquant_layer,
)

__all__ = [
    "QuantConfig",
    "MinMaxObserver",
    "PercentileObserver",
    "HistogramObserver",
    "MSEObserver",
    "EntropyObserver",
    "calibrate",
    "quantize_static",
    "quantize_static_ao",
    "quantize_static_layers",
    "set_layer_qconfig",
    "compute_smooth_scale",
    "apply_smooth_scale",
    "smoothquant_layer",
    "AWQConfig",
    "AWQLinear",
    "AWQConv2d",
    "quantize_awq",
    "awq_quantize_weight",
    "GPTQConfig",
    "GPTQLinear",
    "GPTQConv2d",
    "quantize_gptq",
    "gptq_quantize_weight",
    "PTQQuantizer",
    "AWQQuantizer",
    "GPTQQuantizer",
]
