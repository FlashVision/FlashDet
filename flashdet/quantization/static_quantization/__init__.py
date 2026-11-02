"""Static post-training quantization algorithms."""

from flashdet.quantization.base import (
    EntropyObserver,
    HistogramObserver,
    MSEObserver,
    MinMaxObserver,
    PercentileObserver,
    QuantConfig,
)
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
]
