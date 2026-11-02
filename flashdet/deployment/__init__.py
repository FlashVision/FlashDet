"""Deployment helpers: ONNX export, parity checks and INT8 quantization."""

from flashdet.deployment.onnx_export import (
    build_model_from_checkpoint,
    export_onnx,
    input_size_from_checkpoint,
)
from flashdet.deployment.parity import check_parity, detection_parity, onnx_input_shape
from flashdet.deployment.onnx_quantize import (
    ArrayCalibReader,
    ImageCalibReader,
    onnx_quantize_int8,
)

__all__ = [
    "build_model_from_checkpoint",
    "export_onnx",
    "input_size_from_checkpoint",
    "check_parity",
    "detection_parity",
    "onnx_input_shape",
    "ArrayCalibReader",
    "ImageCalibReader",
    "onnx_quantize_int8",
]
