"""Post-training INT8 quantization of an exported ONNX model.

Uses onnxruntime's ``quantize_static`` with a calibration dataset stored as
COCO-format images or an ``.npz`` with a ``nchw_f32`` key, so the exported
INT8 graph reflects real algorithm data instead of random noise.
"""

from __future__ import annotations

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


class ArrayCalibReader:
    """Calibration reader for pre-processed NCHW float32 arrays."""

    def __init__(self, data: np.ndarray, batch_size: int = 8):
        self._data = np.ascontiguousarray(data, dtype=np.float32)
        self._index = 0
        self._batch = max(int(batch_size), 1)

    def get_next(self) -> dict | None:
        if self._index >= self._data.shape[0]:
            return None
        end = min(self._index + self._batch, self._data.shape[0])
        batch = self._data[self._index:end]
        self._index = end
        return {"images": batch}


class ImageCalibReader(ArrayCalibReader):
    """Calibration reader over a folder of raw images.

    Applies the exact inference preprocessing (centered letterbox via
    :class:`InferenceTransform`), so calibration statistics match what the
    quantized model will see at inference time.
    """

    def __init__(self, image_dir: str, input_size: tuple[int, int],
                 batch_size: int = 8,
                 max_batches: int | None = None):
        self._batches: list[np.ndarray] = []
        self._input_size = tuple(int(v) for v in input_size)
        self._batch = max(int(batch_size), 1)
        self._max_batches = int(max_batches) if max_batches else None
        self._index = 0
        batches = self._load(image_dir)
        super().__init__(np.concatenate(batches, axis=0), batch_size)

    def _load(self, image_dir: str) -> list[np.ndarray]:
        import cv2

        from flashdet.data.transforms import InferenceTransform

        if not os.path.isdir(image_dir):
            raise FileNotFoundError(image_dir)
        names = sorted(
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        )
        if not names:
            raise ValueError(f"No images found in {image_dir!r}")
        transform = InferenceTransform(self._input_size)
        batches, buf = [], []
        for name in names:
            image = cv2.imread(os.path.join(image_dir, name), cv2.IMREAD_COLOR)
            if image is None:
                continue
            array, _meta = transform(image[:, :, ::-1])
            buf.append(array[None])
            if len(buf) == self._batch:
                batches.append(np.concatenate(buf, axis=0))
                buf = []
                if self._max_batches is not None and len(batches) >= self._max_batches:
                    break
        if buf:
            batches.append(np.concatenate(buf, axis=0))
        if not batches:
            raise ValueError("Calibration dataset has no readable images")
        return batches


def onnx_quantize_int8(
    onnx_path: str,
    output_path: str | None = None,
    calib_images: str | None = None,
    calib_npz: str | None = None,
    batch: np.ndarray | None = None,
    input_size: tuple[int, int] | None = None,
    per_channel: bool = True,
    min_calib_images: int = 16,
    max_batches: int | None = None,
) -> str:
    """Quantize a FlashDet ONNX graph to INT8 (weights + activations).

    Provide either ``calib_images`` (folder), an ``.npz`` file with an
    ``nchw_f32`` array, or a ready NCHW ``batch``.
    """
    try:
        from onnxruntime.quantization import (
            CalibrationMethod,
            QuantFormat,
            QuantType,
            quantize_static,
        )
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "onnxruntime is required for INT8 export: pip install onnxruntime"
        ) from exc

    import onnx

    model = onnx.load(onnx_path)
    graph_input = model.graph.input[0]
    dims = [d.dim_value for d in graph_input.type.tensor_type.shape.dim]
    if len(dims) >= 4 and dims[2] > 0 and input_size is None:
        input_size = (int(dims[2]), int(dims[3]))

    if batch is not None:
        reader = ArrayCalibReader(batch)
    elif calib_npz is not None:
        data = np.load(calib_npz)
        if "nchw_f32" in data:
            data = data["nchw_f32"]
        else:
            first = next(iter(data.keys()))
            data = data[first]
        reader = ArrayCalibReader(data)
    elif calib_images is not None:
        if input_size is None:
            raise ValueError("input_size is required with calib_images")
        reader = ImageCalibReader(
            calib_images, input_size, max_batches=max_batches
        )
        logger.info(
            "Calibrating over %d images", reader._data.shape[0]
        )
    else:
        raise ValueError("Provide calib_images, calib_npz or batch")

    if reader._data.shape[0] < min_calib_images and calib_images is not None:
        logger.warning(
            "Only %d calibration images — static quantization is least "
            "reliable with small calibration sets", reader._data.shape[0]
        )

    output_path = output_path or os.path.splitext(onnx_path)[0] + "_int8.onnx"
    quantize_static(
        onnx_path,
        output_path,
        reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        calibrate_method=CalibrationMethod.MinMax,
    )
    logger.info("Wrote INT8 model: %s", output_path)
    return output_path
