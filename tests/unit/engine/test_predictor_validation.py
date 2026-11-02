"""Input-validation tests for the inference predictor."""

import numpy as np
import pytest
import torch

from flashdet.engine.inference.predictor import Predictor


@pytest.fixture
def predictor():
    return Predictor(model=torch.nn.Identity(), device="cpu", input_size=64)


def test_grayscale_is_converted(predictor):
    tensor, _, _ = predictor.preprocess(
        np.random.randint(0, 255, (32, 32), dtype=np.uint8)
    )
    assert tensor.shape == (1, 3, 64, 64)


def test_rgba_is_converted(predictor):
    tensor, _, _ = predictor.preprocess(
        np.random.randint(0, 255, (32, 32, 4), dtype=np.uint8)
    )
    assert tensor.shape == (1, 3, 64, 64)


def test_float_image_is_converted(predictor):
    tensor, _, _ = predictor.preprocess(
        np.random.rand(32, 32, 3).astype(np.float32) * 255
    )
    assert tensor.dtype == torch.float32
    assert tensor.max() <= 1.0


def test_invalid_shape_is_rejected(predictor):
    with pytest.raises(ValueError, match="HxWx3"):
        predictor.preprocess(np.zeros((10,), dtype=np.uint8))


def test_non_array_is_rejected(predictor):
    with pytest.raises(TypeError, match="numpy image array"):
        predictor.preprocess([[1, 2, 3]])


def test_oversized_image_is_rejected(predictor):
    predictor.MAX_INPUT_PIXELS = 16
    with pytest.raises(ValueError, match="too large"):
        predictor.preprocess(np.zeros((8, 8, 3), dtype=np.uint8))


def test_zero_size_image_is_rejected(predictor):
    with pytest.raises(ValueError, match="invalid dimensions"):
        predictor.preprocess(np.zeros((0, 8, 3), dtype=np.uint8))
