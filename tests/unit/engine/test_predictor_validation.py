"""Input-validation tests for the inference predictor."""

import cv2
import numpy as np
import pytest
import torch

from flashdet.data.transforms import InferenceTransform
from flashdet.engine.inference.predictor import Predictor


@pytest.fixture
def predictor():
    return Predictor(model=torch.nn.Identity(), device="cpu", input_size=64)


def test_grayscale_is_converted(predictor):
    tensor, _ = predictor.preprocess(
        np.random.randint(0, 255, (32, 32), dtype=np.uint8)
    )
    assert tensor.shape == (1, 3, 64, 64)


def test_rgba_is_converted(predictor):
    tensor, _ = predictor.preprocess(
        np.random.randint(0, 255, (32, 32, 4), dtype=np.uint8)
    )
    assert tensor.shape == (1, 3, 64, 64)


def test_float_image_is_converted(predictor):
    tensor, _ = predictor.preprocess(
        np.random.rand(32, 32, 3).astype(np.float32) * 255
    )
    assert tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all()


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


def test_preprocess_matches_training_transform(predictor):
    """Inference must use the same RGB + mean/std + letterbox as ValTransform."""
    image = np.random.randint(0, 255, (48, 80, 3), dtype=np.uint8)
    tensor, _ = predictor.preprocess(image)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    reference, _ = InferenceTransform(input_size=(64, 64))(rgb)
    assert torch.allclose(tensor[0].cpu(), torch.from_numpy(reference), atol=1e-6)


def test_boxes_are_remapped_through_centred_letterbox(predictor):
    image = np.zeros((50, 100, 3), dtype=np.uint8)
    _, meta = predictor.preprocess(image)

    # The original image occupies x=0..64, y=16..48 in a 64x64 centred
    # letterbox. Mapping that area back should recover the full image.
    box = np.array([[0.0, 16.0, 64.0, 48.0]], dtype=np.float32)
    mapped = Predictor._boxes_to_original(box, meta)
    assert np.allclose(mapped[0], [0.0, 0.0, 99.0, 49.0], atol=1.0)
