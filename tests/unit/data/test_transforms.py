"""Tests for image transforms — train, val, inference."""

import numpy as np
import pytest
import torch

from flashdet.data.transforms import TrainTransform, ValTransform, InferenceTransform


class TestTrainTransform:
    """Training transform pipeline."""

    def test_output_shape(self):
        tf = TrainTransform(input_size=(320, 320))
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        boxes = np.array([[10, 10, 100, 100]], dtype=np.float32)
        labels = np.array([0], dtype=np.int64)
        out_img, out_boxes, out_labels = tf(img, boxes, labels)
        assert out_img.shape == torch.Size([3, 320, 320])

    def test_preserves_label_count(self):
        tf = TrainTransform(input_size=(320, 320))
        img = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
        boxes = np.array([[10, 10, 100, 100], [150, 150, 200, 200]], dtype=np.float32)
        labels = np.array([0, 1], dtype=np.int64)
        _, out_boxes, out_labels = tf(img, boxes, labels)
        assert len(out_labels) <= len(labels)

    def test_output_is_float(self):
        tf = TrainTransform(input_size=(320, 320))
        img = np.ones((320, 320, 3), dtype=np.uint8) * 128
        boxes = np.array([[10, 10, 100, 100]], dtype=np.float32)
        labels = np.array([0], dtype=np.int64)
        out_img, _, _ = tf(img, boxes, labels)
        assert out_img.dtype == torch.float32


class TestValTransform:
    """Validation / test transform."""

    def test_output_shape(self):
        tf = ValTransform(input_size=(320, 320))
        img = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
        boxes = np.array([[10, 10, 100, 100]], dtype=np.float32)
        labels = np.array([0], dtype=np.int64)
        out_img, _, _ = tf(img, boxes, labels)
        assert out_img.shape == torch.Size([3, 320, 320])

    @pytest.mark.parametrize("size", [320, 416, 640])
    def test_various_input_sizes(self, size):
        tf = ValTransform(input_size=(size, size))
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        boxes = np.array([[10, 10, 100, 100]], dtype=np.float32)
        labels = np.array([0], dtype=np.int64)
        out_img, _, _ = tf(img, boxes, labels)
        assert out_img.shape == torch.Size([3, size, size])


class TestInferenceTransform:
    """Inference-time preprocessing."""

    def test_output_is_float32(self):
        tf = InferenceTransform(input_size=(320, 320))
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        out = tf(img)
        if isinstance(out, tuple):
            out = out[0]
        if isinstance(out, torch.Tensor):
            assert out.dtype == torch.float32
        else:
            assert out.dtype == np.float32

    def test_output_channels_first(self):
        tf = InferenceTransform(input_size=(320, 320))
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        out = tf(img)
        if isinstance(out, tuple):
            out = out[0]
        assert out.shape[0] == 3 or out.shape[-3] == 3
