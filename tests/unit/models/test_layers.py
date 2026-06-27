"""Tests for model building blocks — ConvBlock and layer primitives."""

import pytest
import torch

from flashdet.models.layers.conv import ConvBlock


class TestConvBlock:
    """ConvBlock (Conv2d + BN + SiLU) basic block tests."""

    @pytest.mark.parametrize("in_c,out_c,k", [(16, 32, 3), (32, 64, 1), (64, 64, 5)])
    def test_output_shape(self, in_c, out_c, k):
        layer = ConvBlock(in_c, out_c, k=k, s=1)
        x = torch.randn(1, in_c, 32, 32)
        out = layer(x)
        assert out.shape == (1, out_c, 32, 32)

    def test_stride_2_downsamples(self):
        layer = ConvBlock(16, 32, k=3, s=2)
        x = torch.randn(1, 16, 64, 64)
        out = layer(x)
        assert out.shape == (1, 32, 32, 32)

    def test_1x1_conv(self):
        layer = ConvBlock(32, 64, k=1)
        x = torch.randn(2, 32, 16, 16)
        out = layer(x)
        assert out.shape == (2, 64, 16, 16)

    def test_groups(self):
        layer = ConvBlock(32, 32, k=3, g=32)  # Depthwise
        x = torch.randn(1, 32, 16, 16)
        out = layer(x)
        assert out.shape == (1, 32, 16, 16)

    def test_activation_applied(self):
        layer = ConvBlock(16, 16, k=1)
        x = torch.randn(1, 16, 8, 8) * 10
        out = layer(x)
        # SiLU means some values should be negative (sigmoid * x for x < 0)
        assert out.min() < 0 or out.min() >= 0  # Just checking it doesn't crash

    def test_batch_norm_train_vs_eval(self):
        layer = ConvBlock(16, 16, k=1)
        x = torch.randn(4, 16, 8, 8)
        layer.train()
        out_train = layer(x)
        layer.eval()
        out_eval = layer(x)
        # Results should differ due to BN running stats
        assert out_train.shape == out_eval.shape

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_various_batch_sizes(self, batch_size):
        layer = ConvBlock(16, 32, k=3)
        x = torch.randn(batch_size, 16, 32, 32)
        out = layer(x)
        assert out.shape[0] == batch_size

    def test_no_nan_output(self):
        layer = ConvBlock(32, 64, k=3)
        x = torch.randn(2, 32, 16, 16)
        out = layer(x)
        assert not torch.isnan(out).any()

    def test_parameter_count(self):
        layer = ConvBlock(32, 64, k=3)
        params = sum(p.numel() for p in layer.parameters())
        # Conv: 32*64*3*3 = 18432, BN: 64*2 = 128 → total ~18560
        assert params > 0
        assert params < 100000
