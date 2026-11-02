"""Tests for training performance helpers."""

import torch

from flashdet.utils.performance import (
    configure_performance,
    make_adamw,
    make_sgd,
    prepare_images,
    to_channels_last,
)


def test_configure_performance_cpu_is_safe():
    configure_performance(torch.device("cpu"))
    assert torch.get_float32_matmul_precision() in ("highest", "high", "medium")


def test_channels_last_conversion_and_input_prep():
    model = torch.nn.Conv2d(3, 4, 3, padding=1)
    model = to_channels_last(model)
    x = prepare_images(torch.randn(1, 3, 16, 16), torch.device("cpu"), True)
    assert x.is_contiguous(memory_format=torch.channels_last)
    assert model(x).shape == (1, 4, 16, 16)


def test_optimizer_helpers_step():
    params = [torch.nn.Parameter(torch.randn(4, 4))]
    for optimizer in (
        make_adamw(params, lr=1e-3, weight_decay=0.0),
        make_sgd(params, lr=1e-3, momentum=0.9, weight_decay=0.0),
    ):
        loss = params[0].pow(2).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
