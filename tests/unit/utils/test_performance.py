"""Tests for training performance helpers."""

import torch

from flashdet.utils import create_optimizer
from flashdet.utils.performance import (
    build_param_groups,
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


def test_param_groups_exclude_bias_and_norm():
    model = torch.nn.Sequential(
        torch.nn.BatchNorm2d(3),
        torch.nn.Conv2d(3, 8, 3, padding=1),
        torch.nn.GroupNorm(2, 8),
    )
    groups = build_param_groups(model, weight_decay=0.05)
    assert len(groups) == 2
    decay_group = next(g for g in groups if g["weight_decay"] > 0)
    no_decay_group = next(g for g in groups if g["weight_decay"] == 0)
    assert len(decay_group["params"]) == 1  # conv weight only
    assert len(no_decay_group["params"]) == 5  # bn (affine+bias) + conv bias + gn


def test_param_groups_skips_frozen_params():
    model = torch.nn.Linear(4, 2)
    model.bias.requires_grad_(False)
    groups = build_param_groups(model, weight_decay=0.001)
    all_params = [p for g in groups for p in g["params"]]
    assert len(all_params) == 1


def test_create_optimizer_uses_decay_groups():
    model = torch.nn.Sequential(
        torch.nn.BatchNorm2d(3),
        torch.nn.Conv2d(3, 4, 3, padding=1),
    )
    optimizer = create_optimizer(model, lr=1e-3, weight_decay=0.05)
    decay = {id(p): g["weight_decay"] for g in optimizer.param_groups
             for p in g["params"]}
    conv_weight = id(model[1].weight)
    conv_bias = id(model[1].bias)
    bn_weight = id(model[0].weight)
    assert decay[conv_weight] == 0.05
    assert decay[conv_bias] == 0.0
    assert decay[bn_weight] == 0.0
    # Step must still work with grouped parameters.
    loss = model(torch.randn(1, 3, 8, 8)).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
