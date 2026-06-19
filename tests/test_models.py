"""Unit tests for FlashDet models."""

import torch
import pytest
from flashdet.cfg import get_config
from flashdet.models import build_model, FlashDet, apply_lora


@pytest.mark.parametrize("model_size", ["m-0.5x", "m", "m-1.5x"])
def test_model_forward(model_size):
    cfg = get_config(model_size=model_size, input_size=320, num_classes=10)
    model = build_model(cfg)
    model.eval()
    x = torch.randn(1, 3, 320, 320)
    with torch.no_grad():
        out = model(x)
    assert out is not None


@pytest.mark.parametrize("input_size", [320, 416])
def test_input_sizes(input_size):
    cfg = get_config(model_size="m", input_size=input_size, num_classes=5)
    model = build_model(cfg)
    model.eval()
    x = torch.randn(1, 3, input_size, input_size)
    with torch.no_grad():
        out = model(x)
    assert out is not None


def test_lora_reduces_trainable_params():
    cfg = get_config(model_size="m", input_size=320, num_classes=10)
    model = build_model(cfg)
    total_before = sum(p.numel() for p in model.parameters() if p.requires_grad)

    apply_lora(model, rank=4, alpha=8.0)
    total_after = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert total_after < total_before


def test_model_relative_sizes():
    """Larger model variants should have more parameters."""
    params = {}
    for size in ["m-0.5x", "m", "m-1.5x"]:
        cfg = get_config(model_size=size, input_size=320, num_classes=10)
        model = build_model(cfg)
        params[size] = sum(p.numel() for p in model.parameters())

    assert params["m-0.5x"] < params["m"] < params["m-1.5x"]
