"""Tests for FLOPsCounter — computational complexity measurement."""

import pytest
import torch

from flashdet.analytics.flops import FLOPsCounter
from flashdet.models.detector import FlashDet


class TestFLOPsCounter:
    """FLOPs/MACs counting tests."""

    @pytest.fixture
    def small_model(self):
        return FlashDet(num_classes=5, size="n")

    def test_returns_total_flops(self, small_model):
        counter = FLOPsCounter(model=small_model, input_size=320, device="cpu")
        result = counter.count()
        assert result["total_flops"] > 0
        assert result["total_macs"] > 0
        assert result["total_params"] > 0

    def test_flops_readable_format(self, small_model):
        counter = FLOPsCounter(model=small_model, input_size=320, device="cpu")
        result = counter.count()
        assert result["flops_readable"].endswith(("K", "M", "G", "T"))
        assert result["macs_readable"].endswith(("K", "M", "G", "T"))

    def test_per_layer_breakdown(self, small_model):
        counter = FLOPsCounter(model=small_model, input_size=320, device="cpu")
        result = counter.count()
        assert "per_layer" in result
        assert len(result["per_layer"]) > 0
        for layer in result["per_layer"]:
            assert "name" in layer
            assert "type" in layer
            assert "flops" in layer
            assert "flops_pct" in layer
            assert layer["flops"] >= 0

    def test_per_layer_pct_sums_to_100(self, small_model):
        counter = FLOPsCounter(model=small_model, input_size=320, device="cpu")
        result = counter.count()
        total_pct = sum(l["flops_pct"] for l in result["per_layer"])
        assert abs(total_pct - 100.0) < 1.0  # Allow small rounding

    def test_macs_equals_half_flops(self, small_model):
        counter = FLOPsCounter(model=small_model, input_size=320, device="cpu")
        result = counter.count()
        assert result["total_macs"] == result["total_flops"] // 2

    def test_summary_string(self, small_model):
        counter = FLOPsCounter(model=small_model, input_size=320, device="cpu")
        summary = counter.summary()
        assert "TOTAL" in summary
        assert "FLOPs" in summary

    def test_larger_input_more_flops(self):
        model = FlashDet(num_classes=5, size="n")
        c320 = FLOPsCounter(model=model, input_size=320, device="cpu")
        c640 = FLOPsCounter(model=model, input_size=640, device="cpu")
        r320 = c320.count()
        r640 = c640.count()
        assert r640["total_flops"] > r320["total_flops"]

    def test_requires_model_or_path(self):
        with pytest.raises(ValueError):
            FLOPsCounter(model=None, model_path=None)
