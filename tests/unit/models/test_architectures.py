"""Tests for all supported YOLO architectures — YOLOv8, v9, v10, v11, YOLOX."""

import pytest
import torch

from flashdet.models.detector import build_model
from flashdet.cfg import get_config


ARCHITECTURES = ["flashdet", "yolov8", "yolov9", "yolov10", "yolov11", "yolox"]


class TestArchitectureForward:
    """Verify every architecture produces valid output."""

    @pytest.mark.parametrize("arch", ARCHITECTURES)
    def test_forward_pass(self, arch):
        input_size = 320 if arch == "flashdet" else 640
        cfg = get_config(num_classes=10, input_size=input_size)
        cfg.model.architecture = arch
        if arch in ("yolov8", "yolov9", "yolov10", "yolov11", "yolox"):
            cfg.model.width_mult = 0.25
            cfg.model.depth_mult = 0.33
        model = build_model(cfg, architecture=arch)
        model.eval()
        x = torch.randn(1, 3, input_size, input_size)
        with torch.no_grad():
            out = model(x)
        assert out is not None

    @pytest.mark.parametrize("arch", ARCHITECTURES)
    def test_output_not_empty(self, arch):
        input_size = 320 if arch == "flashdet" else 640
        cfg = get_config(num_classes=5, input_size=input_size)
        cfg.model.architecture = arch
        if arch in ("yolov8", "yolov9", "yolov10", "yolov11", "yolox"):
            cfg.model.width_mult = 0.25
            cfg.model.depth_mult = 0.33
        model = build_model(cfg, architecture=arch)
        model.eval()
        x = torch.randn(2, 3, input_size, input_size)
        with torch.no_grad():
            out = model(x)
        if isinstance(out, dict) and "preds" in out:
            preds = out["preds"]
            if isinstance(preds, torch.Tensor):
                assert preds.shape[0] == 2
            elif isinstance(preds, list):
                # Multi-level outputs (FPN heads) — each tensor should have batch=2
                assert len(preds) > 0
                for p in preds:
                    if isinstance(p, torch.Tensor):
                        assert p.shape[0] == 2
        elif isinstance(out, torch.Tensor):
            assert out.shape[0] == 2
        elif isinstance(out, (list, tuple)):
            assert len(out) > 0

    @pytest.mark.parametrize("arch", ARCHITECTURES)
    def test_no_nan_output(self, arch):
        input_size = 320 if arch == "flashdet" else 640
        cfg = get_config(num_classes=5, input_size=input_size)
        cfg.model.architecture = arch
        if arch in ("yolov8", "yolov9", "yolov10", "yolov11", "yolox"):
            cfg.model.width_mult = 0.25
            cfg.model.depth_mult = 0.33
        model = build_model(cfg, architecture=arch)
        model.eval()
        x = torch.randn(1, 3, input_size, input_size)
        with torch.no_grad():
            out = model(x)
        if isinstance(out, dict):
            for v in out.values():
                if isinstance(v, torch.Tensor):
                    assert not torch.isnan(v).any(), f"NaN in {arch} output"
        elif isinstance(out, torch.Tensor):
            assert not torch.isnan(out).any()
        elif isinstance(out, (list, tuple)):
            for item in out:
                if isinstance(item, torch.Tensor):
                    assert not torch.isnan(item).any()


class TestArchitectureParams:
    """Verify architectures have reasonable parameter counts."""

    @pytest.mark.parametrize("arch", ARCHITECTURES)
    def test_param_count_nonzero(self, arch):
        input_size = 320 if arch == "flashdet" else 640
        cfg = get_config(num_classes=10, input_size=input_size)
        cfg.model.architecture = arch
        if arch in ("yolov8", "yolov9", "yolov10", "yolov11", "yolox"):
            cfg.model.width_mult = 0.25
            cfg.model.depth_mult = 0.33
        model = build_model(cfg, architecture=arch)
        params = sum(p.numel() for p in model.parameters())
        assert params > 10000, f"{arch} has too few params: {params}"
        assert params < 200_000_000, f"{arch} has too many params: {params}"
