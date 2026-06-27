"""Tests for LoRA and QLoRA fine-tuning."""

import pytest
import torch

from flashdet.models.detector import FlashDet
from flashdet.models.lora import apply_lora, apply_qlora, merge_lora_weights


class TestLoRA:
    """LoRA adapter tests."""

    def test_lora_reduces_trainable_params(self):
        model = FlashDet(num_classes=5, size="n")
        total_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
        apply_lora(model, rank=4, alpha=8.0)
        total_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert total_after < total_before

    def test_lora_forward_still_works(self):
        model = FlashDet(num_classes=5, size="n")
        apply_lora(model, rank=4, alpha=8.0)
        model.eval()
        x = torch.randn(1, 3, 320, 320)
        with torch.no_grad():
            out = model(x)
        assert "preds" in out

    def test_lora_training_produces_gradients(self):
        model = FlashDet(num_classes=5, size="n")
        apply_lora(model, rank=4, alpha=8.0)
        model.train()
        x = torch.randn(1, 3, 320, 320)
        gt = {
            "gt_bboxes": [np.array([[10, 10, 100, 100]], dtype=np.float32)],
            "gt_labels": [np.array([0], dtype=np.int64)],
        }
        out = model(x, gt_meta=gt, epoch=0)
        out["loss"].backward()
        lora_grads = sum(
            1 for n, p in model.named_parameters()
            if "lora" in n and p.grad is not None and p.grad.abs().sum() > 0
        )
        assert lora_grads > 0

    @pytest.mark.parametrize("rank", [2, 4, 8, 16])
    def test_various_ranks(self, rank):
        model = FlashDet(num_classes=5, size="n")
        apply_lora(model, rank=rank, alpha=rank * 2.0)
        model.eval()
        out = model(torch.randn(1, 3, 320, 320))
        assert out is not None

    def test_merge_lora_weights(self):
        model = FlashDet(num_classes=5, size="n")
        apply_lora(model, rank=4, alpha=8.0)
        model.eval()
        x = torch.randn(1, 3, 320, 320)
        with torch.no_grad():
            out_before = model(x)
        merge_lora_weights(model)
        with torch.no_grad():
            out_after = model(x)
        # Outputs should be very close (numerical precision)
        if isinstance(out_before, dict) and "preds" in out_before:
            diff = (out_before["preds"] - out_after["preds"]).abs().max()
            assert diff < 1e-4


class TestQLoRA:
    """QLoRA tests (quantized LoRA)."""

    def test_qlora_applies(self):
        model = FlashDet(num_classes=5, size="n")
        apply_qlora(model, rank=4, alpha=8.0)
        model.eval()
        out = model(torch.randn(1, 3, 320, 320))
        assert out is not None


# Need numpy for gt_meta
import numpy as np
