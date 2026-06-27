"""Integration tests — end-to-end training pipeline."""

import os
import tempfile
import shutil

import numpy as np
import pytest
import torch

from flashdet.models.detector import FlashDet
from flashdet.models.lora import apply_lora


class TestTrainingPipeline:
    """Full training loop simulation (few steps, tiny model)."""

    @pytest.mark.slow
    def test_overfit_single_batch(self):
        """Model should reduce loss on a fixed batch over 20 steps."""
        torch.manual_seed(42)
        np.random.seed(42)

        model = FlashDet(num_classes=3, size="n")
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        x = torch.randn(2, 3, 320, 320)
        gt = {
            "gt_bboxes": [
                np.array([[50, 50, 150, 150], [200, 200, 280, 280]], dtype=np.float32),
                np.array([[30, 30, 120, 120]], dtype=np.float32),
            ],
            "gt_labels": [
                np.array([0, 1], dtype=np.int64),
                np.array([2], dtype=np.int64),
            ],
        }

        losses = []
        for step in range(20):
            optimizer.zero_grad()
            out = model(x, gt_meta=gt, epoch=0)
            out["loss"].backward()
            optimizer.step()
            losses.append(out["loss"].item())

        # Loss should decrease
        assert losses[-1] < losses[0], f"Loss didn't decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
        # No NaN
        assert all(not np.isnan(l) for l in losses)

    @pytest.mark.slow
    def test_lora_finetuning_pipeline(self):
        """LoRA fine-tuning should also reduce loss."""
        model = FlashDet(num_classes=3, size="n")
        apply_lora(model, rank=4, alpha=8.0)
        model.train()

        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=1e-3)

        x = torch.randn(1, 3, 320, 320)
        gt = {
            "gt_bboxes": [np.array([[50, 50, 150, 150]], dtype=np.float32)],
            "gt_labels": [np.array([0], dtype=np.int64)],
        }

        losses = []
        for _ in range(10):
            optimizer.zero_grad()
            out = model(x, gt_meta=gt, epoch=0)
            out["loss"].backward()
            optimizer.step()
            losses.append(out["loss"].item())

        assert all(not np.isnan(l) for l in losses)

    def test_checkpoint_save_load(self, tmp_path):
        """Save and reload a checkpoint."""
        model = FlashDet(num_classes=5, size="n")
        model.train()

        path = str(tmp_path / "ckpt.pth")
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": {"num_classes": 5, "size": "n"},
        }, path)

        # Load back
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model2 = FlashDet(num_classes=5, size="n")
        model2.load_state_dict(ckpt["model_state_dict"])
        model2.eval()
        out = model2(torch.randn(1, 3, 320, 320))
        assert out is not None

    def test_ema_integration(self):
        """EMA should track model without errors."""
        from flashdet.engine.core.ema import ModelEMA

        model = FlashDet(num_classes=5, size="n")
        model.train()
        ema = ModelEMA(model, decay=0.999, warmup=10)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        x = torch.randn(1, 3, 320, 320)
        gt = {
            "gt_bboxes": [np.array([[50, 50, 150, 150]], dtype=np.float32)],
            "gt_labels": [np.array([0], dtype=np.int64)],
        }

        for _ in range(5):
            optimizer.zero_grad()
            out = model(x, gt_meta=gt, epoch=0)
            out["loss"].backward()
            optimizer.step()
            ema.update(model)

        assert ema.num_updates == 5
