"""Tests for new FlashDet architectures: DETR, RT-DETR, YOLOv9/v10/v11, GroundingDINO."""

import pytest
import torch
import numpy as np


class TestDETR:
    def test_forward_inference(self):
        from flashdet.models.architectures.detr import DETR

        model = DETR(num_classes=10, num_queries=20, d_model=64, nhead=4,
                     num_encoder_layers=1, num_decoder_layers=1,
                     dim_feedforward=128, backbone="resnet18",
                     pretrained_backbone=False)
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert "preds" in out
        assert out["preds"]["logits"].shape == (1, 20, 11)
        assert out["preds"]["boxes"].shape == (1, 20, 4)

    def test_forward_training(self):
        from flashdet.models.architectures.detr import DETR

        model = DETR(num_classes=5, num_queries=10, d_model=64, nhead=4,
                     num_encoder_layers=1, num_decoder_layers=1,
                     dim_feedforward=128, backbone="resnet18",
                     pretrained_backbone=False)
        model.train()
        x = torch.randn(2, 3, 224, 224)
        gt_meta = {
            "img": x,
            "gt_bboxes": [np.array([[10, 10, 50, 50]], dtype=np.float32),
                          np.array([[20, 20, 80, 80]], dtype=np.float32)],
            "gt_labels": [np.array([1], dtype=np.int64),
                          np.array([3], dtype=np.int64)],
        }
        out = model(x, gt_meta=gt_meta)
        assert "loss" in out
        assert out["loss"].requires_grad

    def test_predict(self):
        from flashdet.models.architectures.detr import DETR

        model = DETR(num_classes=5, num_queries=10, d_model=64, nhead=4,
                     num_encoder_layers=1, num_decoder_layers=1,
                     dim_feedforward=128, backbone="resnet18",
                     pretrained_backbone=False)
        x = torch.randn(1, 3, 224, 224)
        results = model.predict(x, score_thr=0.0)
        assert len(results) == 1
        assert "boxes" in results[0]

    def test_registry(self):
        from flashdet.registry import BACKBONES
        assert "DETR" in BACKBONES

    def test_model_info(self):
        from flashdet.models.architectures.detr import DETR
        model = DETR(num_classes=5, num_queries=10, d_model=64, nhead=4,
                     num_encoder_layers=1, num_decoder_layers=1,
                     dim_feedforward=128, backbone="resnet18",
                     pretrained_backbone=False)
        info = model.get_model_info()
        assert info["name"] == "DETR"
        assert info["total_params"] > 0


class TestRTDETR:
    def test_forward_inference(self):
        from flashdet.models.architectures.rt_detr import RTDETR

        model = RTDETR(num_classes=10, backbone="resnet18", hidden_dim=64,
                       nhead=4, num_encoder_layers=1, num_decoder_layers=1,
                       dim_feedforward=128, num_queries=20, num_csp_blocks=1,
                       pretrained_backbone=False)
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert "preds" in out
        assert out["preds"]["logits"].shape[0] == 1

    def test_forward_training(self):
        from flashdet.models.architectures.rt_detr import RTDETR

        model = RTDETR(num_classes=5, backbone="resnet18", hidden_dim=64,
                       nhead=4, num_encoder_layers=1, num_decoder_layers=1,
                       dim_feedforward=128, num_queries=20, num_csp_blocks=1,
                       pretrained_backbone=False)
        model.train()
        x = torch.randn(2, 3, 224, 224)
        gt_meta = {
            "img": x,
            "gt_bboxes": [np.array([[10, 10, 50, 50]], dtype=np.float32),
                          np.array([[20, 20, 80, 80]], dtype=np.float32)],
            "gt_labels": [np.array([1], dtype=np.int64),
                          np.array([3], dtype=np.int64)],
        }
        out = model(x, gt_meta=gt_meta)
        assert "loss" in out
        assert out["loss"].requires_grad

    def test_registry(self):
        from flashdet.registry import BACKBONES
        assert "RTDETR" in BACKBONES


class TestYOLOv9:
    def test_forward(self):
        from flashdet.models.architectures.yolov9 import YOLOv9

        model = YOLOv9(num_classes=10, width_mult=0.25, depth_mult=0.34, use_pgi=True)
        model.eval()
        x = torch.randn(1, 3, 320, 320)
        with torch.no_grad():
            out = model(x)
        assert "preds" in out
        assert len(out["preds"]) == 3

    def test_pgi_training(self):
        from flashdet.models.architectures.yolov9 import YOLOv9

        model = YOLOv9(num_classes=5, width_mult=0.25, depth_mult=0.34, use_pgi=True)
        model.train()
        x = torch.randn(1, 3, 320, 320)
        out = model(x)
        assert "aux_preds" in out
        assert len(out["aux_preds"]) == 3

    def test_registry(self):
        from flashdet.registry import BACKBONES
        assert "YOLOv9" in BACKBONES


class TestYOLOv10:
    def test_forward(self):
        from flashdet.models.architectures.yolov10 import YOLOv10

        model = YOLOv10(num_classes=10, width_mult=0.25, depth_mult=0.34, use_psa=False)
        model.eval()
        x = torch.randn(1, 3, 320, 320)
        with torch.no_grad():
            out = model(x)
        assert "preds" in out
        assert len(out["preds"]) == 3

    def test_dual_heads_training(self):
        from flashdet.models.architectures.yolov10 import YOLOv10

        model = YOLOv10(num_classes=5, width_mult=0.25, depth_mult=0.34, use_psa=False)
        model.train()
        x = torch.randn(1, 3, 320, 320)
        out = model(x)
        assert "o2m_preds" in out

    def test_registry(self):
        from flashdet.registry import BACKBONES
        assert "YOLOv10" in BACKBONES


class TestYOLOv11:
    def test_forward(self):
        from flashdet.models.architectures.yolov11 import YOLOv11

        model = YOLOv11(num_classes=10, width_mult=0.25, depth_mult=0.34, use_c2psa=True)
        model.eval()
        x = torch.randn(1, 3, 320, 320)
        with torch.no_grad():
            out = model(x)
        assert "preds" in out
        assert len(out["preds"]) == 3

    def test_model_info(self):
        from flashdet.models.architectures.yolov11 import YOLOv11

        model = YOLOv11(num_classes=10, width_mult=0.25, depth_mult=0.34, use_c2psa=False)
        info = model.get_model_info()
        assert info["name"] == "YOLOv11"
        assert info["total_params"] > 0

    def test_registry(self):
        from flashdet.registry import BACKBONES
        assert "YOLOv11" in BACKBONES


class TestGroundingDINO:
    def test_forward_inference(self):
        from flashdet.models.architectures.grounding_dino import GroundingDINO

        model = GroundingDINO(
            num_queries=20, d_model=64, nhead=4,
            num_encoder_layers=1, num_decoder_layers=1,
            backbone="resnet50", pretrained_backbone=False,
            text_encoder_depth=1,
        )
        model.eval()
        images = torch.randn(1, 3, 224, 224)
        input_ids = torch.randint(0, 1000, (1, 10))
        attention_mask = torch.ones(1, 10, dtype=torch.long)
        with torch.no_grad():
            out = model(images, input_ids, attention_mask)
        assert "preds" in out
        assert out["preds"]["pred_boxes"].shape == (1, 20, 4)

    def test_registry(self):
        from flashdet.registry import BACKBONES
        assert "GroundingDINO" in BACKBONES
