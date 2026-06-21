"""Tests for new FlashDet architectures: DETR, RT-DETR, YOLOv9/v10/v11, GroundingDINO."""

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

    def test_training_empty_gt(self):
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
                          np.array([], dtype=np.float32).reshape(0, 4)],
            "gt_labels": [np.array([1], dtype=np.int64),
                          np.array([], dtype=np.int64)],
        }
        out = model(x, gt_meta=gt_meta)
        assert "loss" in out
        assert not torch.isnan(out["loss"])
        out["loss"].backward()

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
        assert "scores" in results[0]
        assert "labels" in results[0]
        assert results[0]["boxes"].shape[-1] == 4

    def test_predict_batch(self):
        from flashdet.models.architectures.detr import DETR

        model = DETR(num_classes=5, num_queries=10, d_model=64, nhead=4,
                     num_encoder_layers=1, num_decoder_layers=1,
                     dim_feedforward=128, backbone="resnet18",
                     pretrained_backbone=False)
        x = torch.randn(3, 3, 224, 224)
        results = model.predict(x, score_thr=0.0)
        assert len(results) == 3

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
        assert info["trainable_params"] > 0
        assert info["params_mb"] > 0


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
        assert out["preds"]["boxes"].shape[-1] == 4

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

    def test_training_empty_gt(self):
        from flashdet.models.architectures.rt_detr import RTDETR

        model = RTDETR(num_classes=5, backbone="resnet18", hidden_dim=64,
                       nhead=4, num_encoder_layers=1, num_decoder_layers=1,
                       dim_feedforward=128, num_queries=20, num_csp_blocks=1,
                       pretrained_backbone=False)
        model.train()
        x = torch.randn(2, 3, 224, 224)
        gt_meta = {
            "img": x,
            "gt_bboxes": [np.array([], dtype=np.float32).reshape(0, 4),
                          np.array([], dtype=np.float32).reshape(0, 4)],
            "gt_labels": [np.array([], dtype=np.int64),
                          np.array([], dtype=np.int64)],
        }
        out = model(x, gt_meta=gt_meta)
        assert "loss" in out
        assert not torch.isnan(out["loss"])

    def test_predict(self):
        from flashdet.models.architectures.rt_detr import RTDETR

        model = RTDETR(num_classes=5, backbone="resnet18", hidden_dim=64,
                       nhead=4, num_encoder_layers=1, num_decoder_layers=1,
                       dim_feedforward=128, num_queries=20, num_csp_blocks=1,
                       pretrained_backbone=False)
        x = torch.randn(1, 3, 224, 224)
        results = model.predict(x, score_thr=0.0)
        assert len(results) == 1
        assert "boxes" in results[0]
        assert "scores" in results[0]
        assert "labels" in results[0]

    def test_model_info(self):
        from flashdet.models.architectures.rt_detr import RTDETR

        model = RTDETR(num_classes=5, backbone="resnet18", hidden_dim=64,
                       nhead=4, num_encoder_layers=1, num_decoder_layers=1,
                       dim_feedforward=128, num_queries=20, num_csp_blocks=1,
                       pretrained_backbone=False)
        info = model.get_model_info()
        assert info["name"] == "RT-DETR"
        assert info["total_params"] > 0

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

    def test_forward_no_pgi(self):
        from flashdet.models.architectures.yolov9 import YOLOv9

        model = YOLOv9(num_classes=10, width_mult=0.25, depth_mult=0.34, use_pgi=False)
        model.train()
        x = torch.randn(1, 3, 320, 320)
        out = model(x)
        assert "preds" in out
        assert "aux_preds" not in out

    def test_pgi_training(self):
        from flashdet.models.architectures.yolov9 import YOLOv9

        model = YOLOv9(num_classes=5, width_mult=0.25, depth_mult=0.34, use_pgi=True)
        model.train()
        x = torch.randn(1, 3, 320, 320)
        out = model(x)
        assert "aux_preds" in out
        assert len(out["aux_preds"]) == 3

    def test_gradient_flow(self):
        from flashdet.models.architectures.yolov9 import YOLOv9

        model = YOLOv9(num_classes=5, width_mult=0.25, depth_mult=0.34, use_pgi=True)
        model.train()
        x = torch.randn(1, 3, 320, 320, requires_grad=True)
        out = model(x)
        loss = sum(p.sum() for p in out["preds"])
        loss.backward()
        assert x.grad is not None

    def test_model_info(self):
        from flashdet.models.architectures.yolov9 import YOLOv9

        model = YOLOv9(num_classes=10, width_mult=0.25, depth_mult=0.34)
        info = model.get_model_info()
        assert info["name"] == "YOLOv9"
        assert info["total_params"] > 0

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

    def test_forward_with_psa(self):
        from flashdet.models.architectures.yolov10 import YOLOv10

        model = YOLOv10(num_classes=10, width_mult=0.25, depth_mult=0.34, use_psa=True)
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
        assert len(out["o2m_preds"]) == 3

    def test_inference_no_o2m(self):
        from flashdet.models.architectures.yolov10 import YOLOv10

        model = YOLOv10(num_classes=5, width_mult=0.25, depth_mult=0.34, use_psa=False)
        model.eval()
        x = torch.randn(1, 3, 320, 320)
        with torch.no_grad():
            out = model(x)
        assert "o2m_preds" not in out

    def test_model_info(self):
        from flashdet.models.architectures.yolov10 import YOLOv10

        model = YOLOv10(num_classes=10, width_mult=0.25, depth_mult=0.34)
        info = model.get_model_info()
        assert info["name"] == "YOLOv10"
        assert info["total_params"] > 0

    def test_gradient_flow_with_psa(self):
        from flashdet.models.architectures.yolov10 import YOLOv10

        model = YOLOv10(num_classes=5, width_mult=0.25, depth_mult=0.34, use_psa=True)
        model.train()
        x = torch.randn(1, 3, 320, 320, requires_grad=True)
        out = model(x)
        loss = sum(p.sum() for p in out["preds"])
        loss.backward()
        assert x.grad is not None

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

    def test_forward_no_c2psa(self):
        from flashdet.models.architectures.yolov11 import YOLOv11

        model = YOLOv11(num_classes=10, width_mult=0.25, depth_mult=0.34, use_c2psa=False)
        model.eval()
        x = torch.randn(1, 3, 320, 320)
        with torch.no_grad():
            out = model(x)
        assert "preds" in out

    def test_gradient_flow(self):
        from flashdet.models.architectures.yolov11 import YOLOv11

        model = YOLOv11(num_classes=5, width_mult=0.25, depth_mult=0.34, use_c2psa=True)
        model.train()
        x = torch.randn(1, 3, 320, 320, requires_grad=True)
        out = model(x)
        loss = sum(p.sum() for p in out["preds"])
        loss.backward()
        assert x.grad is not None

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
        assert out["preds"]["pred_logits"].shape[:2] == (1, 20)

    def test_forward_no_text(self):
        from flashdet.models.architectures.grounding_dino import GroundingDINO

        model = GroundingDINO(
            num_queries=20, d_model=64, nhead=4,
            num_encoder_layers=1, num_decoder_layers=1,
            backbone="resnet50", pretrained_backbone=False,
            text_encoder_depth=1,
        )
        model.eval()
        images = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(images)
        assert "preds" in out

    def test_forward_training(self):
        from flashdet.models.architectures.grounding_dino import GroundingDINO

        model = GroundingDINO(
            num_queries=20, d_model=64, nhead=4,
            num_encoder_layers=1, num_decoder_layers=1,
            backbone="resnet50", pretrained_backbone=False,
            text_encoder_depth=1,
        )
        model.train()
        images = torch.randn(2, 3, 224, 224)
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones(2, 10, dtype=torch.long)
        gt_meta = {
            "gt_bboxes": [np.array([[10, 10, 50, 50]], dtype=np.float32),
                          np.array([[20, 20, 80, 80]], dtype=np.float32)],
            "gt_labels": [np.array([1], dtype=np.int64),
                          np.array([3], dtype=np.int64)],
        }
        out = model(images, input_ids, attention_mask, gt_meta=gt_meta)
        assert "loss" in out
        assert out["loss"].requires_grad

    def test_predict(self):
        from flashdet.models.architectures.grounding_dino import GroundingDINO

        model = GroundingDINO(
            num_queries=20, d_model=64, nhead=4,
            num_encoder_layers=1, num_decoder_layers=1,
            backbone="resnet50", pretrained_backbone=False,
            text_encoder_depth=1,
        )
        images = torch.randn(1, 3, 224, 224)
        input_ids = torch.randint(0, 1000, (1, 10))
        attention_mask = torch.ones(1, 10, dtype=torch.long)
        results = model.predict(images, input_ids, attention_mask, score_thr=0.0)
        assert len(results) == 1
        assert "boxes" in results[0]
        assert "scores" in results[0]
        assert results[0]["boxes"].shape[-1] == 4

    def test_model_info(self):
        from flashdet.models.architectures.grounding_dino import GroundingDINO

        model = GroundingDINO(
            num_queries=20, d_model=64, nhead=4,
            num_encoder_layers=1, num_decoder_layers=1,
            backbone="resnet50", pretrained_backbone=False,
            text_encoder_depth=1,
        )
        info = model.get_model_info()
        assert info["name"] == "GroundingDINO"
        assert info["total_params"] > 0

    def test_registry(self):
        from flashdet.registry import BACKBONES
        assert "GroundingDINO" in BACKBONES
