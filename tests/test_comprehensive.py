"""Comprehensive test suite for FlashDet."""

import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn

DEVICE = "cpu"
B, C, H, W = 2, 3, 64, 64
NUM_CLASSES = 5


@pytest.fixture
def dummy_input():
    return torch.randn(B, C, H, W)


# ===================================================================
# 1. MODEL ARCHITECTURES
# ===================================================================


class TestFlashDetModel:
    def test_forward_pass(self, dummy_input):
        from flashdet.cfg import get_config
        from flashdet.models import build_model

        cfg = get_config(model_size="m-0.5x", input_size=H, num_classes=NUM_CLASSES)
        model = build_model(cfg)
        model.eval()
        with torch.no_grad():
            out = model(dummy_input)
        assert "preds" in out
        assert out["preds"].dim() == 3

    def test_gradient_flow(self):
        from flashdet.cfg import get_config
        from flashdet.models import build_model

        cfg = get_config(model_size="m-0.5x", input_size=32, num_classes=NUM_CLASSES)
        model = build_model(cfg)
        model.train()
        x = torch.randn(2, 3, 32, 32, requires_grad=True)
        out = model(x)
        out["preds"].sum().backward()
        assert x.grad is not None

    def test_model_info(self):
        from flashdet.cfg import get_config
        from flashdet.models import build_model

        cfg = get_config(model_size="m-0.5x", input_size=32, num_classes=NUM_CLASSES)
        model = build_model(cfg)
        info = model.get_model_info()
        assert info["total_params"] > 0
        assert info["trainable_params"] > 0
        assert "params_mb" in info


class TestDETR:
    def test_forward_eval(self):
        from flashdet.models.architectures.detr import DETR

        model = DETR(
            num_classes=NUM_CLASSES,
            num_queries=10,
            d_model=64,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=128,
            backbone="resnet18",
            pretrained_backbone=False,
        )
        model.eval()
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert "preds" in out
        assert out["preds"]["logits"].shape == (1, 10, NUM_CLASSES + 1)
        assert out["preds"]["boxes"].shape == (1, 10, 4)

    def test_forward_train_with_loss(self):
        from flashdet.models.architectures.detr import DETR

        model = DETR(
            num_classes=NUM_CLASSES,
            num_queries=10,
            d_model=64,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=128,
            backbone="resnet18",
            pretrained_backbone=False,
        )
        model.train()
        x = torch.randn(1, 3, 64, 64)
        gt_meta = {
            "img": x,
            "gt_bboxes": [np.array([[10, 10, 30, 30]], dtype=np.float32)],
            "gt_labels": [np.array([0], dtype=np.int64)],
        }
        out = model(x, gt_meta=gt_meta)
        assert "loss" in out
        out["loss"].backward()

    def test_gradient_flow(self):
        from flashdet.models.architectures.detr import DETR

        model = DETR(
            num_classes=3,
            num_queries=5,
            d_model=32,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=64,
            backbone="resnet18",
            pretrained_backbone=False,
        )
        model.eval()
        x = torch.randn(1, 3, 32, 32, requires_grad=True)
        out = model(x)
        out["preds"]["logits"].sum().backward()
        assert x.grad is not None

    def test_model_info(self):
        from flashdet.models.architectures.detr import DETR

        model = DETR(
            num_classes=5,
            num_queries=5,
            d_model=32,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=64,
            backbone="resnet18",
            pretrained_backbone=False,
        )
        info = model.get_model_info()
        assert info["name"] == "DETR"
        assert info["total_params"] > 0


class TestRTDETR:
    def test_forward_eval(self):
        from flashdet.models.architectures.rt_detr import RTDETR

        model = RTDETR(
            num_classes=NUM_CLASSES,
            backbone="resnet18",
            hidden_dim=64,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=128,
            num_queries=10,
            num_csp_blocks=1,
            pretrained_backbone=False,
        )
        model.eval()
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert "preds" in out
        assert out["preds"]["logits"].shape[0] == 1
        assert out["preds"]["logits"].shape[1] == 10

    def test_model_info(self):
        from flashdet.models.architectures.rt_detr import RTDETR

        model = RTDETR(
            num_classes=5,
            backbone="resnet18",
            hidden_dim=32,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            num_queries=5,
            num_csp_blocks=1,
            pretrained_backbone=False,
        )
        info = model.get_model_info()
        assert info["name"] == "RT-DETR"


class TestYOLOv9:
    def test_forward(self):
        from flashdet.models.architectures.yolov9 import YOLOv9

        model = YOLOv9(num_classes=NUM_CLASSES, width_mult=0.25, depth_mult=0.33, use_pgi=True)
        model.eval()
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert "preds" in out
        assert isinstance(out["preds"], list)
        assert len(out["preds"]) == 3

    def test_pgi_aux_branch(self):
        from flashdet.models.architectures.yolov9 import YOLOv9

        model = YOLOv9(num_classes=NUM_CLASSES, width_mult=0.25, depth_mult=0.33, use_pgi=True)
        model.train()
        x = torch.randn(1, 3, 64, 64)
        out = model(x)
        assert "aux_preds" in out

    def test_gradient_flow(self):
        from flashdet.models.architectures.yolov9 import YOLOv9

        model = YOLOv9(num_classes=3, width_mult=0.25, depth_mult=0.33, use_pgi=False)
        x = torch.randn(2, 3, 64, 64, requires_grad=True)
        out = model(x)
        out["preds"][0].sum().backward()
        assert x.grad is not None


class TestYOLOv10:
    def test_forward(self):
        from flashdet.models.architectures.yolov10 import YOLOv10

        model = YOLOv10(num_classes=NUM_CLASSES, width_mult=0.25, depth_mult=0.33)
        model.eval()
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert "preds" in out
        assert len(out["preds"]) == 3

    def test_dual_heads_training(self):
        from flashdet.models.architectures.yolov10 import YOLOv10

        model = YOLOv10(num_classes=NUM_CLASSES, width_mult=0.25, depth_mult=0.33)
        model.train()
        x = torch.randn(1, 3, 64, 64)
        out = model(x)
        assert "o2m_preds" in out


class TestYOLOv11:
    def test_forward(self):
        from flashdet.models.architectures.yolov11 import YOLOv11

        model = YOLOv11(num_classes=NUM_CLASSES, width_mult=0.25, depth_mult=0.33, use_c2psa=True)
        model.eval()
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert "preds" in out
        assert len(out["preds"]) == 3

    def test_without_c2psa(self):
        from flashdet.models.architectures.yolov11 import YOLOv11

        model = YOLOv11(num_classes=3, width_mult=0.25, depth_mult=0.33, use_c2psa=False)
        model.eval()
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert len(out["preds"]) == 3


class TestGroundingDINO:
    def test_forward(self):
        from flashdet.models.architectures.grounding_dino import GroundingDINO

        model = GroundingDINO(
            num_queries=10,
            d_model=64,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            backbone="resnet50",
            pretrained_backbone=False,
            vocab_size=100,
            max_text_len=10,
            text_encoder_depth=1,
        )
        model.eval()
        x = torch.randn(1, 3, 64, 64)
        input_ids = torch.randint(0, 100, (1, 10))
        attention_mask = torch.ones(1, 10, dtype=torch.long)
        with torch.no_grad():
            out = model(x, input_ids=input_ids, attention_mask=attention_mask)
        assert "preds" in out

    def test_forward_no_text(self):
        from flashdet.models.architectures.grounding_dino import GroundingDINO

        model = GroundingDINO(
            num_queries=10,
            d_model=64,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            backbone="resnet50",
            pretrained_backbone=False,
            vocab_size=100,
            max_text_len=10,
            text_encoder_depth=1,
        )
        model.eval()
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert "preds" in out


# ===================================================================
# 2. LOSSES
# ===================================================================


class TestLosses:
    def test_quality_focal_loss(self):
        from flashdet.losses import QualityFocalLoss

        loss_fn = QualityFocalLoss(beta=2.0)
        pred = torch.randn(100, NUM_CLASSES)
        labels = torch.randint(0, NUM_CLASSES, (100,))
        scores = torch.rand(100)
        loss = loss_fn(pred, (labels, scores))
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_distribution_focal_loss(self):
        from flashdet.losses import DistributionFocalLoss

        loss_fn = DistributionFocalLoss()
        pred = torch.randn(50, 8)
        label = torch.rand(50) * 6.9
        loss = loss_fn(pred, label)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_giou_loss(self):
        from flashdet.losses import GIoULoss

        loss_fn = GIoULoss()
        pred = torch.tensor([[10, 10, 50, 50], [20, 20, 60, 60]], dtype=torch.float32)
        target = torch.tensor([[12, 12, 48, 48], [22, 22, 58, 58]], dtype=torch.float32)
        loss = loss_fn(pred, target)
        assert loss.shape == ()

    def test_iou_loss(self):
        from flashdet.losses import IoULoss

        loss_fn = IoULoss()
        pred = torch.tensor([[10, 10, 50, 50]], dtype=torch.float32)
        target = torch.tensor([[12, 12, 48, 48]], dtype=torch.float32)
        loss = loss_fn(pred, target)
        assert torch.isfinite(loss)

    def test_varifocal_loss(self):
        from flashdet.losses import VarifocalLoss

        loss_fn = VarifocalLoss()
        pred = torch.randn(20, NUM_CLASSES)
        target = torch.zeros(20, NUM_CLASSES)
        target[0, 0] = 0.9
        target[1, 2] = 0.7
        loss = loss_fn(pred, target)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_sigmoid_focal_loss(self):
        from flashdet.losses import SigmoidFocalLoss

        loss_fn = SigmoidFocalLoss()
        pred = torch.randn(20, NUM_CLASSES)
        target = torch.zeros(20, NUM_CLASSES)
        target[0, 0] = 1.0
        loss = loss_fn(pred, target)
        assert torch.isfinite(loss)

    def test_kd_logit_loss(self):
        from flashdet.losses import LogitDistillationLoss

        loss_fn = LogitDistillationLoss()
        s_pred = torch.randn(1, 100, NUM_CLASSES + 32)
        t_pred = torch.randn(1, 100, NUM_CLASSES + 32)
        result = loss_fn(s_pred, t_pred, NUM_CLASSES)
        assert "kd_logit_loss" in result

    def test_kd_feature_loss(self):
        from flashdet.losses import FeatureDistillationLoss

        loss_fn = FeatureDistillationLoss(student_channels=32, teacher_channels=64, num_levels=2)
        s_feats = [torch.randn(1, 32, 8, 8), torch.randn(1, 32, 4, 4)]
        t_feats = [torch.randn(1, 64, 8, 8), torch.randn(1, 64, 4, 4)]
        loss = loss_fn(s_feats, t_feats)
        assert torch.isfinite(loss)

    def test_chunked_qfl(self):
        from flashdet.losses.chunked_loss import chunked_quality_focal_loss

        pred = torch.randn(200, NUM_CLASSES)
        labels = torch.randint(0, NUM_CLASSES, (200,))
        scores = torch.rand(200)
        loss = chunked_quality_focal_loss(pred, (labels, scores), chunk_size=50)
        assert torch.isfinite(loss)

    def test_loss_gradient(self):
        from flashdet.losses import QualityFocalLoss

        loss_fn = QualityFocalLoss()
        pred = torch.randn(10, NUM_CLASSES, requires_grad=True)
        labels = torch.randint(0, NUM_CLASSES, (10,))
        scores = torch.rand(10)
        loss = loss_fn(pred, (labels, scores))
        loss.backward()
        assert pred.grad is not None

    def test_loss_empty_positives(self):
        from flashdet.losses import QualityFocalLoss

        loss_fn = QualityFocalLoss()
        pred = torch.randn(10, NUM_CLASSES)
        labels = torch.full((10,), NUM_CLASSES, dtype=torch.long)
        scores = torch.zeros(10)
        loss = loss_fn(pred, (labels, scores))
        assert torch.isfinite(loss)


# ===================================================================
# 3. REGISTRY
# ===================================================================


class TestRegistry:
    def test_backbone_registry(self):
        from flashdet.registry import BACKBONES

        assert "ShuffleNetV2" in BACKBONES
        assert "DETR" in BACKBONES
        assert "RTDETR" in BACKBONES
        assert "YOLOv9" in BACKBONES
        assert "YOLOv10" in BACKBONES
        assert "YOLOv11" in BACKBONES
        assert "GroundingDINO" in BACKBONES

    def test_head_registry(self):
        from flashdet.registry import HEADS

        assert "FlashDetHead" in HEADS
        assert "SimpleConvHead" in HEADS
        assert "OBBHead" in HEADS

    def test_neck_registry(self):
        from flashdet.registry import NECKS

        assert "GhostPAN" in NECKS

    def test_registry_build(self):
        from flashdet.registry import Registry

        reg = Registry("test")

        @reg.register("TestClass")
        class TestClass:
            def __init__(self, val=1):
                self.val = val

        obj = reg.build("TestClass", val=42)
        assert obj.val == 42
        assert reg.list() == ["TestClass"]
        assert len(reg) == 1

    def test_registry_error(self):
        from flashdet.registry import Registry

        reg = Registry("test")
        with pytest.raises(KeyError):
            reg.build("NonExistent")

    def test_registry_duplicate(self):
        from flashdet.registry import Registry

        reg = Registry("test")

        @reg.register("A")
        class A:
            pass

        with pytest.raises(KeyError):

            @reg.register("A")
            class A2:
                pass


# ===================================================================
# 4. CLI
# ===================================================================


class TestCLI:
    def test_version_command(self):
        result = subprocess.run(
            [sys.executable, "-m", "flashdet.cli", "version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "FlashDet" in result.stdout

    def test_no_command_shows_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "flashdet.cli"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0

    def test_main_import(self):
        from flashdet.cli import main

        assert callable(main)

    def test_cli_functions_exist(self):
        from flashdet.cli import cmd_check, cmd_settings, cmd_version

        assert callable(cmd_version)
        assert callable(cmd_check)
        assert callable(cmd_settings)


# ===================================================================
# 5. ENGINE
# ===================================================================


class TestEngine:
    def test_trainer_import(self):
        from flashdet.engine.trainer import Trainer

        assert Trainer is not None

    def test_validator_import(self):
        from flashdet.engine.validator import Validator

        assert Validator is not None

    def test_predictor_import(self):
        from flashdet.engine.predictor import Predictor

        assert Predictor is not None

    def test_exporter_import(self):
        from flashdet.engine.exporter import Exporter

        assert Exporter is not None

    def test_callbacks_import(self):
        from flashdet.engine.callbacks import CallbackList, EarlyStopping

        cb_list = CallbackList()
        assert cb_list is not None
        assert EarlyStopping is not None


# ===================================================================
# 6. DATA
# ===================================================================


class TestData:
    def test_transforms_import(self):
        from flashdet.data.transforms import InferenceTransform, TrainTransform, ValTransform

        assert TrainTransform is not None
        assert ValTransform is not None
        assert InferenceTransform is not None

    def test_augmentations_import(self):
        from flashdet.data.augmentations import CopyPaste, MixUp, Mosaic

        assert Mosaic is not None
        assert MixUp is not None
        assert CopyPaste is not None

    def test_val_transform(self):
        from flashdet.data.transforms import ValTransform

        t = ValTransform(input_size=(64, 64))
        img = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
        labels = np.array([0], dtype=np.int64)
        result = t(img, boxes, labels)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_train_transform(self):
        from flashdet.data.transforms import TrainTransform

        t = TrainTransform(input_size=(64, 64))
        assert t is not None


# ===================================================================
# 7. UTILS
# ===================================================================


class TestUtils:
    def test_box_utils(self):
        from flashdet.utils.box_utils import bbox2distance, distance2bbox

        center = torch.tensor([[50.0, 50.0]])
        dist = torch.tensor([[10.0, 10.0, 10.0, 10.0]])
        bbox = distance2bbox(center, dist)
        assert bbox.shape == (1, 4)
        assert torch.allclose(bbox, torch.tensor([[40.0, 40.0, 60.0, 60.0]]))

        dist_back = bbox2distance(center, bbox)
        assert dist_back.shape == (1, 4)

    def test_bbox_overlaps(self):
        from flashdet.utils.box_utils import bbox_overlaps

        b1 = torch.tensor([[0, 0, 10, 10], [0, 0, 5, 5]], dtype=torch.float32)
        b2 = torch.tensor([[5, 5, 15, 15]], dtype=torch.float32)
        ious = bbox_overlaps(b1, b2)
        assert ious.shape == (2, 1)
        assert (ious >= 0).all() and (ious <= 1).all()

    def test_metrics_import(self):
        from flashdet.utils.metrics import compute_map

        assert callable(compute_map)

    def test_logger(self):
        from flashdet.utils.logger import setup_logger

        log = setup_logger("test_logger")
        assert log is not None

    def test_visualization_import(self):
        from flashdet.utils.visualization import draw_boxes

        assert callable(draw_boxes)

    def test_checkpoint_utils(self):
        from flashdet.utils.checkpoint import load_checkpoint, save_checkpoint

        assert callable(load_checkpoint)
        assert callable(save_checkpoint)

    def test_nms(self):
        from flashdet.utils.box_utils import multiclass_nms

        assert callable(multiclass_nms)


# ===================================================================
# 8. SOLUTIONS
# ===================================================================


class TestSolutions:
    def test_object_counter_init(self):
        from flashdet.solutions import ObjectCounter

        counter = ObjectCounter.__new__(ObjectCounter)
        assert counter is not None

    def test_speed_estimator_init(self):
        from flashdet.solutions import SpeedEstimator

        est = SpeedEstimator.__new__(SpeedEstimator)
        assert est is not None

    def test_heatmap_init(self):
        from flashdet.solutions import Heatmap

        hm = Heatmap.__new__(Heatmap)
        assert hm is not None

    def test_region_counter_init(self):
        from flashdet.solutions import RegionCounter

        rc = RegionCounter.__new__(RegionCounter)
        assert rc is not None

    def test_solutions_import(self):
        from flashdet.solutions import (
            AnalyticsDashboard,
            SecurityAlarm,
        )

        assert AnalyticsDashboard is not None
        assert SecurityAlarm is not None


# ===================================================================
# 9. TRACKERS
# ===================================================================


class TestTrackers:
    def test_byte_tracker(self):
        from flashdet.trackers import ByteTracker

        tracker = ByteTracker(max_age=30, min_hits=3, iou_threshold=0.3)
        assert tracker is not None

    def test_sort_tracker(self):
        from flashdet.trackers import SORTTracker

        tracker = SORTTracker(max_age=30, min_hits=3)
        assert tracker is not None

    def test_botsort_tracker(self):
        from flashdet.trackers import BoTSORT

        tracker = BoTSORT(max_age=30, reid_weight=0.3)
        assert tracker is not None


# ===================================================================
# 10. OBB HEAD
# ===================================================================


class TestOBBHead:
    def test_forward(self):
        from flashdet.models.head.obb_head import OBBHead

        head = OBBHead(num_classes=5, in_channels=64, feat_channels=64, stacked_convs=1, strides=[8, 16])
        feats = [torch.randn(1, 64, 8, 8), torch.randn(1, 64, 4, 4)]
        out = head(feats)
        assert "cls" in out
        assert "reg" in out
        assert "angle" in out

    def test_rotated_iou(self):
        from flashdet.models.head.obb_head import rotated_iou

        b1 = torch.tensor([[50, 50, 20, 20, 0.0]])
        b2 = torch.tensor([[50, 50, 20, 20, 0.0]])
        iou = rotated_iou(b1, b2)
        assert iou.shape == (1,)
        assert iou[0] > 0.9

    def test_rotated_nms(self):
        from flashdet.models.head.obb_head import rotated_nms

        boxes = torch.tensor([[50, 50, 20, 20, 0.0], [52, 52, 20, 20, 0.0], [200, 200, 20, 20, 0.0]])
        scores = torch.tensor([0.9, 0.8, 0.95])
        keep = rotated_nms(boxes, scores, iou_thr=0.5)
        assert len(keep) >= 2


# ===================================================================
# 11. LoRA
# ===================================================================


class TestLoRA:
    def test_apply_lora(self):
        from flashdet.cfg import get_config
        from flashdet.models import build_model
        from flashdet.models.lora import apply_lora

        cfg = get_config(model_size="m-0.5x", input_size=32, num_classes=5)
        model = build_model(cfg)
        model = apply_lora(model, rank=4, alpha=8.0, variant="standard")
        trainable = sum(1 for p in model.parameters() if p.requires_grad)
        assert trainable > 0

    def test_lora_variants_list(self):
        from flashdet.models.lora import LORA_VARIANTS, get_variant_description

        for v in LORA_VARIANTS:
            desc = get_variant_description(v)
            assert len(desc) > 0

    def test_merge_lora(self):
        from flashdet.models.lora import LoRALinear, merge_lora_weights

        layer = LoRALinear(16, 16, rank=4)
        model = nn.Sequential(layer)
        merged = merge_lora_weights(model)
        assert merged is not None


# ===================================================================
# 12. EDGE CASES
# ===================================================================


class TestEdgeCases:
    def test_empty_gt(self):
        from flashdet.models.architectures.detr import DETR

        model = DETR(
            num_classes=5,
            num_queries=5,
            d_model=32,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=64,
            backbone="resnet18",
            pretrained_backbone=False,
        )
        model.train()
        x = torch.randn(1, 3, 64, 64)
        gt_meta = {
            "img": x,
            "gt_bboxes": [np.array([], dtype=np.float32).reshape(0, 4)],
            "gt_labels": [np.array([], dtype=np.int64)],
        }
        out = model(x, gt_meta=gt_meta)
        assert "loss" in out
        assert torch.isfinite(out["loss"])

    def test_wrong_input_channels(self):
        from flashdet.models.architectures.yolov11 import YOLOv11

        model = YOLOv11(num_classes=3, width_mult=0.25, depth_mult=0.33)
        with pytest.raises(RuntimeError):
            model(torch.randn(1, 1, 64, 64))

    def test_single_sample_batch(self):
        from flashdet.cfg import get_config
        from flashdet.models import build_model

        cfg = get_config(model_size="m-0.5x", input_size=64, num_classes=3)
        model = build_model(cfg)
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(1, 3, 64, 64))
        assert out["preds"].shape[0] == 1


# ===================================================================
# 13. INTEGRATION / PIPELINE
# ===================================================================


class TestIntegration:
    def test_full_pipeline(self):
        from flashdet.cfg import get_config
        from flashdet.models import build_model

        cfg = get_config(model_size="m-0.5x", input_size=64, num_classes=3)
        model = build_model(cfg)
        model.train()

        x = torch.randn(2, 3, 64, 64)
        gt_meta = {
            "img": x,
            "gt_bboxes": [
                np.array([[5, 5, 20, 20]], dtype=np.float32),
                np.array([[10, 10, 25, 25]], dtype=np.float32),
            ],
            "gt_labels": [
                np.array([0], dtype=np.int64),
                np.array([1], dtype=np.int64),
            ],
        }
        out = model(x, gt_meta=gt_meta)
        assert "loss" in out
        loss = out["loss"]
        assert torch.isfinite(loss)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            eval_out = model(x)
        assert "preds" in eval_out

    def test_export_mock(self):
        from flashdet.cfg import get_config
        from flashdet.models import build_model

        cfg = get_config(model_size="m-0.5x", input_size=64, num_classes=3)
        model = build_model(cfg)
        model.eval()
        dummy = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out = model(dummy)
        assert "preds" in out


# ===================================================================
# 14. ANALYTICS
# ===================================================================


class TestAnalytics:
    def test_benchmark_import(self):
        from flashdet.analytics import Benchmark

        assert Benchmark is not None

    def test_profiler_import(self):
        from flashdet.analytics import Profiler

        assert Profiler is not None


# ===================================================================
# 15. CONFIG
# ===================================================================


class TestConfig:
    def test_get_config(self):
        from flashdet.cfg import get_config

        cfg = get_config(model_size="m", input_size=320, num_classes=10)
        assert cfg.model.num_classes == 10
        assert cfg.model.input_size == (320, 320)

    def test_get_config_variants(self):
        from flashdet.cfg import get_config

        for size in ["m-0.5x", "m", "m-1.5x"]:
            cfg = get_config(model_size=size, input_size=64, num_classes=5)
            assert cfg.model.num_classes == 5


# ===================================================================
# 16. ASSIGNMENT
# ===================================================================


class TestAssignment:
    def test_dsl_assigner(self):
        from flashdet.models.assignment import AssignResult, DynamicSoftLabelAssigner

        assigner = DynamicSoftLabelAssigner(topk=5)
        pred_scores = torch.randn(50, 5).sigmoid()
        priors = torch.rand(50, 4) * 100
        priors[:, 2:] = 8.0
        decoded = torch.rand(50, 4) * 100
        decoded[:, 2:] = decoded[:, :2] + 20
        gt_bboxes = torch.tensor([[10, 10, 40, 40], [50, 50, 80, 80]], dtype=torch.float32)
        gt_labels = torch.tensor([0, 1], dtype=torch.long)

        result = assigner.assign(pred_scores, priors, decoded, gt_bboxes, gt_labels)
        assert isinstance(result, AssignResult)
        assert result.gt_inds.shape[0] == 50

    def test_dsl_assigner_no_gt(self):
        from flashdet.models.assignment import DynamicSoftLabelAssigner

        assigner = DynamicSoftLabelAssigner(topk=5)
        pred_scores = torch.randn(50, 5).sigmoid()
        priors = torch.rand(50, 4) * 100
        decoded = torch.rand(50, 4) * 100
        gt_bboxes = torch.zeros(0, 4)
        gt_labels = torch.zeros(0, dtype=torch.long)

        result = assigner.assign(pred_scores, priors, decoded, gt_bboxes, gt_labels)
        assert (result.gt_inds == 0).all()


# ===================================================================
# 17. NECK AND BACKBONE SUBMODULES
# ===================================================================


class TestSubmodules:
    def test_ghost_pan(self):
        from flashdet.models.neck import GhostPAN

        fpn = GhostPAN(in_channels=[48, 96, 192], out_channels=64, num_extra_level=1)
        feats = [torch.randn(1, 48, 16, 16), torch.randn(1, 96, 8, 8), torch.randn(1, 192, 4, 4)]
        outs = fpn(feats)
        assert len(outs) == 4

    def test_ghost_module(self):
        from flashdet.models.neck import GhostModule

        m = GhostModule(32, 64)
        x = torch.randn(1, 32, 8, 8)
        out = m(x)
        assert out.shape[1] >= 64

    def test_shufflenet_backbone(self):
        from flashdet.models.backbone import ShuffleNetV2

        bb = ShuffleNetV2(model_size="0.5x", pretrained=False)
        x = torch.randn(1, 3, 64, 64)
        feats = bb(x)
        assert isinstance(feats, list)
        assert len(feats) == 3

    def test_channel_shuffle(self):
        from flashdet.models.backbone import channel_shuffle

        x = torch.randn(1, 8, 4, 4)
        out = channel_shuffle(x, 2)
        assert out.shape == x.shape


# ===================================================================
# 18. MOCK TensorRT EXPORT
# ===================================================================


class TestTensorRTMock:
    @patch("flashdet.engine.exporter.torch.onnx.export")
    def test_onnx_export_call(self, mock_export):
        from flashdet.cfg import get_config
        from flashdet.models import build_model

        cfg = get_config(model_size="m-0.5x", input_size=64, num_classes=3)
        model = build_model(cfg)
        model.eval()
        dummy = torch.randn(1, 3, 64, 64)
        mock_export.return_value = None
        torch.onnx.export(model, dummy, "/tmp/test.onnx")
        mock_export.assert_called_once()
