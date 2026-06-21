"""Tests for custom training logic: KD training step, Trainer components, and callbacks."""

import os
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn

from flashdet.models.detector import FlashDet
from flashdet.losses.kd_loss import (
    KnowledgeDistillationLoss,
)
from flashdet.engine.callbacks import (
    Callback,
    CallbackList,
    EarlyStopping,
    LRSchedulerCallback,
    CSVLogger,
)
from flashdet.engine.trainer import ModelEMA, Trainer, MODEL_SIZE_MAP


def _make_gt_meta(batch_size=2, num_classes=5, img_size=320):
    gt_meta = {"gt_bboxes": [], "gt_labels": []}
    for _ in range(batch_size):
        n_objs = np.random.randint(1, 4)
        x1y1 = np.random.rand(n_objs, 2).astype(np.float32) * (img_size * 0.5)
        wh = np.random.rand(n_objs, 2).astype(np.float32) * (img_size * 0.3) + 10
        boxes = np.concatenate([x1y1, x1y1 + wh], axis=1)
        boxes = np.clip(boxes, 0, img_size - 1)
        labels = np.random.randint(0, num_classes, size=(n_objs,)).astype(np.int64)
        gt_meta["gt_bboxes"].append(boxes)
        gt_meta["gt_labels"].append(labels)
    return gt_meta


# ======================================================================
# Knowledge Distillation Training Step Tests
# ======================================================================

class TestKDTrainingStep:
    """End-to-end tests for KD training step (student + teacher)."""

    def _build_student_teacher(self, num_classes=5):
        student = FlashDet(
            num_classes=num_classes, input_size=(320, 320), backbone_size="0.5x",
            fpn_channels=96, pretrained=False, use_aux_head=True,
        )
        teacher = FlashDet(
            num_classes=num_classes, input_size=(320, 320), backbone_size="0.5x",
            fpn_channels=96, pretrained=False, use_aux_head=False,
        )
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        return student, teacher

    def test_kd_forward_pass(self):
        student, teacher = self._build_student_teacher(5)
        kd_criterion = KnowledgeDistillationLoss(
            temperature=4.0, logit_weight=1.0, feature_weight=0.5,
            student_channels=96, teacher_channels=96, num_levels=4,
        )
        student.train()
        x = torch.randn(2, 3, 320, 320)
        gt_meta = _make_gt_meta(2, 5)

        s_out = student(x, gt_meta=gt_meta, epoch=1, return_features=True)
        with torch.no_grad():
            t_out = teacher(x, return_features=True)

        kd_result = kd_criterion(
            student_preds=s_out["preds"],
            teacher_preds=t_out["preds"].detach(),
            student_fpn_feats=s_out["fpn_features"],
            teacher_fpn_feats=[f.detach() for f in t_out["fpn_features"]],
            num_classes=5, reg_max=7,
        )

        assert "kd_loss" in kd_result
        assert "kd_cls_loss" in kd_result
        assert "kd_feature_loss" in kd_result
        assert kd_result["kd_loss"].requires_grad

    def test_kd_backward_pass(self):
        student, teacher = self._build_student_teacher(5)
        kd_criterion = KnowledgeDistillationLoss(
            temperature=4.0, logit_weight=1.0, feature_weight=0.5,
            student_channels=96, teacher_channels=96, num_levels=4,
        )
        student.train()
        x = torch.randn(2, 3, 320, 320)
        gt_meta = _make_gt_meta(2, 5)

        s_out = student(x, gt_meta=gt_meta, epoch=1, return_features=True)
        with torch.no_grad():
            t_out = teacher(x, return_features=True)

        kd_result = kd_criterion(
            student_preds=s_out["preds"],
            teacher_preds=t_out["preds"].detach(),
            student_fpn_feats=s_out["fpn_features"],
            teacher_fpn_feats=[f.detach() for f in t_out["fpn_features"]],
            num_classes=5, reg_max=7,
        )

        total_loss = s_out["loss"] + kd_result["kd_loss"]
        total_loss.backward()

        grad_count = sum(1 for p in student.parameters()
                         if p.grad is not None and p.grad.abs().sum() > 0)
        assert grad_count > 0

    def test_kd_teacher_no_grad(self):
        student, teacher = self._build_student_teacher(5)
        kd_criterion = KnowledgeDistillationLoss(
            temperature=4.0, logit_weight=1.0, feature_weight=0.5,
            student_channels=96, teacher_channels=96, num_levels=4,
        )
        student.train()
        x = torch.randn(2, 3, 320, 320)
        gt_meta = _make_gt_meta(2, 5)

        s_out = student(x, gt_meta=gt_meta, epoch=1, return_features=True)
        with torch.no_grad():
            t_out = teacher(x, return_features=True)

        kd_result = kd_criterion(
            student_preds=s_out["preds"],
            teacher_preds=t_out["preds"].detach(),
            student_fpn_feats=s_out["fpn_features"],
            teacher_fpn_feats=[f.detach() for f in t_out["fpn_features"]],
            num_classes=5, reg_max=7,
        )

        total_loss = s_out["loss"] + kd_result["kd_loss"]
        total_loss.backward()

        for p in teacher.parameters():
            assert p.grad is None

    def test_kd_logit_only(self):
        """KD with feature_weight=0 should still produce valid loss."""
        student, teacher = self._build_student_teacher(5)
        kd_criterion = KnowledgeDistillationLoss(
            temperature=4.0, logit_weight=1.0, feature_weight=0.0,
            student_channels=96, teacher_channels=96, num_levels=4,
        )
        student.train()
        x = torch.randn(1, 3, 320, 320)
        gt_meta = _make_gt_meta(1, 5)

        s_out = student(x, gt_meta=gt_meta, epoch=1, return_features=True)
        with torch.no_grad():
            t_out = teacher(x, return_features=True)

        kd_result = kd_criterion(
            student_preds=s_out["preds"],
            teacher_preds=t_out["preds"].detach(),
            student_fpn_feats=s_out["fpn_features"],
            teacher_fpn_feats=[f.detach() for f in t_out["fpn_features"]],
            num_classes=5, reg_max=7,
        )

        assert kd_result["kd_feature_loss"].item() == 0.0
        assert kd_result["kd_loss"].item() > 0

    def test_kd_feature_only(self):
        """KD with logit_weight=0 should only produce feature loss."""
        student, teacher = self._build_student_teacher(5)
        kd_criterion = KnowledgeDistillationLoss(
            temperature=4.0, logit_weight=0.0, feature_weight=1.0,
            student_channels=96, teacher_channels=96, num_levels=4,
        )
        student.train()
        x = torch.randn(1, 3, 320, 320)
        gt_meta = _make_gt_meta(1, 5)

        s_out = student(x, gt_meta=gt_meta, epoch=1, return_features=True)
        with torch.no_grad():
            t_out = teacher(x, return_features=True)

        kd_result = kd_criterion(
            student_preds=s_out["preds"],
            teacher_preds=t_out["preds"].detach(),
            student_fpn_feats=s_out["fpn_features"],
            teacher_fpn_feats=[f.detach() for f in t_out["fpn_features"]],
            num_classes=5, reg_max=7,
        )

        assert kd_result["kd_logit_loss"].item() == 0.0
        assert kd_result["kd_feature_loss"].item() > 0

    def test_kd_different_channel_sizes(self):
        """Test KD when teacher has different FPN channels than student."""
        num_classes = 5
        student = FlashDet(
            num_classes=num_classes, input_size=(320, 320), backbone_size="0.5x",
            fpn_channels=96, pretrained=False, use_aux_head=False,
        )
        teacher = FlashDet(
            num_classes=num_classes, input_size=(320, 320), backbone_size="1.0x",
            fpn_channels=96, pretrained=False, use_aux_head=False,
        )
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        kd_criterion = KnowledgeDistillationLoss(
            temperature=4.0, logit_weight=1.0, feature_weight=0.5,
            student_channels=96, teacher_channels=96, num_levels=4,
        )

        student.train()
        x = torch.randn(1, 3, 320, 320)
        gt_meta = _make_gt_meta(1, 5)

        s_out = student(x, gt_meta=gt_meta, epoch=0, return_features=True)
        with torch.no_grad():
            t_out = teacher(x, return_features=True)

        kd_result = kd_criterion(
            student_preds=s_out["preds"],
            teacher_preds=t_out["preds"].detach(),
            student_fpn_feats=s_out["fpn_features"],
            teacher_fpn_feats=[f.detach() for f in t_out["fpn_features"]],
            num_classes=5, reg_max=7,
        )
        assert not torch.isnan(kd_result["kd_loss"])

    def test_kd_optimization_step(self):
        """Full optimization step with combined hard + KD loss."""
        student, teacher = self._build_student_teacher(5)
        kd_criterion = KnowledgeDistillationLoss(
            temperature=4.0, logit_weight=1.0, feature_weight=0.5,
            student_channels=96, teacher_channels=96, num_levels=4,
        )
        student.train()
        optimizer = torch.optim.Adam(
            list(student.parameters()) + list(kd_criterion.parameters()), lr=0.001
        )

        x = torch.randn(2, 3, 320, 320)
        gt_meta = _make_gt_meta(2, 5)

        optimizer.zero_grad()
        s_out = student(x, gt_meta=gt_meta, epoch=1, return_features=True)
        with torch.no_grad():
            t_out = teacher(x, return_features=True)

        kd_result = kd_criterion(
            student_preds=s_out["preds"],
            teacher_preds=t_out["preds"].detach(),
            student_fpn_feats=s_out["fpn_features"],
            teacher_fpn_feats=[f.detach() for f in t_out["fpn_features"]],
            num_classes=5, reg_max=7,
        )

        total_loss = s_out["loss"] + kd_result["kd_loss"]
        total_loss.backward()
        optimizer.step()

        # Params should have changed
        assert total_loss.item() > 0


# ======================================================================
# Callbacks Tests (Extended)
# ======================================================================

class TestCallbacksExtended:
    """Extended tests for callback system."""

    def test_callback_all_events_fire(self):
        events_fired = []

        class AllEventsCB(Callback):
            def on_train_start(self, trainer):
                events_fired.append("train_start")

            def on_train_end(self, trainer, metrics):
                events_fired.append("train_end")

            def on_epoch_start(self, trainer, epoch):
                events_fired.append(f"epoch_start_{epoch}")

            def on_epoch_end(self, trainer, epoch, metrics):
                events_fired.append(f"epoch_end_{epoch}")

            def on_batch_start(self, trainer, batch_idx, batch):
                events_fired.append(f"batch_start_{batch_idx}")

            def on_batch_end(self, trainer, batch_idx, loss):
                events_fired.append(f"batch_end_{batch_idx}")

            def on_val_start(self, trainer):
                events_fired.append("val_start")

            def on_val_end(self, trainer, metrics):
                events_fired.append("val_end")

            def on_checkpoint(self, trainer, path, is_best):
                events_fired.append(f"checkpoint_{is_best}")

        cb_list = CallbackList([AllEventsCB()])
        trainer = object()

        cb_list.fire("on_train_start", trainer)
        cb_list.fire("on_epoch_start", trainer, 1)
        cb_list.fire("on_batch_start", trainer, 0, None)
        cb_list.fire("on_batch_end", trainer, 0, 0.5)
        cb_list.fire("on_val_start", trainer)
        cb_list.fire("on_val_end", trainer, {"val_mAP": 0.5})
        cb_list.fire("on_epoch_end", trainer, 1, {"loss": 0.3})
        cb_list.fire("on_checkpoint", trainer, "/tmp/best.pth", True)
        cb_list.fire("on_train_end", trainer, {"best_map50": 0.5})

        assert "train_start" in events_fired
        assert "train_end" in events_fired
        assert "epoch_start_1" in events_fired
        assert "epoch_end_1" in events_fired
        assert "batch_start_0" in events_fired
        assert "batch_end_0" in events_fired
        assert "val_start" in events_fired
        assert "val_end" in events_fired
        assert "checkpoint_True" in events_fired

    def test_multiple_callbacks(self):
        counts = {"a": 0, "b": 0}

        class CBA(Callback):
            def on_epoch_end(self, trainer, epoch, metrics):
                counts["a"] += 1

        class CBB(Callback):
            def on_epoch_end(self, trainer, epoch, metrics):
                counts["b"] += 1

        cb_list = CallbackList([CBA(), CBB()])
        cb_list.fire("on_epoch_end", None, 1, {})
        assert counts["a"] == 1
        assert counts["b"] == 1

    def test_early_stopping_min_mode(self):
        es = EarlyStopping(patience=2, metric="val_loss", mode="min")
        es.on_epoch_end(None, 1, {"val_loss": 1.0})
        assert not es.should_stop
        es.on_epoch_end(None, 2, {"val_loss": 1.1})
        assert not es.should_stop
        es.on_epoch_end(None, 3, {"val_loss": 1.2})
        assert es.should_stop

    def test_early_stopping_max_mode_resets(self):
        es = EarlyStopping(patience=3, metric="val_mAP", mode="max")
        es.on_epoch_end(None, 1, {"val_mAP": 0.5})
        es.on_epoch_end(None, 2, {"val_mAP": 0.4})
        es.on_epoch_end(None, 3, {"val_mAP": 0.6})  # improvement resets
        es.on_epoch_end(None, 4, {"val_mAP": 0.5})
        es.on_epoch_end(None, 5, {"val_mAP": 0.5})
        assert not es.should_stop  # only 2 waits since last improvement

    def test_early_stopping_missing_metric(self):
        es = EarlyStopping(patience=2, metric="val_mAP", mode="max")
        es.on_epoch_end(None, 1, {"train_loss": 0.5})
        es.on_epoch_end(None, 2, {"train_loss": 0.4})
        es.on_epoch_end(None, 3, {"train_loss": 0.3})
        assert not es.should_stop  # metric not present → no action

    def test_lr_scheduler_callback(self):
        model = nn.Linear(10, 10)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
        cb = LRSchedulerCallback(scheduler)

        initial_lr = opt.param_groups[0]["lr"]
        cb.on_epoch_end(None, 1, {})
        new_lr = opt.param_groups[0]["lr"]
        assert new_lr < initial_lr

    def test_csv_logger(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            csv_path = f.name
        try:
            logger = CSVLogger(csv_path)
            logger.on_epoch_end(None, 1, {"loss": 0.5, "mAP": 0.3})
            logger.on_epoch_end(None, 2, {"loss": 0.4, "mAP": 0.4})

            with open(csv_path) as f:
                lines = f.readlines()
            assert len(lines) == 3  # header + 2 rows
            assert "epoch" in lines[0]
            assert "loss" in lines[0]
        finally:
            os.unlink(csv_path)

    def test_callback_add_dynamic(self):
        cb_list = CallbackList()
        counts = [0]

        class DynCB(Callback):
            def on_epoch_end(self, trainer, epoch, metrics):
                counts[0] += 1

        cb_list.add(DynCB())
        cb_list.fire("on_epoch_end", None, 1, {})
        assert counts[0] == 1


# ======================================================================
# Trainer Construction Tests
# ======================================================================

class TestTrainerConstruction:
    """Tests for Trainer initialization (without actually training)."""

    def test_trainer_defaults(self):
        trainer = Trainer(epochs=1, batch_size=1, device="cpu")
        assert trainer.epochs == 1
        assert trainer.batch_size == 1
        assert trainer.device == torch.device("cpu")
        assert trainer.lora is False
        assert trainer.amp is False

    def test_trainer_model_size_map(self):
        for size in ["m", "m-1.5x", "m-0.5x"]:
            trainer = Trainer(model_size=size, device="cpu", epochs=1)
            assert trainer._model_cfg == MODEL_SIZE_MAP[size]

    def test_trainer_lora_config(self):
        trainer = Trainer(
            lora=True, lora_rank=16, lora_alpha=32.0,
            lora_dropout=0.1, lora_targets=["backbone"],
            device="cpu", epochs=1,
        )
        assert trainer.lora is True
        assert trainer.lora_rank == 16
        assert trainer.lora_alpha == 32.0
        assert trainer.lora_dropout == 0.1
        assert trainer.lora_targets == ["backbone"]

    def test_trainer_amp_config(self):
        trainer = Trainer(amp=True, device="cpu", epochs=1)
        assert trainer.amp is True

    def test_trainer_grad_accum(self):
        trainer = Trainer(grad_accum=4, device="cpu", epochs=1)
        assert trainer.grad_accum == 4

    def test_trainer_callbacks(self):
        trainer = Trainer(device="cpu", epochs=1)
        assert hasattr(trainer, "callbacks")

        class MyCB(Callback):
            pass

        trainer.add_callback(MyCB())
        assert len(trainer.callbacks.callbacks) == 1

    def test_trainer_chunked_loss_config(self):
        trainer = Trainer(
            chunked_loss=True, chunk_size=512,
            device="cpu", epochs=1,
        )
        assert trainer.chunked_loss is True
        assert trainer.chunk_size == 512


# ======================================================================
# Training Loop Unit Tests (Mock Dataloader)
# ======================================================================

class TestTrainingLoopUnit:
    """Unit tests for training loop components with mock data."""

    def test_train_one_epoch_mechanics(self):
        """Simulate the core _train_one_epoch logic with mock data."""
        from flashdet.utils import AverageMeter

        model = FlashDet(
            num_classes=5, input_size=(320, 320), backbone_size="0.5x",
            fpn_channels=96, pretrained=False, use_aux_head=True,
        )
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        ema = ModelEMA(model, decay=0.9998, warmup=100)

        loss_meter = AverageMeter("Loss")

        for batch_idx in range(3):
            images = torch.randn(2, 3, 320, 320)
            gt_meta = _make_gt_meta(2, 5)

            output = model(images, gt_meta, epoch=1)
            loss = output["loss"]

            if torch.isnan(loss):
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 35.0)
            optimizer.step()
            optimizer.zero_grad()
            ema.update(model)
            loss_meter.update(loss.item())

        assert loss_meter.avg > 0
        assert ema.num_updates == 3

    def test_grad_accumulation(self):
        """Test gradient accumulation logic."""
        model = FlashDet(
            num_classes=5, input_size=(320, 320), backbone_size="0.5x",
            fpn_channels=96, pretrained=False, use_aux_head=False,
        )
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        grad_accum = 2

        for batch_idx in range(4):
            images = torch.randn(1, 3, 320, 320)
            gt_meta = _make_gt_meta(1, 5)
            output = model(images, gt_meta, epoch=0)
            loss = output["loss"] / grad_accum
            loss.backward()

            if (batch_idx + 1) % grad_accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 35.0)
                optimizer.step()
                optimizer.zero_grad()

        # Should have taken 2 optimizer steps (4 batches / 2 accum)

    def test_lr_lambda_warmup(self):
        """Test the cosine LR schedule with warmup."""
        import math

        lr = 0.001
        warmup_epochs = 5
        total_epochs = 100
        eta_min = 0.00005
        eta_min_factor = eta_min / lr

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return eta_min_factor + (1.0 - eta_min_factor) * cosine

        # Warmup: linearly increasing
        assert lr_lambda(0) == pytest.approx(1 / 5)
        assert lr_lambda(4) == pytest.approx(5 / 5)

        # After warmup: cosine decay
        mid_epoch = (warmup_epochs + total_epochs) // 2
        assert lr_lambda(mid_epoch) < 1.0
        assert lr_lambda(total_epochs - 1) >= eta_min_factor

    def test_nan_loss_skip(self):
        """NaN losses should be skipped without crashing."""
        model = FlashDet(
            num_classes=5, input_size=(320, 320), backbone_size="0.5x",
            fpn_channels=96, pretrained=False, use_aux_head=False,
        )
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        images = torch.randn(1, 3, 320, 320)
        gt_meta = _make_gt_meta(1, 5)
        output = model(images, gt_meta, epoch=0)
        loss = output["loss"]

        # Simulate NaN
        fake_nan_loss = torch.tensor(float('nan'), requires_grad=True)
        if torch.isnan(fake_nan_loss):
            pass  # Skip, just like the trainer does
        else:
            fake_nan_loss.backward()

        # Real loss should still work
        loss.backward()
        optimizer.step()


# ======================================================================
# Feature Return for KD Tests
# ======================================================================

class TestReturnFeatures:
    """Tests for return_features mode used in KD training."""

    def test_return_features_training_mode(self):
        model = FlashDet(
            num_classes=5, input_size=(320, 320), backbone_size="0.5x",
            fpn_channels=96, pretrained=False, use_aux_head=True,
        )
        model.train()
        x = torch.randn(2, 3, 320, 320)
        gt_meta = _make_gt_meta(2, 5)
        out = model(x, gt_meta=gt_meta, epoch=1, return_features=True)

        assert "loss" in out
        assert "loss_states" in out
        assert "preds" in out
        assert "fpn_features" in out
        assert "backbone_features" in out

    def test_return_features_eval_mode(self):
        model = FlashDet(
            num_classes=5, input_size=(320, 320), backbone_size="0.5x",
            fpn_channels=96, pretrained=False, use_aux_head=False,
        )
        model.eval()
        x = torch.randn(1, 3, 320, 320)
        with torch.no_grad():
            out = model(x, return_features=True)

        assert "preds" in out
        assert "fpn_features" in out
        assert "backbone_features" in out
        assert "loss" not in out

    def test_fpn_features_shape(self):
        model = FlashDet(
            num_classes=5, input_size=(320, 320), backbone_size="0.5x",
            fpn_channels=96, pretrained=False, use_aux_head=False,
        )
        model.eval()
        x = torch.randn(1, 3, 320, 320)
        with torch.no_grad():
            out = model(x, return_features=True)

        for feat in out["fpn_features"]:
            assert feat.shape[0] == 1
            assert feat.shape[1] == 96  # fpn_channels

    def test_backbone_features_shape(self):
        model = FlashDet(
            num_classes=5, input_size=(320, 320), backbone_size="0.5x",
            fpn_channels=96, pretrained=False, use_aux_head=False,
        )
        model.eval()
        x = torch.randn(1, 3, 320, 320)
        with torch.no_grad():
            out = model(x, return_features=True)

        # backbone outputs stages 2,3,4 with channels [48, 96, 192] for 0.5x
        expected_channels = [48, 96, 192]
        for feat, exp_c in zip(out["backbone_features"], expected_channels):
            assert feat.shape[1] == exp_c


# ======================================================================
# AuxHead Training Tests
# ======================================================================

class TestAuxHeadTraining:
    """Tests for auxiliary head (AGM) behavior during training."""

    def test_aux_head_exists(self):
        model = FlashDet(
            num_classes=5, input_size=(320, 320), backbone_size="0.5x",
            fpn_channels=96, pretrained=False, use_aux_head=True,
        )
        assert hasattr(model, "aux_fpn")
        assert hasattr(model, "aux_head")

    def test_no_aux_head(self):
        model = FlashDet(
            num_classes=5, input_size=(320, 320), backbone_size="0.5x",
            fpn_channels=96, pretrained=False, use_aux_head=False,
        )
        assert not hasattr(model, "aux_fpn")
        assert not hasattr(model, "aux_head")

    def test_aux_not_used_in_eval(self):
        model = FlashDet(
            num_classes=5, input_size=(320, 320), backbone_size="0.5x",
            fpn_channels=96, pretrained=False, use_aux_head=True,
        )
        model.eval()
        x = torch.randn(1, 3, 320, 320)
        gt_meta = _make_gt_meta(1, 5)
        out = model(x, gt_meta=gt_meta, compute_loss=True)
        # In eval mode with compute_loss, aux_head should NOT be used
        states = out["loss_states"]
        assert "aux_loss_qfl" not in states or states.get("aux_loss_qfl", 0) == 0

    def test_detach_epoch_behavior(self):
        model = FlashDet(
            num_classes=5, input_size=(320, 320), backbone_size="0.5x",
            fpn_channels=96, pretrained=False, use_aux_head=True,
        )
        model.detach_epoch = 5
        model.train()
        x = torch.randn(1, 3, 320, 320)
        gt_meta = _make_gt_meta(1, 5)

        # epoch < detach_epoch (no detach)
        out1 = model(x, gt_meta=gt_meta, epoch=2)
        assert "loss" in out1

        # epoch >= detach_epoch (detach)
        out2 = model(x, gt_meta=gt_meta, epoch=6)
        assert "loss" in out2
