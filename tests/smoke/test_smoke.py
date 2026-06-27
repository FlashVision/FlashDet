"""Smoke tests — quick sanity checks that run in <10s total.

These verify the package is importable and basic operations work.
Run on every PR as the first CI gate.
"""

import pytest


@pytest.mark.smoke
class TestImports:
    """Verify all public modules are importable."""

    def test_import_flashdet(self):
        import flashdet
        assert hasattr(flashdet, "__version__")

    def test_import_models(self):
        from flashdet.models import build_model
        from flashdet.models.detector import FlashDet
        assert FlashDet is not None

    def test_import_engine(self):
        from flashdet.engine import Trainer, Validator, Predictor
        assert Trainer is not None

    def test_import_trackers(self):
        from flashdet.trackers import FlashTracker
        assert FlashTracker is not None

    def test_import_solutions(self):
        from flashdet.solutions import (
            ObjectCounter, SpeedEstimator, Heatmap, RegionCounter,
            QueueManager, DistanceCalculator, ParkingManager,
            SecurityAlarm, WorkoutMonitor, LiveInference, AnalyticsDashboard,
        )
        assert ObjectCounter is not None

    def test_import_analytics(self):
        from flashdet.analytics import (
            Benchmark, Profiler, FLOPsCounter,
            DetectionMetrics, DetectionErrorAnalyzer,
            DatasetAnalyzer, AnalyticsReport,
        )
        assert FLOPsCounter is not None

    def test_import_data(self):
        from flashdet.data import (
            FlashDetDataset, create_dataloader,
            convert_dataset, detect_dataset_format,
        )
        assert FlashDetDataset is not None

    def test_import_losses(self):
        from flashdet.losses import E2EDetectionLoss
        assert E2EDetectionLoss is not None

    def test_import_cfg(self):
        from flashdet.cfg import get_config
        assert get_config is not None

    def test_import_lora(self):
        from flashdet.models.lora import apply_lora, apply_qlora, merge_lora_weights
        assert apply_lora is not None


@pytest.mark.smoke
class TestBasicOperations:
    """Quick functional checks."""

    def test_model_instantiation(self):
        import torch
        from flashdet.models.detector import FlashDet
        model = FlashDet(num_classes=5, size="n")
        assert model is not None
        params = sum(p.numel() for p in model.parameters())
        assert params > 0

    def test_model_forward(self):
        import torch
        from flashdet.models.detector import FlashDet
        model = FlashDet(num_classes=5, size="n")
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(1, 3, 320, 320))
        assert "preds" in out

    def test_config_creation(self):
        from flashdet.cfg import get_config
        cfg = get_config(model_size="n", input_size=320, num_classes=10)
        assert cfg.model.num_classes == 10

    def test_tracker_instantiation(self):
        from flashdet.trackers import FlashTracker
        tracker = FlashTracker()
        assert tracker is not None

    def test_metrics_instantiation(self):
        from flashdet.analytics import DetectionMetrics
        m = DetectionMetrics(num_classes=5)
        assert m is not None

    def test_dataset_format_detection(self):
        import tempfile, os
        from flashdet.data import detect_dataset_format
        tmp = tempfile.mkdtemp()
        fmt = detect_dataset_format(tmp)
        assert fmt == "unknown"
        os.rmdir(tmp)

    def test_convert_dataset_noop(self):
        """convert_dataset should no-op for already-correct format."""
        import json, os, tempfile, shutil
        from flashdet.data import convert_dataset

        tmp = tempfile.mkdtemp()
        train_dir = os.path.join(tmp, "train")
        os.makedirs(train_dir)
        coco = {"images": [], "annotations": [], "categories": []}
        with open(os.path.join(train_dir, "_annotations.coco.json"), "w") as f:
            json.dump(coco, f)

        result = convert_dataset(tmp, target_format="coco")
        assert result["status"] == "already_in_target_format"
        shutil.rmtree(tmp)
