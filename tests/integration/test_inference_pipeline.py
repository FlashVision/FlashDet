"""Integration tests — end-to-end inference pipeline."""

import numpy as np
import pytest
import torch

from flashdet.models.detector import FlashDet


class TestInferencePipeline:
    """Full inference pipeline tests."""

    def test_numpy_image_to_detections(self):
        """Simulate full inference: raw image → detections."""
        from flashdet.data.transforms import InferenceTransform

        model = FlashDet(num_classes=10, size="n")
        model.eval()

        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        tf = InferenceTransform(input_size=(320, 320))
        result = tf(img)
        if isinstance(result, tuple):
            result = result[0]
        if isinstance(result, np.ndarray):
            tensor = torch.from_numpy(result).unsqueeze(0)
        else:
            tensor = result.unsqueeze(0) if result.ndim == 3 else result

        with torch.no_grad():
            out = model(tensor)

        assert "preds" in out
        assert out["preds"].shape[0] == 1

    def test_batch_inference_consistency(self):
        """Single vs batch inference should be consistent."""
        model = FlashDet(num_classes=5, size="n")
        model.eval()

        torch.manual_seed(0)
        x1 = torch.randn(1, 3, 320, 320)
        x2 = torch.randn(1, 3, 320, 320)
        batch = torch.cat([x1, x2], dim=0)

        with torch.no_grad():
            out_single_1 = model(x1)
            out_single_2 = model(x2)
            out_batch = model(batch)

        # Batch results should match individual
        assert torch.allclose(out_batch["preds"][0], out_single_1["preds"][0], atol=1e-5)
        assert torch.allclose(out_batch["preds"][1], out_single_2["preds"][0], atol=1e-5)

    def test_predict_api(self):
        """model.predict() end-to-end."""
        model = FlashDet(num_classes=10, size="n")
        x = torch.randn(3, 3, 320, 320)
        results = model.predict(x, score_thr=0.01)
        assert len(results) == 3
        for bboxes, labels in results:
            assert bboxes.ndim == 2
            assert labels.ndim == 1
            if len(bboxes) > 0:
                # Scores should be in [0, 1]
                assert (bboxes[:, 4] >= 0).all()
                assert (bboxes[:, 4] <= 1).all()
                # Labels should be valid
                assert (labels >= 0).all()
                assert (labels < 10).all()

    def test_feature_extraction(self):
        """Return intermediate features for downstream use."""
        model = FlashDet(num_classes=5, size="n")
        model.eval()
        x = torch.randn(1, 3, 320, 320)
        with torch.no_grad():
            out = model(x, return_features=True)
        assert len(out["backbone_features"]) == 3
        assert len(out["fpn_features"]) == 3
        # Features should have spatial dimensions
        for feat in out["fpn_features"]:
            assert feat.ndim == 4
            assert feat.shape[0] == 1
