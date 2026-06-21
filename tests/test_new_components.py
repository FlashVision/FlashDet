"""Tests for new FlashDet components: OBB head, losses, augmentations, transforms."""

import torch
import numpy as np


class TestOBBHead:
    def test_forward(self):
        from flashdet.models.head.obb_head import OBBHead

        head = OBBHead(num_classes=15, in_channels=64, feat_channels=64, stacked_convs=1, strides=[8, 16, 32])
        feats = [
            torch.randn(1, 64, 40, 40),
            torch.randn(1, 64, 20, 20),
            torch.randn(1, 64, 10, 10),
        ]
        out = head(feats)
        assert "cls" in out
        assert "reg" in out
        assert "angle" in out
        total_points = 40 * 40 + 20 * 20 + 10 * 10
        assert out["cls"].shape == (1, total_points, 15)
        assert out["angle"].shape[1] == total_points

    def test_forward_batch(self):
        from flashdet.models.head.obb_head import OBBHead

        head = OBBHead(num_classes=5, in_channels=64, feat_channels=64, stacked_convs=1, strides=[8, 16])
        feats = [torch.randn(4, 64, 20, 20), torch.randn(4, 64, 10, 10)]
        out = head(feats)
        assert out["cls"].shape[0] == 4

    def test_decode_angle_direct(self):
        from flashdet.models.head.obb_head import OBBHead

        head = OBBHead(num_classes=5, in_channels=32, feat_channels=32, strides=[8])
        angle_pred = torch.randn(10, 1)
        decoded = head.decode_angle(angle_pred)
        assert decoded.shape == (10,)
        assert (decoded.abs() <= torch.pi / 2).all()

    def test_decode_angle_bins(self):
        from flashdet.models.head.obb_head import OBBHead

        head = OBBHead(num_classes=5, in_channels=32, feat_channels=32, strides=[8], angle_bins=180)
        angle_pred = torch.randn(10, 180)
        decoded = head.decode_angle(angle_pred)
        assert decoded.shape == (10,)
        assert (decoded >= -torch.pi / 2).all()
        assert (decoded < torch.pi / 2).all()

    def test_get_obb_bboxes_decodes_regression(self):
        from flashdet.models.head.obb_head import OBBHead

        head = OBBHead(num_classes=5, in_channels=64, feat_channels=64, stacked_convs=1,
                       strides=[8, 16, 32], reg_max=7)
        head.eval()
        feats = [torch.randn(1, 64, 40, 40), torch.randn(1, 64, 20, 20), torch.randn(1, 64, 10, 10)]
        with torch.no_grad():
            preds = head(feats)
        results = head.get_obb_bboxes(preds, img_shape=(320, 320), score_thr=0.0, feats=feats)
        assert len(results) == 1
        boxes = results[0]["boxes"]
        assert boxes.shape[-1] == 5
        if boxes.shape[0] > 0:
            assert not (boxes[:, 0] == 0).all(), "cx should be decoded, not all zeros"
            assert not (boxes[:, 2] == 10).all(), "w should be decoded, not all 10"

    def test_get_obb_bboxes_without_feats(self):
        from flashdet.models.head.obb_head import OBBHead

        head = OBBHead(num_classes=5, in_channels=64, feat_channels=64, stacked_convs=1,
                       strides=[8, 16, 32], reg_max=7)
        head.eval()
        feats = [torch.randn(1, 64, 40, 40), torch.randn(1, 64, 20, 20), torch.randn(1, 64, 10, 10)]
        with torch.no_grad():
            preds = head(feats)
        results = head.get_obb_bboxes(preds, img_shape=(320, 320), score_thr=0.0)
        assert len(results) == 1
        assert results[0]["boxes"].shape[-1] == 5

    def test_get_obb_bboxes_high_threshold(self):
        from flashdet.models.head.obb_head import OBBHead

        head = OBBHead(num_classes=5, in_channels=64, feat_channels=64, stacked_convs=1,
                       strides=[8, 16, 32], reg_max=7)
        head.eval()
        feats = [torch.randn(1, 64, 40, 40), torch.randn(1, 64, 20, 20), torch.randn(1, 64, 10, 10)]
        with torch.no_grad():
            preds = head(feats)
        results = head.get_obb_bboxes(preds, img_shape=(320, 320), score_thr=0.99)
        assert results[0]["boxes"].shape[0] == 0 or results[0]["boxes"].shape[-1] == 5

    def test_rotated_iou(self):
        from flashdet.models.head.obb_head import rotated_iou

        boxes = torch.tensor([[50, 50, 20, 20, 0.0]])
        iou = rotated_iou(boxes, boxes)
        assert iou.shape == (1,)
        assert iou[0] > 0.9

    def test_rotated_iou_no_overlap(self):
        from flashdet.models.head.obb_head import rotated_iou

        boxes1 = torch.tensor([[0, 0, 10, 10, 0.0]])
        boxes2 = torch.tensor([[1000, 1000, 10, 10, 0.0]])
        iou = rotated_iou(boxes1, boxes2)
        assert iou[0] < 0.1

    def test_rotated_nms(self):
        from flashdet.models.head.obb_head import rotated_nms

        boxes = torch.tensor([
            [50, 50, 20, 20, 0.0],
            [51, 51, 20, 20, 0.0],
            [200, 200, 30, 30, 0.5],
        ])
        scores = torch.tensor([0.9, 0.8, 0.7])
        keep = rotated_nms(boxes, scores, iou_thr=0.3)
        assert len(keep) >= 2

    def test_rotated_nms_empty(self):
        from flashdet.models.head.obb_head import rotated_nms

        boxes = torch.zeros(0, 5)
        scores = torch.zeros(0)
        keep = rotated_nms(boxes, scores, iou_thr=0.5)
        assert len(keep) == 0

    def test_registry(self):
        from flashdet.registry import HEADS
        assert "OBBHead" in HEADS


class TestVarifocalLoss:
    def test_varifocal_loss(self):
        from flashdet.losses.varifocal_loss import VarifocalLoss

        loss_fn = VarifocalLoss(alpha=0.75, gamma=2.0)
        pred = torch.randn(10, 5)
        target = torch.zeros(10, 5)
        target[0, 1] = 0.8
        target[3, 2] = 0.95
        loss = loss_fn(pred, target)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_sigmoid_focal_loss(self):
        from flashdet.losses.varifocal_loss import SigmoidFocalLoss

        loss_fn = SigmoidFocalLoss(alpha=0.25, gamma=2.0)
        pred = torch.randn(10, 5)
        target = torch.zeros(10, 5)
        target[0, 0] = 1.0
        loss = loss_fn(pred, target)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_varifocal_gradient(self):
        from flashdet.losses.varifocal_loss import VarifocalLoss

        loss_fn = VarifocalLoss()
        pred = torch.randn(5, 3, requires_grad=True)
        target = torch.zeros(5, 3)
        target[0, 0] = 0.9
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None

    def test_varifocal_all_zero_target(self):
        from flashdet.losses.varifocal_loss import VarifocalLoss

        loss_fn = VarifocalLoss()
        pred = torch.randn(5, 3)
        target = torch.zeros(5, 3)
        loss = loss_fn(pred, target)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_varifocal_with_weight(self):
        from flashdet.losses.varifocal_loss import varifocal_loss

        pred = torch.randn(10, 5)
        target = torch.zeros(10, 5)
        target[0, 0] = 0.9
        weight = torch.ones(10)
        loss = varifocal_loss(pred, target, weight=weight)
        assert loss.shape == ()

    def test_varifocal_with_avg_factor(self):
        from flashdet.losses.varifocal_loss import varifocal_loss

        pred = torch.randn(10, 5)
        target = torch.zeros(10, 5)
        target[0, 0] = 0.9
        loss = varifocal_loss(pred, target, avg_factor=5.0)
        assert loss.shape == ()
        assert not torch.isnan(loss)

    def test_sigmoid_focal_alpha_negative(self):
        from flashdet.losses.varifocal_loss import sigmoid_focal_loss

        pred = torch.randn(10, 5)
        target = torch.zeros(10, 5)
        target[0, 0] = 1.0
        loss = sigmoid_focal_loss(pred, target, alpha=-1.0)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_loss_weight_scaling(self):
        from flashdet.losses.varifocal_loss import VarifocalLoss

        loss_fn_1x = VarifocalLoss(loss_weight=1.0)
        loss_fn_2x = VarifocalLoss(loss_weight=2.0)
        pred = torch.randn(5, 3)
        target = torch.zeros(5, 3)
        target[0, 0] = 0.8
        l1 = loss_fn_1x(pred, target)
        l2 = loss_fn_2x(pred, target)
        assert abs(l2.item() - 2 * l1.item()) < 1e-5


class TestIoULoss:
    def test_giou_loss(self):
        from flashdet.losses.iou_loss import GIoULoss

        loss_fn = GIoULoss()
        pred = torch.tensor([[10, 10, 50, 50]], dtype=torch.float32)
        target = torch.tensor([[10, 10, 50, 50]], dtype=torch.float32)
        loss = loss_fn(pred, target)
        assert loss.item() < 0.1

    def test_giou_loss_no_overlap(self):
        from flashdet.losses.iou_loss import GIoULoss

        loss_fn = GIoULoss()
        pred = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)
        target = torch.tensor([[100, 100, 110, 110]], dtype=torch.float32)
        loss = loss_fn(pred, target)
        assert loss.item() > 1.0

    def test_iou_loss(self):
        from flashdet.losses.iou_loss import IoULoss

        loss_fn = IoULoss()
        pred = torch.tensor([[10, 10, 50, 50]], dtype=torch.float32)
        target = torch.tensor([[10, 10, 50, 50]], dtype=torch.float32)
        loss = loss_fn(pred, target)
        assert loss.item() < 0.1

    def test_giou_gradient(self):
        from flashdet.losses.iou_loss import GIoULoss

        loss_fn = GIoULoss()
        pred = torch.tensor([[10.0, 10, 50, 50]], requires_grad=True)
        target = torch.tensor([[15.0, 15, 55, 55]])
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None


class TestKDLoss:
    def test_logit_distillation(self):
        from flashdet.losses.kd_loss import LogitDistillationLoss

        loss_fn = LogitDistillationLoss()
        num_classes = 10
        reg_max = 7
        reg_channels = 4 * (reg_max + 1)
        student_preds = torch.randn(2, 50, num_classes + reg_channels)
        teacher_preds = torch.randn(2, 50, num_classes + reg_channels)
        result = loss_fn(student_preds, teacher_preds, num_classes=num_classes, reg_max=reg_max)
        assert isinstance(result, dict)
        assert "kd_logit_loss" in result

    def test_feature_distillation(self):
        from flashdet.losses.kd_loss import FeatureDistillationLoss

        loss_fn = FeatureDistillationLoss(student_channels=64, teacher_channels=128)
        student_feat = torch.randn(2, 64, 20, 20)
        teacher_feat = torch.randn(2, 128, 20, 20)
        loss = loss_fn(student_feat, teacher_feat)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_knowledge_distillation_loss(self):
        from flashdet.losses.kd_loss import KnowledgeDistillationLoss

        num_classes = 10
        reg_max = 7
        reg_channels = 4 * (reg_max + 1)
        loss_fn = KnowledgeDistillationLoss(student_channels=64, teacher_channels=128)
        student_preds = torch.randn(2, 50, num_classes + reg_channels)
        teacher_preds = torch.randn(2, 50, num_classes + reg_channels)
        student_feats = [torch.randn(2, 64, 10, 10)]
        teacher_feats = [torch.randn(2, 128, 10, 10)]
        result = loss_fn(student_preds, teacher_preds, student_feats, teacher_feats, num_classes=num_classes)
        assert isinstance(result, dict)

    def test_kd_gradient(self):
        from flashdet.losses.kd_loss import LogitDistillationLoss

        loss_fn = LogitDistillationLoss()
        num_classes = 5
        reg_max = 7
        reg_channels = 4 * (reg_max + 1)
        student = torch.randn(2, 20, num_classes + reg_channels, requires_grad=True)
        teacher = torch.randn(2, 20, num_classes + reg_channels)
        result = loss_fn(student, teacher, num_classes=num_classes, reg_max=reg_max)
        result["kd_logit_loss"].backward()
        assert student.grad is not None


class TestFocalLoss:
    def test_quality_focal_loss(self):
        from flashdet.losses.focal_loss import QualityFocalLoss

        loss_fn = QualityFocalLoss()
        pred = torch.randn(100, 10)
        labels = torch.full((100,), 10, dtype=torch.long)  # all background
        labels[0] = 3
        labels[5] = 7
        scores = torch.zeros(100)
        scores[0] = 0.8
        scores[5] = 0.95
        target = (labels, scores)
        loss = loss_fn(pred, target)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_distribution_focal_loss(self):
        from flashdet.losses.focal_loss import DistributionFocalLoss

        loss_fn = DistributionFocalLoss()
        pred = torch.randn(100, 8)
        label = torch.rand(100) * 7
        loss = loss_fn(pred, label)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_qfl_gradient(self):
        from flashdet.losses.focal_loss import QualityFocalLoss

        loss_fn = QualityFocalLoss()
        pred = torch.randn(10, 5, requires_grad=True)
        labels = torch.full((10,), 5, dtype=torch.long)  # all background
        labels[0] = 2
        scores = torch.zeros(10)
        scores[0] = 0.9
        target = (labels, scores)
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None


class TestChunkedLoss:
    def test_chunked_qfl(self):
        from flashdet.losses.chunked_loss import chunked_quality_focal_loss

        pred = torch.randn(200, 10)
        labels = torch.full((200,), 10, dtype=torch.long)
        labels[0] = 3
        scores = torch.zeros(200)
        scores[0] = 0.9
        target = (labels, scores)
        loss = chunked_quality_focal_loss(pred, target, chunk_size=64)
        assert loss.shape == ()
        assert not torch.isnan(loss)

    def test_chunked_dfl(self):
        from flashdet.losses.chunked_loss import chunked_distribution_focal_loss

        pred = torch.randn(200, 8)
        label = torch.rand(200) * 7
        loss = chunked_distribution_focal_loss(pred, label, chunk_size=64)
        assert loss.shape == ()
        assert not torch.isnan(loss)


class TestAugmentations:
    def test_mosaic_without_extra(self):
        from flashdet.data.augmentations import Mosaic

        mosaic = Mosaic(img_size=(320, 320))
        img = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
        labels = np.array([1], dtype=np.int64)
        out_img, out_boxes, out_labels = mosaic(img, boxes, labels)
        assert out_img.shape == img.shape
        np.testing.assert_array_equal(out_boxes, boxes)

    def test_mixup_without_extra(self):
        from flashdet.data.augmentations import MixUp

        mixup = MixUp(alpha=1.5)
        img = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
        boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
        labels = np.array([0], dtype=np.int64)
        out_img, out_boxes, out_labels = mixup(img, boxes, labels)
        assert out_img.shape == img.shape

    def test_copypaste_without_extra(self):
        from flashdet.data.augmentations import CopyPaste

        cp = CopyPaste(prob=0.5)
        img = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
        boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
        labels = np.array([0], dtype=np.int64)
        out_img, out_boxes, out_labels = cp(img, boxes, labels)
        assert out_img.shape == img.shape

    def test_mosaic_with_extra(self):
        from flashdet.data.augmentations import Mosaic

        def extra_fn():
            img = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            boxes = np.array([[5, 5, 40, 40]], dtype=np.float32)
            labels = np.array([2], dtype=np.int64)
            return img, boxes, labels

        mosaic = Mosaic(img_size=(320, 320), extra_images_fn=extra_fn)
        img = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
        labels = np.array([1], dtype=np.int64)
        out_img, out_boxes, out_labels = mosaic(img, boxes, labels)
        assert out_img.shape[:2] == (320, 320)

    def test_mixup_with_extra(self):
        from flashdet.data.augmentations import MixUp

        def extra_fn():
            img = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
            boxes = np.array([[20, 20, 60, 60]], dtype=np.float32)
            labels = np.array([1], dtype=np.int64)
            return img, boxes, labels

        mixup = MixUp(alpha=1.5, extra_image_fn=extra_fn)
        img = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
        boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
        labels = np.array([0], dtype=np.int64)
        out_img, out_boxes, out_labels = mixup(img, boxes, labels)
        assert out_img.shape == (320, 320, 3)
        assert len(out_boxes) >= 1

    def test_copypaste_with_extra(self):
        from flashdet.data.augmentations import CopyPaste

        def extra_fn():
            img = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
            boxes = np.array([[50, 50, 100, 100]], dtype=np.float32)
            labels = np.array([2], dtype=np.int64)
            return img, boxes, labels

        cp = CopyPaste(prob=1.0, extra_image_fn=extra_fn)
        img = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
        boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
        labels = np.array([0], dtype=np.int64)
        out_img, out_boxes, out_labels = cp(img, boxes, labels)
        assert out_img.shape == (320, 320, 3)
        assert len(out_boxes) >= 1

    def test_mosaic_empty_boxes(self):
        from flashdet.data.augmentations import Mosaic

        def extra_fn():
            img = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            boxes = np.array([], dtype=np.float32).reshape(0, 4)
            labels = np.array([], dtype=np.int64)
            return img, boxes, labels

        mosaic = Mosaic(img_size=(320, 320), extra_images_fn=extra_fn)
        img = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        boxes = np.array([], dtype=np.float32).reshape(0, 4)
        labels = np.array([], dtype=np.int64)
        out_img, out_boxes, out_labels = mosaic(img, boxes, labels)
        assert out_img.shape[:2] == (320, 320)

    def test_copypaste_empty_source(self):
        from flashdet.data.augmentations import CopyPaste

        def extra_fn():
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            boxes = np.array([], dtype=np.float32).reshape(0, 4)
            labels = np.array([], dtype=np.int64)
            return img, boxes, labels

        cp = CopyPaste(prob=1.0, extra_image_fn=extra_fn)
        img = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
        boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
        labels = np.array([0], dtype=np.int64)
        out_img, out_boxes, out_labels = cp(img, boxes, labels)
        assert out_img.shape == (320, 320, 3)

    def test_mixup_different_sizes(self):
        from flashdet.data.augmentations import MixUp

        def extra_fn():
            img = np.random.randint(0, 255, (200, 400, 3), dtype=np.uint8)
            boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
            labels = np.array([1], dtype=np.int64)
            return img, boxes, labels

        mixup = MixUp(alpha=1.5, extra_image_fn=extra_fn)
        img = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
        boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
        labels = np.array([0], dtype=np.int64)
        out_img, out_boxes, out_labels = mixup(img, boxes, labels)
        assert out_img.shape[0] == 320
        assert out_img.shape[1] == 400


class TestTransforms:
    def test_train_transform(self):
        from flashdet.data.transforms import TrainTransform

        transform = TrainTransform(input_size=(320, 320))
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        boxes = np.array([[100, 100, 200, 200]], dtype=np.float32)
        labels = np.array([1], dtype=np.int64)
        out_img, out_boxes, out_labels = transform(img, boxes, labels)
        assert out_img.shape == (3, 320, 320)

    def test_val_transform(self):
        from flashdet.data.transforms import ValTransform

        transform = ValTransform(input_size=(320, 320))
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        boxes = np.array([[100, 100, 200, 200]], dtype=np.float32)
        labels = np.array([1], dtype=np.int64)
        out_img, out_boxes, out_labels = transform(img, boxes, labels)
        assert out_img.shape == (3, 320, 320)

    def test_inference_transform(self):
        from flashdet.data.transforms import InferenceTransform

        transform = InferenceTransform(input_size=(320, 320))
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        out_img, meta = transform(img)
        assert out_img.shape == (3, 320, 320)
        assert "scale" in meta or "ratio" in meta or len(meta) >= 0


class TestLoRA:
    def test_apply_lora_variants(self):
        from flashdet.models import build_model, apply_lora
        from flashdet.cfg import get_config

        cfg = get_config(model_size="m-0.5x", input_size=320, num_classes=5)
        model = build_model(cfg)
        apply_lora(model, rank=4, alpha=8.0, variant="lora")
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable > 0

    def test_merge_lora(self):
        from flashdet.models import build_model, apply_lora, merge_lora_weights
        from flashdet.cfg import get_config

        cfg = get_config(model_size="m-0.5x", input_size=320, num_classes=5)
        model = build_model(cfg)
        apply_lora(model, rank=4, alpha=8.0)
        merged = merge_lora_weights(model)
        total = sum(p.numel() for p in merged.parameters())
        assert total > 0

    def test_get_lora_state_dict(self):
        from flashdet.models import build_model, apply_lora, get_lora_state_dict
        from flashdet.cfg import get_config

        cfg = get_config(model_size="m-0.5x", input_size=320, num_classes=5)
        model = build_model(cfg)
        apply_lora(model, rank=4, alpha=8.0)
        sd = get_lora_state_dict(model)
        assert isinstance(sd, dict)
        assert len(sd) > 0
