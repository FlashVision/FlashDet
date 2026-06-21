"""Tests for new FlashDet components: OBB head, losses, augmentations."""

import pytest
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

    def test_decode_angle(self):
        from flashdet.models.head.obb_head import OBBHead

        head = OBBHead(num_classes=5, in_channels=32, feat_channels=32, strides=[8])
        angle_pred = torch.randn(10, 1)
        decoded = head.decode_angle(angle_pred)
        assert decoded.shape == (10,)
        assert (decoded.abs() <= torch.pi / 2).all()

    def test_rotated_iou(self):
        from flashdet.models.head.obb_head import rotated_iou

        boxes = torch.tensor([[50, 50, 20, 20, 0.0]])
        iou = rotated_iou(boxes, boxes)
        assert iou.shape == (1,)
        assert iou[0] > 0.9

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
