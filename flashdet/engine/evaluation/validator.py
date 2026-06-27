"""FlashDet Validator — compute mAP metrics on a validation set."""

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from flashdet.cfg import get_config
from flashdet.models import FlashDet
from flashdet.data import create_dataloader
from flashdet.utils import AverageMeter
from flashdet.utils.metrics import compute_map

logger = logging.getLogger(__name__)


class Validator:
    """Validate a FlashDet model on a dataset with mAP computation.

    Example::

        from flashdet import Validator

        val = Validator(model_path="workspace/checkpoint_best.pth")
        results = val.validate()
        print(f"mAP@0.5: {results['mAP']:.4f}")
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[nn.Module] = None,
        device: str = "cuda",
        batch_size: int = 32,
        workers: int = 4,
        input_size: int = 320,
        conf_thresh: float = 0.05,
        nms_thresh: float = 0.6,
        iou_threshold: float = 0.5,
        val_images: Optional[str] = None,
        val_annotations: Optional[str] = None,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.workers = workers
        self.input_size = (input_size, input_size)
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.iou_threshold = iou_threshold

        cfg = get_config()
        self.val_images = val_images or cfg.data.val_images
        self.val_annotations = val_annotations or cfg.data.val_annotations

        if model is not None:
            self.model = model.to(self.device)
            self.class_names = list(cfg.class_names)
        elif model_path is not None:
            self.model, self.class_names = self._load_model(model_path, cfg)
        else:
            raise ValueError("Either model_path or model must be provided")

    def _load_model(self, model_path: str, cfg):
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        num_classes = cfg.model.num_classes
        class_names = list(cfg.class_names)
        size = getattr(cfg.model, "size", "n")

        if "config" in checkpoint:
            ckpt_cfg = checkpoint["config"]
            num_classes = ckpt_cfg.get("num_classes", num_classes)
            size = ckpt_cfg.get("model_size", ckpt_cfg.get("size", size))
            if "class_names" in ckpt_cfg and ckpt_cfg["class_names"]:
                class_names = ckpt_cfg["class_names"]

        model = FlashDet(
            num_classes=num_classes,
            size=size,
        )

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        elif "state_dict" in checkpoint:
            sd = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
            model.load_state_dict(sd, strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

        model = model.to(self.device).eval()
        return model, class_names

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation and return mAP metrics.

        Returns:
            Dict with keys: mAP, val_loss, AP_per_class
        """
        self.model.eval()

        val_loader = create_dataloader(
            img_dir=self.val_images,
            ann_file=self.val_annotations,
            batch_size=self.batch_size,
            input_size=self.input_size,
            num_workers=self.workers,
            is_train=False,
        )

        loss_meter = AverageMeter("Loss")
        all_preds: List[Dict] = []
        all_gts: List[Dict] = []

        for images, gt_meta in val_loader:
            images = images.to(self.device)

            try:
                out = self.model(images, gt_meta, epoch=0, compute_loss=True)
                loss_meter.update(out["loss"].item())
            except Exception:
                pass

            results = self.model.predict(
                images, None,
                score_thr=self.conf_thresh,
                nms_thr=self.nms_thresh,
            )

            for i, (dets, lbs) in enumerate(results):
                gt_boxes = gt_meta["gt_bboxes"][i]
                gt_labels = gt_meta["gt_labels"][i]

                if dets is not None and dets.numel() > 0:
                    boxes_np = dets[:, :4].cpu().numpy()
                    scores_np = dets[:, 4].cpu().numpy()
                    lbs_np = lbs.cpu().numpy()
                else:
                    boxes_np = np.zeros((0, 4), dtype=np.float32)
                    scores_np = np.zeros(0, dtype=np.float32)
                    lbs_np = np.zeros(0, dtype=np.int64)

                all_preds.append({"boxes": boxes_np, "scores": scores_np, "labels": lbs_np})
                all_gts.append({"boxes": gt_boxes, "labels": gt_labels})

        num_cls = len(self.class_names)
        map_results = compute_map(
            all_preds, all_gts,
            iou_threshold=self.iou_threshold,
            num_classes=num_cls,
        )

        result = {
            "mAP": map_results["mAP"],
            "val_loss": loss_meter.avg,
            "AP_per_class": map_results.get("AP_per_class", {}),
        }

        logger.info(f"Validation: mAP@{self.iou_threshold:.2f} = {result['mAP']:.4f}, "
                    f"Loss = {result['val_loss']:.4f}")

        ap_per_cls = result["AP_per_class"]
        if self.class_names and ap_per_cls:
            for cid, v in sorted(ap_per_cls.items()):
                if cid < len(self.class_names):
                    logger.info(f"  {self.class_names[cid]}: AP={v:.3f}")

        return result
