"""FlashDet Predictor — unified inference for all architectures.

Supports FlashDet (NMS-free) and YOLO family (with NMS).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from flashdet.models.detector import build_model

logger = logging.getLogger(__name__)


class Predictor:
    """High-level inference wrapper for all registered architectures.

    Example::

        from flashdet.engine.inference import Predictor

        pred = Predictor(model_path="workspace/model_best.pth")
        results = pred("image.jpg")
        for box, score, cls_id in results:
            print(f"class {cls_id}: {score:.2f} @ {box}")
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[torch.nn.Module] = None,
        device: str = "cuda",
        conf_thresh: float = 0.35,
        nms_thresh: float = 0.5,
        input_size: int = 640,
        class_names: Optional[List[str]] = None,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.input_size = input_size
        self.class_names = class_names

        if model is not None:
            self.model = model.to(self.device).eval()
            self.num_classes = getattr(model, "num_classes", 80)
        elif model_path is not None:
            self.model, self.num_classes = self._load_model(model_path)
        else:
            raise ValueError("Must provide either model_path or model")

    def _load_model(self, model_path: str):
        """Load model from checkpoint."""
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)

        if "config" in ckpt:
            cfg = ckpt["config"]
            arch = cfg.get("architecture", "flashdet")
            num_classes = cfg.get("num_classes", 80)
        else:
            arch = "flashdet"
            num_classes = 80

        from flashdet.cfg import get_config
        config = get_config(num_classes=num_classes)
        config.model.architecture = arch

        if arch in ("yolov8", "yolov9", "yolov10", "yolov11", "yolox"):
            config.model.width_mult = ckpt.get("config", {}).get("width_mult", 1.0)
            config.model.depth_mult = ckpt.get("config", {}).get("depth_mult", 1.0)

        model = build_model(config, architecture=arch)

        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        if isinstance(state_dict, dict) and not any(k.startswith("backbone") or k.startswith("stem") for k in state_dict):
            state_dict = ckpt
        model.load_state_dict(state_dict, strict=False)
        model = model.to(self.device).eval()

        if self.class_names is None and "class_names" in ckpt:
            self.class_names = ckpt["class_names"]

        return model, num_classes

    def preprocess(self, image: np.ndarray) -> Tuple[torch.Tensor, float, Tuple[int, int]]:
        """Resize and normalize image for inference."""
        h, w = image.shape[:2]
        scale = min(self.input_size / h, self.input_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(image, (new_w, new_h))

        padded = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w] = resized

        tensor = torch.from_numpy(padded).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)
        return tensor, scale, (h, w)

    @torch.no_grad()
    def __call__(self, source) -> List[Tuple[np.ndarray, float, int]]:
        """Run inference on an image path or numpy array.

        Returns:
            List of (bbox_xyxy, score, class_id) tuples.
        """
        if isinstance(source, (str, Path)):
            image = cv2.imread(str(source))
            if image is None:
                raise FileNotFoundError(f"Cannot read image: {source}")
        else:
            image = source

        tensor, scale, (orig_h, orig_w) = self.preprocess(image)

        if hasattr(self.model, "predict"):
            results = self.model.predict(
                tensor, score_thr=self.conf_thresh, nms_thr=self.nms_thresh
            )
            if results and len(results[0]) == 2:
                dets, labels = results[0]
                if dets.numel() == 0:
                    return []
                boxes = dets[:, :4].cpu().numpy() / scale
                scores = dets[:, 4].cpu().numpy()
                class_ids = labels.cpu().numpy()
                return [(boxes[i], float(scores[i]), int(class_ids[i]))
                        for i in range(len(scores))]

        out = self.model(tensor)
        if "preds" in out:
            from flashdet.engine.inference.postprocess import decode_yolo_predictions
            results = decode_yolo_predictions(
                out["preds"], self.num_classes, tensor.shape[2:],
                score_thr=self.conf_thresh, nms_thr=self.nms_thresh,
            )
            if results and len(results[0]) == 2:
                dets, labels = results[0]
                if dets.numel() == 0:
                    return []
                boxes = dets[:, :4].cpu().numpy() / scale
                scores = dets[:, 4].cpu().numpy()
                class_ids = labels.cpu().numpy()
                return [(boxes[i], float(scores[i]), int(class_ids[i]))
                        for i in range(len(scores))]

        return []

    def predict_image(self, image_path: str) -> List[Dict]:
        """Predict on a single image and return structured results."""
        results = self(image_path)
        output = []
        for bbox, score, cls_id in results:
            name = self.class_names[cls_id] if self.class_names and cls_id < len(self.class_names) else str(cls_id)
            output.append({
                "class_name": name,
                "class_id": cls_id,
                "confidence": score,
                "bbox": bbox.tolist() if hasattr(bbox, 'tolist') else list(bbox),
            })
        return output
