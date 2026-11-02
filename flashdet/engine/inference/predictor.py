"""FlashDet Predictor — unified inference for all architectures.

Supports FlashDet (NMS-free) and YOLO family (with NMS).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from flashdet.data.transforms import InferenceTransform
from flashdet.models.detector import build_model
from flashdet.utils.checkpoint import safe_torch_load

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
        input_size: Optional[int] = None,
        class_names: Optional[List[str]] = None,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        # The checkpoint config overrides this when one is available.
        self.input_size = int(input_size) if input_size else 320
        self.class_names = class_names

        if model is not None:
            self.model = model.to(self.device).eval()
            self.num_classes = getattr(model, "num_classes", 80)
        elif model_path is not None:
            self.model, self.num_classes = self._load_model(model_path)
        else:
            raise ValueError("Must provide either model_path or model")

        # Must match the training/validation preprocessing exactly:
        # RGB, ImageNet mean/std, centred letterbox.
        self.transform = InferenceTransform(
            input_size=(self.input_size, self.input_size)
        )

    def _load_model(self, model_path: str):
        """Load model from checkpoint."""
        ckpt = safe_torch_load(model_path, map_location=self.device)

        backbone_type = "lite"
        neck_type = None
        if "config" in ckpt:
            cfg = ckpt["config"]
            arch = cfg.get("architecture", "flashdet")
            num_classes = cfg.get("num_classes", 80)
            model_size = cfg.get("model_size", "n")
            input_size = cfg.get("input_size", self.input_size)
            backbone_type = cfg.get("backbone_type") or "lite"
            neck_type = cfg.get("neck_type")
            if isinstance(input_size, (list, tuple)):
                input_size = input_size[0]
            self.input_size = input_size
        else:
            arch = "flashdet"
            num_classes = 80
            model_size = "n"

        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        # Backwards compatibility for old Pico-v2 checkpoints.
        if backbone_type == "lite" and isinstance(state_dict, dict):
            if any(k.startswith("backbone.stem.") or k.startswith("backbone.stages.")
                   for k in state_dict):
                backbone_type = "repnext"

        from flashdet.cfg import get_config
        config = get_config(num_classes=num_classes)
        config.model.architecture = arch
        config.model.size = model_size
        config.model.backbone_type = backbone_type
        config.model.neck_type = neck_type

        if arch in ("yolov8", "yolov9", "yolov10", "yolov11", "yolox"):
            config.model.width_mult = ckpt.get("config", {}).get("width_mult", 1.0)
            config.model.depth_mult = ckpt.get("config", {}).get("depth_mult", 1.0)

        model = build_model(config, architecture=arch)

        if isinstance(state_dict, dict) and not any(k.startswith("backbone") or k.startswith("stem") for k in state_dict):
            state_dict = ckpt
        model.load_state_dict(state_dict, strict=False)
        model = model.to(self.device).eval()

        if self.class_names is None:
            if "class_names" in ckpt:
                self.class_names = ckpt["class_names"]
            elif "config" in ckpt and "class_names" in ckpt["config"]:
                self.class_names = ckpt["config"]["class_names"]

        return model, num_classes

    # Reject decompression-bomb style inputs before resizing.  The model itself
    # is fixed-size, so a gigantic source image only wastes memory/CPU.
    MAX_INPUT_PIXELS = 64_000_000  # ~8000 x 8000

    def preprocess(self, image: np.ndarray) -> Tuple[torch.Tensor, Dict]:
        """Validate, letterbox, normalize and return a batch tensor + meta."""
        if not isinstance(image, np.ndarray):
            raise TypeError(
                f"Expected a numpy image array, got {type(image).__name__}"
            )
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Expected an HxWx3 BGR image, got shape {image.shape}"
            )
        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            raise ValueError(f"Image has invalid dimensions: {image.shape}")
        if h * w > self.MAX_INPUT_PIXELS:
            raise ValueError(
                f"Image is too large ({w}x{h}); max is "
                f"{self.MAX_INPUT_PIXELS} pixels"
            )
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor, meta = self.transform(rgb)
        tensor = torch.from_numpy(tensor).unsqueeze(0).to(self.device)
        return tensor, meta

    @staticmethod
    def _boxes_to_original(boxes: np.ndarray, meta: Dict) -> np.ndarray:
        """Map letterboxed input-space boxes back to original image pixels."""
        if len(boxes) == 0:
            return boxes
        inv_warp = np.linalg.inv(meta["warp_matrix"])
        n = len(boxes)
        xy = np.ones((n * 4, 3), dtype=np.float64)
        xy[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)
        xy = xy @ inv_warp.T
        xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)
        xs = xy[:, [0, 2, 4, 6]]
        ys = xy[:, [1, 3, 5, 7]]
        orig_w, orig_h = meta["original_size"]
        return np.stack([
            np.clip(xs.min(1), 0, orig_w - 1),
            np.clip(ys.min(1), 0, orig_h - 1),
            np.clip(xs.max(1), 0, orig_w - 1),
            np.clip(ys.max(1), 0, orig_h - 1),
        ], axis=1).astype(np.float32)

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

        tensor, meta = self.preprocess(image)

        if hasattr(self.model, "predict"):
            results = self.model.predict(
                tensor, score_thr=self.conf_thresh, nms_thr=self.nms_thresh
            )
            if results and len(results[0]) == 2:
                dets, labels = results[0]
                if dets.numel() == 0:
                    return []
                boxes = self._boxes_to_original(dets[:, :4].cpu().numpy(), meta)
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
                boxes = self._boxes_to_original(dets[:, :4].cpu().numpy(), meta)
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
