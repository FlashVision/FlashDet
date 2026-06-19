"""FlashDet Predictor — inference on images, video, or webcam."""

import os
import time
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from flashdet.cfg import get_config
from flashdet.models import FlashDet, load_coco_pretrained
from flashdet.data.transforms import InferenceTransform
from flashdet.utils import draw_detections

logger = logging.getLogger(__name__)


class Predictor:
    """High-level inference wrapper for FlashDet.

    Example::

        from flashdet import Predictor

        pred = Predictor(model_path="workspace/model_best_inference.pth")
        detections = pred.predict_image("test.jpg")
        for cls_name, score, x1, y1, x2, y2 in detections:
            print(f"{cls_name}: {score:.2f}")
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        conf_thresh: float = 0.35,
        nms_thresh: float = 0.4,
        pretrained_coco: bool = False,
        model_size: str = "m",
        input_size: int = 416,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

        MODEL_SIZE_MAP = {
            "m": {"backbone": "1.0x", "fpn_channels": 96},
            "m-1.5x": {"backbone": "1.5x", "fpn_channels": 128},
            "m-0.5x": {"backbone": "0.5x", "fpn_channels": 96},
        }

        if pretrained_coco:
            COCO_NAMES = [
                "person", "bicycle", "car", "motorcycle", "airplane", "bus",
                "train", "truck", "boat", "traffic light", "fire hydrant",
                "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                "skis", "snowboard", "sports ball", "kite", "baseball bat",
                "baseball glove", "skateboard", "surfboard", "tennis racket",
                "bottle", "wine glass", "cup", "fork", "knife", "spoon",
                "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                "carrot", "hot dog", "pizza", "donut", "cake", "chair",
                "couch", "potted plant", "bed", "dining table", "toilet",
                "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                "microwave", "oven", "toaster", "sink", "refrigerator", "book",
                "clock", "vase", "scissors", "teddy bear", "hair drier",
                "toothbrush",
            ]
            mcfg = MODEL_SIZE_MAP.get(model_size, MODEL_SIZE_MAP["m"])
            self.class_names = COCO_NAMES
            self.input_size = (input_size, input_size)

            self.model = FlashDet(
                num_classes=80,
                input_size=self.input_size,
                backbone_size=mcfg["backbone"],
                fpn_channels=mcfg["fpn_channels"],
                pretrained=False,
                use_aux_head=False,
            )
            load_coco_pretrained(
                self.model,
                backbone_size=mcfg["backbone"],
                fpn_channels=mcfg["fpn_channels"],
                input_size=input_size,
            )
            logger.info(f"COCO pretrained model loaded ({model_size}, {input_size}px)")
        else:
            if model_path is None:
                raise ValueError("Either model_path or pretrained_coco=True is required")

            config = get_config()
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

            backbone_size = config.model.backbone_size
            num_classes = config.model.num_classes
            fpn_channels = config.model.fpn_out_channels
            inp_size = config.model.input_size
            class_names = list(config.class_names)

            if "config" in checkpoint:
                ckpt_cfg = checkpoint["config"]
                backbone_size = ckpt_cfg.get("backbone_size", backbone_size)
                num_classes = ckpt_cfg.get("num_classes", num_classes)
                fpn_channels = ckpt_cfg.get("fpn_channels", fpn_channels)
                inp_size = ckpt_cfg.get("input_size", inp_size)
                if "class_names" in ckpt_cfg and ckpt_cfg["class_names"]:
                    class_names = ckpt_cfg["class_names"]

            self.class_names = class_names
            self.input_size = inp_size

            self.model = FlashDet(
                num_classes=num_classes,
                input_size=inp_size,
                backbone_size=backbone_size,
                fpn_channels=fpn_channels,
                pretrained=False,
                use_aux_head=False,
            )

            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            elif "state_dict" in checkpoint:
                sd = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
                self.model.load_state_dict(sd, strict=False)
            else:
                self.model.load_state_dict(checkpoint, strict=False)

        self.model = self.model.to(self.device).eval()
        self.transform = InferenceTransform(input_size=self.input_size)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Make Predictor callable for use with Solutions (ObjectCounter, etc.).

        Returns:
            Nx6 numpy array of [x1, y1, x2, y2, score, class_id].
        """
        detections = self.detect(image)
        if not detections:
            return np.empty((0, 6), dtype=np.float64)

        rows = []
        for cls_name, score, x1, y1, x2, y2 in detections:
            cls_id = self.class_names.index(cls_name) if cls_name in self.class_names else 0
            rows.append([x1, y1, x2, y2, score, cls_id])
        return np.array(rows, dtype=np.float64)

    @torch.no_grad()
    def detect(self, image: np.ndarray) -> List[Tuple[str, float, int, int, int, int]]:
        """Run detection on a BGR image.

        Returns:
            List of (class_name, score, x1, y1, x2, y2) tuples.
        """
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor, meta = self.transform(rgb)
        tensor = torch.from_numpy(tensor).unsqueeze(0).to(self.device)

        results = self.model.predict(tensor, None, self.conf_thresh, self.nms_thresh)

        warp_matrix = meta["warp_matrix"]
        inv_warp = np.linalg.inv(warp_matrix)

        detections = []
        if results and len(results[0]) > 0:
            dets, labels = results[0]
            boxes_np = dets[:, :4].cpu().numpy()
            scores_np = dets[:, 4].cpu().numpy()

            n = len(boxes_np)
            if n > 0:
                xy = np.ones((n * 4, 3))
                xy[:, :2] = boxes_np[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)
                xy = xy @ inv_warp.T
                xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)
                xs = xy[:, [0, 2, 4, 6]]
                ys = xy[:, [1, 3, 5, 7]]
                x1s = np.clip(xs.min(1), 0, w - 1).astype(int)
                y1s = np.clip(ys.min(1), 0, h - 1).astype(int)
                x2s = np.clip(xs.max(1), 0, w - 1).astype(int)
                y2s = np.clip(ys.max(1), 0, h - 1).astype(int)

                for i in range(n):
                    cls_name = self.class_names[int(labels[i].cpu().item())]
                    detections.append((
                        cls_name, float(scores_np[i]),
                        int(x1s[i]), int(y1s[i]), int(x2s[i]), int(y2s[i]),
                    ))

        return detections

    def predict(self, source, output_dir: Optional[str] = None) -> List:
        """Run detection on an image path, numpy array, or directory.

        Args:
            source: Path to image/directory, or a BGR numpy array.
            output_dir: If set, saves annotated output here.

        Returns:
            List of detections (class_name, score, x1, y1, x2, y2).
        """
        if isinstance(source, np.ndarray):
            return self.detect(source)
        source = str(source)
        if os.path.isdir(source):
            return self.predict_directory(source, output_dir or "output")
        return self.predict_image(source, output_dir)

    def predict_image(self, image_path: str, output_dir: Optional[str] = None) -> List:
        """Run detection on an image file. Optionally saves annotated output."""
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        detections = self.detect(image)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output = draw_detections(image, detections)
            out_path = os.path.join(output_dir, Path(image_path).name)
            cv2.imwrite(out_path, output)
            logger.info(f"Saved: {out_path}")

        return detections

    def predict_directory(self, dir_path: str, output_dir: str = "output") -> List:
        """Run detection on all images in a directory."""
        all_detections = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for path in Path(dir_path).glob(ext):
                dets = self.predict_image(str(path), output_dir)
                all_detections.append((str(path), dets))
        return all_detections

    def predict_video(
        self,
        video_path: str,
        output_dir: str = "output",
        show: bool = False,
    ) -> str:
        """Process video file. Returns path to output video."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, Path(video_path).name)
        writer = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height),
        )

        frame_count = 0
        total_time = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            start = time.time()
            detections = self.detect(frame)
            total_time += time.time() - start

            output = draw_detections(frame, detections)
            current_fps = frame_count / total_time if total_time > 0 else 0
            cv2.putText(output, f"FPS: {current_fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            writer.write(output)
            frame_count += 1

            if show:
                cv2.imshow("FlashDet", output)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if frame_count % 100 == 0:
                logger.info(f"  {frame_count}/{total} frames processed")

        cap.release()
        writer.release()
        if show:
            cv2.destroyAllWindows()

        avg_fps = frame_count / total_time if total_time > 0 else 0
        logger.info(f"Video processed: {avg_fps:.1f} FPS, saved to {output_path}")
        return output_path

    def predict_camera(self, camera_id: int = 0, output_dir: Optional[str] = None):
        """Run live detection from webcam. Press 'q' to quit."""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        frame_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = self.detect(frame)
            output = draw_detections(frame, detections)

            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(output, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("FlashDet — Press Q to quit", output)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s") and output_dir:
                save_path = os.path.join(output_dir, f"capture_{frame_count}.jpg")
                cv2.imwrite(save_path, output)

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()
