#!/usr/bin/env python3
"""
Test/Inference for FlashDet.

Usage:
    # Single image / directory
    python test.py --model checkpoint.pth --image test.jpg
    python test.py --model checkpoint.pth --image data/coco/val/

    # Video / camera
    python test.py --model checkpoint.pth --video test.mp4
    python test.py --model checkpoint.pth --camera 0

    # Evaluate on validation set with GT comparison visualizations
    python test.py --model checkpoint.pth --eval --output workspace/eval_vis/

    # Quick test with COCO pretrained weights (no training needed)
    python test.py --pretrained-coco --image test.jpg --model-size m --input-size 416
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from flashdet.cfg import get_config
from flashdet.models import FlashDet, load_coco_pretrained
from flashdet.data.transforms import InferenceTransform
from flashdet.utils import draw_detections, load_checkpoint
from flashdet.utils.visualization import make_gt_pred_panel, draw_boxes, make_color_palette


class FlashDetDetector:
    """FlashDet inference wrapper.

    Class names are read from the checkpoint's embedded 'config' dict so that
    models trained on any dataset (PPE, Indoor Objects, custom) always display
    the correct labels without any code changes.
    """

    def __init__(
        self,
        model_path: str = None,
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

        config = get_config()

        MODEL_SIZE_MAP = {
            "m":     {"backbone": "1.0x", "fpn_channels": 96},
            "m-1.5x": {"backbone": "1.5x", "fpn_channels": 128},
            "m-0.5x": {"backbone": "0.5x", "fpn_channels": 96},
        }

        if pretrained_coco:
            # COCO pretrained mode — 80 classes, no custom checkpoint needed
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
            self.CLASS_NAMES = COCO_NAMES
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
            print(f"COCO pretrained model loaded ({model_size}, {input_size}px, 80 classes)")

        else:
            # Load from a user-trained checkpoint
            if model_path is None:
                raise ValueError("Either --model or --pretrained-coco is required")

            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

            backbone_size = config.model.backbone_size
            num_classes   = config.model.num_classes
            fpn_channels  = config.model.fpn_out_channels
            inp_size      = config.model.input_size
            class_names   = list(config.class_names)

            if "config" in checkpoint:
                ckpt_cfg = checkpoint["config"]
                backbone_size = ckpt_cfg.get("backbone_size", backbone_size)
                num_classes   = ckpt_cfg.get("num_classes", num_classes)
                fpn_channels  = ckpt_cfg.get("fpn_channels", fpn_channels)
                inp_size      = ckpt_cfg.get("input_size", inp_size)
                if "class_names" in ckpt_cfg and ckpt_cfg["class_names"]:
                    class_names = ckpt_cfg["class_names"]
                print(f"Detected from checkpoint: backbone={backbone_size}, classes={num_classes}")

            if len(class_names) != num_classes:
                print(
                    f"[WARN] class_names ({len(class_names)}) != num_classes ({num_classes}). "
                    "Falling back to generic labels."
                )
                class_names = [f"class_{i}" for i in range(num_classes)]

            self.CLASS_NAMES = class_names
            self.input_size = inp_size

            print(f"Loading model: {model_path}")

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

        info = self.model.get_model_info()
        print(f"Device: {self.device}")
        print(f"Model: {info['name']}  Params: {info['total_params']:,}")

    @torch.no_grad()
    def detect(self, image: np.ndarray):
        """Run detection on a BGR image.

        Returns list of ``(class_name, score, x1, y1, x2, y2)``.
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
                    cls_name = self.CLASS_NAMES[int(labels[i].cpu().item())]
                    detections.append((
                        cls_name, float(scores_np[i]),
                        x1s[i], y1s[i], x2s[i], y2s[i]
                    ))

        return detections

    @staticmethod
    def count_violations(detections, violation_classes=None, safe_classes=None):
        if violation_classes is None:
            violation_classes = ["NO-Hardhat", "NO-Mask", "NO-Safety Vest"]
        if safe_classes is None:
            safe_classes = ["Hardhat", "Mask", "Safety Vest"]
        violations, safe = [], []
        for det in detections:
            if det[0] in violation_classes:
                violations.append(det)
            elif det[0] in safe_classes:
                safe.append(det)
        return violations, safe


# ──────────────────────────────────────────────────────
#  Processing helpers
# ──────────────────────────────────────────────────────

def process_image(detector, image_path, output_dir):
    """Process single image."""
    print(f"\nProcessing: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read {image_path}")
        return

    start = time.time()
    detections = detector.detect(image)
    elapsed = (time.time() - start) * 1000

    print(f"  Inference: {elapsed:.1f}ms  |  Detections: {len(detections)}")
    for d in detections:
        print(f"    {d[0]:20s}  {d[1]:.2f}  [{d[2]},{d[3]},{d[4]},{d[5]}]")

    output = draw_detections(image, detections)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, Path(image_path).name)
    cv2.imwrite(output_path, output)
    print(f"  Saved: {output_path}")


def process_eval(detector, output_dir):
    """Generate GT-vs-Predictions panels on the **validation** set.

    Reads the COCO annotation file, loads each validation image, runs
    inference, and saves a side-by-side panel.
    """
    config = get_config()
    ann_file = config.data.val_annotations
    img_dir = config.data.val_images
    if not os.path.isfile(ann_file):
        print(f"Cannot find annotation file: {ann_file}")
        sys.exit(1)

    with open(ann_file) as f:
        coco = json.load(f)

    cats = sorted(coco["categories"], key=lambda c: c["id"])
    cat_id_to_idx = {c["id"]: i for i, c in enumerate(cats)}
    class_names = [c["name"] for c in cats]
    colors = make_color_palette(len(class_names))
    color_map = {class_names[i]: colors[i] for i in range(len(class_names))}

    img_id_to_anns = {}
    for ann in coco["annotations"]:
        img_id_to_anns.setdefault(ann["image_id"], []).append(ann)

    os.makedirs(output_dir, exist_ok=True)
    images = coco["images"]
    total = len(images)
    print(f"\nEvaluating {total} validation images → {output_dir}")

    for idx, img_info in enumerate(images):
        fname = img_info["file_name"]
        img_path = os.path.join(img_dir, fname)
        image = cv2.imread(img_path)
        if image is None:
            continue

        detections = detector.detect(image)

        # GT boxes
        anns = img_id_to_anns.get(img_info["id"], [])
        gt_boxes = np.array(
            [[a["bbox"][0], a["bbox"][1],
              a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
             for a in anns], dtype=np.float32
        ).reshape(-1, 4)
        gt_labels = np.array(
            [cat_id_to_idx[a["category_id"]] for a in anns], dtype=int
        )

        # Pred arrays
        if detections:
            pred_boxes = np.array([[d[2], d[3], d[4], d[5]] for d in detections], dtype=np.float32)
            pred_scores = np.array([d[1] for d in detections], dtype=np.float32)
            pred_labels = np.array(
                [class_names.index(d[0]) if d[0] in class_names else 0 for d in detections], dtype=int
            )
        else:
            pred_boxes = np.empty((0, 4), dtype=np.float32)
            pred_scores = np.empty(0)
            pred_labels = np.empty(0, dtype=int)

        panel = make_gt_pred_panel(
            image, gt_boxes, gt_labels,
            pred_boxes, pred_labels, pred_scores,
            class_names=class_names,
            colors=color_map,
            title_extra=f"| {fname}",
        )

        stem = Path(fname).stem
        out_path = os.path.join(output_dir, f"{stem}.jpg")
        cv2.imwrite(out_path, panel, [cv2.IMWRITE_JPEG_QUALITY, 92])

        if (idx + 1) % 50 == 0 or idx + 1 == total:
            print(f"  [{idx+1}/{total}] saved")

    print(f"Done — {total} panels saved to {output_dir}")


def process_video(detector, video_path, output_dir, show=False):
    """Process video file."""
    print(f"\nProcessing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  {width}x{height} @ {fps}fps, {total} frames")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, Path(video_path).name)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_count = 0
    total_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()
        detections = detector.detect(frame)
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
            print(f"  {frame_count}/{total} frames ...")

    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()

    avg_fps = frame_count / total_time if total_time > 0 else 0
    print(f"  Average FPS: {avg_fps:.1f}  |  Saved: {output_path}")


def process_camera(detector, camera_id, output_dir=None):
    """Process live camera feed."""
    print(f"\nStarting camera: {camera_id}")
    print("Press 'q' to quit, 's' to screenshot")

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        output = draw_detections(frame, detections)

        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(output, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        violations, _ = detector.count_violations(detections)
        if violations:
            cv2.putText(output, f"VIOLATIONS: {len(violations)}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("FlashDet — Press Q to quit", output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s") and output_dir:
            save_path = os.path.join(output_dir, f"capture_{frame_count}.jpg")
            cv2.imwrite(save_path, output)
            print(f"Saved: {save_path}")

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


# ──────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FlashDet Inference / Evaluation")
    parser.add_argument("--model", "-m", default=None, help="Model checkpoint path")
    parser.add_argument("--image", "-i", help="Input image or directory")
    parser.add_argument("--video", "-v", help="Input video")
    parser.add_argument("--camera", type=int, help="Camera ID")
    parser.add_argument("--eval", action="store_true",
                        help="Run GT-vs-Pred evaluation on the validation set")
    parser.add_argument("--output", "-o", default="output", help="Output directory")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--nms", type=float, default=0.4, help="NMS IoU threshold")
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--show", action="store_true", help="Show output window")
    parser.add_argument("--pretrained-coco", action="store_true",
                        help="Use official COCO pretrained weights (80 classes, no fine-tuning)")
    parser.add_argument("--model-size", default="m", choices=["m", "m-1.5x", "m-0.5x"],
                        help="Model size (only used with --pretrained-coco)")
    parser.add_argument("--input-size", type=int, default=416,
                        help="Input resolution (only used with --pretrained-coco)")
    args = parser.parse_args()

    if not any([args.image, args.video, args.camera is not None, args.eval]):
        parser.error("Specify --image, --video, --camera, or --eval")

    if not args.pretrained_coco and args.model is None:
        parser.error("Either --model or --pretrained-coco is required")

    detector = FlashDetDetector(
        model_path=args.model,
        device=args.device,
        conf_thresh=args.conf,
        nms_thresh=args.nms,
        pretrained_coco=args.pretrained_coco,
        model_size=args.model_size,
        input_size=args.input_size,
    )

    if args.eval:
        process_eval(detector, args.output)
    elif args.image:
        if os.path.isdir(args.image):
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                for path in Path(args.image).glob(ext):
                    process_image(detector, str(path), args.output)
        else:
            process_image(detector, args.image, args.output)
    elif args.video:
        process_video(detector, args.video, args.output, args.show)
    elif args.camera is not None:
        process_camera(detector, args.camera, args.output)


if __name__ == "__main__":
    main()
