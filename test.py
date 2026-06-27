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

    # Video with tracking
    python test.py --model checkpoint.pth --video test.mp4 --tracker bytetrack
    python test.py --model checkpoint.pth --video test.mp4 --tracker deepsort --max-age 50

    # Video with a solution (auto-creates tracker if not specified)
    python test.py --model checkpoint.pth --video test.mp4 --solution heatmap
    python test.py --model checkpoint.pth --video test.mp4 --solution counter --tracker ocsort

    # Evaluate on validation set with GT comparison visualizations
    python test.py --model checkpoint.pth --eval --output workspace/eval_vis/

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
from flashdet.models import FlashDet
from flashdet.models.detector import build_model
from flashdet.data.transforms import InferenceTransform
from flashdet.utils import draw_detections, load_checkpoint
from flashdet.utils.visualization import make_gt_pred_panel, draw_boxes, make_color_palette


# ──────────────────────────────────────────────────────
#  Tracker factory
# ──────────────────────────────────────────────────────

def _create_tracker(name, max_age=30, min_hits=3, iou_threshold=0.3):
    """Create a tracker by name."""
    from flashdet.trackers import (
        SortTracker, ByteTracker, BoTSortTracker,
        DeepSortTracker, OCSortTracker, StrongSortTracker,
    )
    tracker_map = {
        "sort": SortTracker,
        "bytetrack": ByteTracker,
        "botsort": BoTSortTracker,
        "deepsort": DeepSortTracker,
        "ocsort": OCSortTracker,
        "strongsort": StrongSortTracker,
    }
    cls = tracker_map[name]
    return cls(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)


# ──────────────────────────────────────────────────────
#  Solution factory
# ──────────────────────────────────────────────────────

def _create_solution(name, predictor, tracker):
    """Create a solution by name."""
    from flashdet.solutions import (
        ObjectCounter, Heatmap, SpeedEstimator, DistanceCalculator,
        RegionCounter, QueueManager, ParkingManager, SecurityAlarm,
        WorkoutMonitor, ObjectBlurrer, ObjectCropper, CrowdDensity,
        DwellTimeAnalyzer, TrafficFlow, TrajectoryVisualizer,
    )
    solution_map = {
        "counter": ObjectCounter,
        "heatmap": Heatmap,
        "speed": SpeedEstimator,
        "distance": DistanceCalculator,
        "region": RegionCounter,
        "queue": QueueManager,
        "parking": ParkingManager,
        "security": SecurityAlarm,
        "workout": WorkoutMonitor,
        "blur": ObjectBlurrer,
        "crop": ObjectCropper,
        "crowd": CrowdDensity,
        "dwell": DwellTimeAnalyzer,
        "traffic": TrafficFlow,
        "trajectory": TrajectoryVisualizer,
    }
    cls = solution_map[name]
    return cls(predictor=predictor, tracker=tracker)


# ──────────────────────────────────────────────────────
#  Tracking visualization helpers
# ──────────────────────────────────────────────────────

_TRACK_COLORS = {}
_TRACK_TRAILS = {}
_TRAIL_LENGTH = 30


def _get_track_color(track_id):
    """Return a consistent BGR color for a track ID."""
    if track_id not in _TRACK_COLORS:
        rng = np.random.RandomState(int(track_id) * 7 + 13)
        _TRACK_COLORS[track_id] = tuple(int(c) for c in rng.randint(60, 255, 3))
    return _TRACK_COLORS[track_id]


def _draw_tracks(frame, tracks, class_names=None):
    """Draw tracked boxes with IDs and trails.

    Parameters
    ----------
    frame : np.ndarray
        BGR image to draw on (modified in place).
    tracks : np.ndarray
        Mx7 array ``[x1, y1, x2, y2, track_id, score, class_id]``.
    class_names : list[str] | None
        Class names for label display.
    """
    for trk in tracks:
        x1, y1, x2, y2 = int(trk[0]), int(trk[1]), int(trk[2]), int(trk[3])
        track_id = int(trk[4])
        score = float(trk[5])
        cls_id = int(trk[6])
        color = _get_track_color(track_id)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cls_label = class_names[cls_id] if class_names and cls_id < len(class_names) else f"cls{cls_id}"
        label = f"{cls_label} #{track_id} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if track_id not in _TRACK_TRAILS:
            _TRACK_TRAILS[track_id] = []
        _TRACK_TRAILS[track_id].append((cx, cy))
        if len(_TRACK_TRAILS[track_id]) > _TRAIL_LENGTH:
            _TRACK_TRAILS[track_id] = _TRACK_TRAILS[track_id][-_TRAIL_LENGTH:]

        pts = _TRACK_TRAILS[track_id]
        if len(pts) > 1:
            for i in range(1, len(pts)):
                thickness = max(1, int(i * 2 / len(pts)))
                cv2.line(frame, pts[i - 1], pts[i], color, thickness)

    return frame


class FlashDetDetector:
    """Inference wrapper for FlashDet models.

    Class names are read from the checkpoint's embedded 'config' dict so
    that models trained on any dataset always display the correct labels.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        conf_thresh: float = 0.35,
        nms_thresh: float = 0.4,
        input_size: int = None,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.class_filter = None

        config = get_config()

        if model_path is None:
            raise ValueError("--model is required")

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        num_classes   = config.model.num_classes
        inp_size      = config.model.input_size
        class_names   = list(config.class_names)
        arch = "flashdet"
        ckpt_model_size = "n"

        if "config" in checkpoint:
            ckpt_cfg = checkpoint["config"]
            num_classes   = ckpt_cfg.get("num_classes", num_classes)
            inp_size      = ckpt_cfg.get("input_size", inp_size)
            arch          = ckpt_cfg.get("architecture", "flashdet")
            ckpt_model_size = ckpt_cfg.get("model_size", "n")
            if "class_names" in ckpt_cfg and ckpt_cfg["class_names"]:
                class_names = ckpt_cfg["class_names"]
            print(f"Detected from checkpoint: arch={arch}, size={ckpt_model_size}, classes={num_classes}")

        if len(class_names) != num_classes:
            print(
                f"[WARN] class_names ({len(class_names)}) != num_classes ({num_classes}). "
                "Falling back to generic labels."
            )
            class_names = [f"class_{i}" for i in range(num_classes)]

        self.CLASS_NAMES = class_names
        self.input_size = input_size if input_size is not None else inp_size

        print(f"Loading model: {model_path}")

        sd_for_load = (
            checkpoint.get("model_state_dict")
            or checkpoint.get("state_dict")
            or checkpoint
        )
        backbone_type = "lite"
        if "config" in checkpoint:
            backbone_type = checkpoint["config"].get("backbone_type", "lite")
        if backbone_type == "lite":
            sd_keys = list(sd_for_load.keys())
            if any(k.startswith("backbone.stem.") or k.startswith("backbone.stages.") for k in sd_keys):
                backbone_type = "repnext"

        arch = arch.lower()
        if arch in ("flashdet", ""):
            self.model = FlashDet(
                num_classes=num_classes,
                size=ckpt_model_size,
                backbone_type=backbone_type,
            )
        else:
            config.model.num_classes = num_classes
            self.model = build_model(config, architecture=arch)

        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        elif "state_dict" in checkpoint:
            sd = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
            self.model.load_state_dict(sd, strict=False)
        else:
            self.model.load_state_dict(checkpoint, strict=False)

        self.model = self.model.to(self.device).eval()
        self.transform = InferenceTransform(input_size=self.input_size)

        if hasattr(self.model, "get_model_info"):
            info = self.model.get_model_info()
            print(f"Device: {self.device}")
            print(f"Model: {info['name']}  Params: {info['total_params']:,}")
        else:
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"Device: {self.device}")
            print(f"Model: {arch}  Params: {total_params:,}")

    @torch.no_grad()
    def detect(self, image: np.ndarray):
        """Run detection on a BGR image.

        Returns list of ``(class_name, score, x1, y1, x2, y2)``.
        """
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor, meta = self.transform(rgb)
        tensor = torch.from_numpy(tensor).unsqueeze(0).to(self.device)

        results = self.model.predict(tensor, None, score_thr=self.conf_thresh)

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

        if self.class_filter:
            detections = [d for d in detections if d[0] in self.class_filter]

        return detections



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


def _tracker_accepts_frame(tracker):
    """Check if tracker.update() accepts a frame argument (appearance-based trackers)."""
    import inspect
    sig = inspect.signature(tracker.update)
    params = list(sig.parameters.keys())
    return len(params) >= 3 or "frame" in params


def process_video(detector, video_path, output_dir, show=False, tracker=None, solution=None):
    """Process video file with optional tracking or solution."""
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

        if solution is not None:
            output, _ = solution.process_frame(frame)
        elif tracker is not None:
            detections = detector.detect(frame)
            det_array = np.array(
                [[d[2], d[3], d[4], d[5], d[1],
                  detector.CLASS_NAMES.index(d[0]) if d[0] in detector.CLASS_NAMES else 0]
                 for d in detections],
                dtype=np.float64,
            ) if detections else np.empty((0, 6), dtype=np.float64)
            tracks = tracker.update(det_array, frame) if _tracker_accepts_frame(tracker) else tracker.update(det_array)
            output = _draw_tracks(frame.copy(), tracks, detector.CLASS_NAMES)
        else:
            detections = detector.detect(frame)
            output = draw_detections(frame, detections)

        total_time += time.time() - start

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


def process_camera(detector, camera_id, output_dir=None, tracker=None, solution=None):
    """Process live camera feed with optional tracking or solution."""
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

        if solution is not None:
            output, _ = solution.process_frame(frame)
        elif tracker is not None:
            detections = detector.detect(frame)
            det_array = np.array(
                [[d[2], d[3], d[4], d[5], d[1],
                  detector.CLASS_NAMES.index(d[0]) if d[0] in detector.CLASS_NAMES else 0]
                 for d in detections],
                dtype=np.float64,
            ) if detections else np.empty((0, 6), dtype=np.float64)
            tracks = tracker.update(det_array, frame) if _tracker_accepts_frame(tracker) else tracker.update(det_array)
            output = _draw_tracks(frame.copy(), tracks, detector.CLASS_NAMES)
        else:
            detections = detector.detect(frame)
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

    parser.add_argument("--tracker", default=None,
        choices=["sort", "bytetrack", "botsort", "deepsort", "ocsort", "strongsort"],
        help="Enable tracking with specified algorithm (for video/camera)")
    parser.add_argument("--max-age", type=int, default=30,
        help="Tracker: max frames to keep lost tracks")
    parser.add_argument("--min-hits", type=int, default=3,
        help="Tracker: min detections before track confirmed")
    parser.add_argument("--iou-threshold", type=float, default=0.3,
        help="Tracker: IoU matching threshold")

    parser.add_argument("--solution", default=None,
        choices=["counter", "heatmap", "speed", "distance", "region", "queue",
                 "parking", "security", "workout", "blur", "crop", "crowd",
                 "dwell", "traffic", "trajectory"],
        help="Run a solution (for video/camera). Requires --tracker unless solution is detection-only.")
    parser.add_argument("--classes", nargs="+", default=None,
        help="Filter classes (e.g. --classes person car)")
    parser.add_argument("--input-size", type=int, default=None,
        help="Override input size from checkpoint")

    args = parser.parse_args()

    if not any([args.image, args.video, args.camera is not None, args.eval]):
        parser.error("Specify --image, --video, --camera, or --eval")

    if args.model is None:
        parser.error("--model is required")

    detector = FlashDetDetector(
        model_path=args.model,
        device=args.device,
        conf_thresh=args.conf,
        nms_thresh=args.nms,
        input_size=args.input_size,
    )

    if args.classes:
        detector.class_filter = set(args.classes)

    tracker = None
    if args.tracker and (args.video or args.camera is not None):
        tracker = _create_tracker(args.tracker, args.max_age, args.min_hits, args.iou_threshold)

    solution = None
    if args.solution and (args.video or args.camera is not None):
        from flashdet.engine.inference import Predictor
        predictor = Predictor(
            model_path=args.model, device=args.device,
            conf_thresh=args.conf, nms_thresh=args.nms,
        )
        if tracker is None:
            tracker = _create_tracker("sort", args.max_age, args.min_hits, args.iou_threshold)
        solution = _create_solution(args.solution, predictor, tracker)

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
        process_video(detector, args.video, args.output, args.show,
                      tracker=tracker, solution=solution)
    elif args.camera is not None:
        process_camera(detector, args.camera, args.output,
                       tracker=tracker, solution=solution)


if __name__ == "__main__":
    main()
