"""
Multi-Object Tracking
======================

Track objects across video frames with persistent IDs.
Uses ByteTracker (IoU + Kalman filter) for fast, accurate tracking.
"""

import cv2
import numpy as np
from flashdet import Predictor
from flashdet.trackers import ByteTracker

predictor = Predictor(model_path="workspace/my_model/best.pth", device="cuda")
tracker = ByteTracker(max_age=30, min_hits=3, iou_threshold=0.3)

cap = cv2.VideoCapture("video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    detections = predictor.detect(frame)

    det_array = np.array([
        [x1, y1, x2, y2, score, 0]
        for _, score, x1, y1, x2, y2 in detections
    ], dtype=np.float32).reshape(-1, 6)

    tracks = tracker.update(det_array)

    for track in tracks:
        x1, y1, x2, y2, track_id, score, cls = track
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID:{int(track_id)}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
