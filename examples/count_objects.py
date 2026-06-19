"""
Object Counting
================

Count objects crossing a virtual line in a video stream.
Useful for traffic counting, footfall analysis, etc.
"""

import cv2
from flashdet import Predictor
from flashdet.solutions import ObjectCounter
from flashdet.trackers import ByteTracker

predictor = Predictor(model_path="workspace/my_model/best.pth", device="cuda")
tracker = ByteTracker()

counter = ObjectCounter(
    predictor=predictor,
    tracker=tracker,
    line_points=[(100, 400), (600, 400)],
)

cap = cv2.VideoCapture("video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    annotated, results = counter.process_frame(frame)

    cv2.imshow("Counter", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

counts = counter.get_results()
print(f"Total counted: UP={counts['up']}, DOWN={counts['down']}")

cap.release()
cv2.destroyAllWindows()
