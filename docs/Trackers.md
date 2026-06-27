# Trackers

## Available Trackers

| Tracker | Method | Best For |
|---------|--------|----------|
| FlashTracker | IoU + Kalman filter | General purpose |
| MotionTracker | Kalman + Hungarian matching | Speed-critical |
| AppearanceTracker | Appearance + motion fusion | Crowded scenes |

## Usage

```python
from flashdet.trackers import FlashTracker

tracker = FlashTracker(max_age=30, min_hits=3, iou_threshold=0.3)

# Per frame
tracks = tracker.update(detections)  # [x1,y1,x2,y2,track_id,score,cls]
```

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| max_age | Frames before track deletion | 30 |
| min_hits | Minimum detections to confirm | 3 |
| iou_threshold | IoU threshold for matching | 0.3 |
