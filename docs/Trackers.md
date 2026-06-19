# Trackers

## Available Trackers

| Tracker | Method | Best For |
|---------|--------|----------|
| ByteTracker | IoU + Kalman filter | General purpose |
| SORTTracker | Kalman + Hungarian | Speed-critical |
| BoTSORT | Appearance + motion | Crowded scenes |

## Usage

```python
from flashdet.trackers import ByteTracker

tracker = ByteTracker(max_age=30, min_hits=3, iou_threshold=0.3)

# Per frame
tracks = tracker.update(detections)  # [x1,y1,x2,y2,track_id,score,cls]
```

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| max_age | Frames before track deletion | 30 |
| min_hits | Minimum detections to confirm | 3 |
| iou_threshold | IoU threshold for matching | 0.3 |
