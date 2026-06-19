# Solutions

## Object Counter

Count objects crossing lines or entering regions:

```python
from flashdet.solutions import ObjectCounter

counter = ObjectCounter(predictor, tracker, line_points=[(100, 300), (500, 300)])
count = counter.process_frame(frame)
```

## Speed Estimator

```python
from flashdet.solutions import SpeedEstimator

estimator = SpeedEstimator(predictor, tracker, pixels_per_meter=8.0)
speeds = estimator.process_frame(frame)
```

## Heatmap

```python
from flashdet.solutions import Heatmap

heatmap = Heatmap(predictor, decay=0.95)
overlay = heatmap.process_frame(frame)
```

## All Solutions

| Solution | Description |
|----------|-------------|
| ObjectCounter | Count objects crossing lines |
| SpeedEstimator | Estimate real-world speed |
| Heatmap | Detection density visualization |
| RegionCounter | Count objects in zones |
| QueueManager | Monitor queue lengths |
| DistanceCalculator | Measure distances |
| ParkingManager | Track parking occupancy |
| SecurityAlarm | Intrusion alerts |
