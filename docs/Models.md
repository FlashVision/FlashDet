# Models

## FlashDet (Default)

Ultra-lightweight detector with ShuffleNetV2 backbone — the default model for production deployment.

| Model | Params | FP16 Size | mAP (COCO) | GPU Latency |
|-------|--------|-----------|------------|-------------|
| FlashDet-m-0.5x | 0.49M | ~1.2 MB | — | 3.8 ms |
| FlashDet-m | 1.17M | ~2.6 MB | 27.0 | 5.3 ms |
| FlashDet-m-1.5x | 2.44M | ~5.2 MB | 29.9 | 7.2 ms |

### Training

```python
from flashdet import Trainer

trainer = Trainer(
    model_size="m",          # "m-0.5x", "m", "m-1.5x"
    train_images="data/train",
    val_images="data/val",
    epochs=100,
    pretrained_coco=True,
)
trainer.train()
```

### Inference

```python
from flashdet import Predictor

predictor = Predictor(model_path="workspace/best.pth", device="cuda")
results = predictor.predict("photo.jpg")
```

---

## Using Different Architectures

FlashDet provides multiple detection architectures. Each can be used independently for training and inference.

### Architecture Overview

| Architecture | Type | Key Feature | Best For |
|-------------|------|-------------|----------|
| **FlashDet** | CNN | Ultra-lightweight, ShuffleNetV2 | Edge/mobile deployment |
| **DETR** | Transformer | End-to-end, no NMS | Research, high accuracy |
| **RT-DETR** | Transformer | Real-time DETR | Speed + accuracy balance |
| **YOLOv9** | CNN | PGI (Programmable Gradient Info) | General detection |
| **YOLOv10** | CNN | NMS-free, PSA attention | Real-time no postprocess |
| **YOLOv11** | CNN | C2PSA attention blocks | Latest YOLO features |
| **GroundingDINO** | Multimodal | Text-guided open-vocabulary | Zero-shot detection |

---

## DETR (Detection Transformer)

End-to-end detection with transformer and Hungarian matching — no NMS required.

### Training

```python
from flashdet.models.architectures import DETR
import torch
import numpy as np

model = DETR(
    num_classes=10,
    num_queries=100,          # number of object queries
    d_model=256,              # transformer hidden dim
    nhead=8,                  # attention heads
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
    backbone="resnet50",      # "resnet18", "resnet34", "resnet50", "resnet101"
    pretrained_backbone=True,
)

# Training step
model.train()
images = torch.randn(4, 3, 512, 512)
gt_meta = {
    "img": images,
    "gt_bboxes": [np.array([[x1, y1, x2, y2]], dtype=np.float32) for _ in range(4)],
    "gt_labels": [np.array([class_id], dtype=np.int64) for _ in range(4)],
}
output = model(images, gt_meta=gt_meta)
loss = output["loss"]        # scalar loss (CE + L1 + GIoU)
loss.backward()
```

### Inference

```python
model.eval()
results = model.predict(images, score_thr=0.5)
# results: List[Dict] with keys "boxes", "scores", "labels"
for det in results:
    print(det["boxes"])    # [N, 4] in xyxy format
    print(det["scores"])   # [N] confidence scores
    print(det["labels"])   # [N] class indices
```

---

## RT-DETR (Real-Time DETR)

Efficient transformer detector with hybrid encoder — faster than DETR while maintaining accuracy.

### Training

```python
from flashdet.models.architectures import RTDETR

model = RTDETR(
    num_classes=10,
    backbone="resnet50",       # "resnet18", "resnet34", "resnet50"
    hidden_dim=256,
    nhead=8,
    num_encoder_layers=1,
    num_decoder_layers=6,
    dim_feedforward=1024,
    num_queries=300,
    num_csp_blocks=3,
    pretrained_backbone=True,
)

model.train()
images = torch.randn(2, 3, 640, 640)
gt_meta = {
    "img": images,
    "gt_bboxes": [np.array([[50, 50, 200, 200]], dtype=np.float32),
                  np.array([[100, 100, 300, 300]], dtype=np.float32)],
    "gt_labels": [np.array([0], dtype=np.int64),
                  np.array([5], dtype=np.int64)],
}
output = model(images, gt_meta=gt_meta)
output["loss"].backward()
```

### Inference

```python
model.eval()
results = model.predict(images, score_thr=0.5)
```

---

## YOLOv9

Features Programmable Gradient Information (PGI) for better gradient flow during training.

### Training

```python
from flashdet.models.architectures import YOLOv9

model = YOLOv9(
    num_classes=10,
    width_mult=0.5,     # channel multiplier
    depth_mult=0.34,    # depth multiplier
    use_pgi=True,       # enable PGI auxiliary branch
)

model.train()
images = torch.randn(2, 3, 640, 640)
output = model(images)
# output["preds"]: list of prediction tensors per scale
# output["aux_preds"]: PGI auxiliary predictions (training only)
```

### Inference

```python
model.eval()
output = model(images)
# output["preds"]: list of prediction tensors
# No "aux_preds" at inference — PGI branch is training-only
```

---

## YOLOv10

NMS-free detection with dual label assignment (one-to-many for training, one-to-one for inference).

### Training

```python
from flashdet.models.architectures import YOLOv10

model = YOLOv10(
    num_classes=10,
    width_mult=0.5,
    depth_mult=0.34,
    use_psa=True,       # Partial Self-Attention module
)

model.train()
images = torch.randn(2, 3, 640, 640)
output = model(images)
# output["preds"]: one-to-one head predictions (for NMS-free inference)
# output["o2m_preds"]: one-to-many head predictions (training supervision only)
```

### Inference

```python
model.eval()
output = model(images)
# output["preds"]: one-to-one predictions — no NMS needed!
# "o2m_preds" not present at inference
```

---

## YOLOv11

Latest YOLO with C2PSA (Cross-Stage Partial Self-Attention) blocks.

### Training

```python
from flashdet.models.architectures import YOLOv11

model = YOLOv11(
    num_classes=10,
    width_mult=0.5,
    depth_mult=0.34,
    use_c2psa=True,     # C2PSA attention blocks
)

model.train()
images = torch.randn(2, 3, 640, 640)
output = model(images)
# output["preds"]: list of prediction tensors per scale
```

### Inference

```python
model.eval()
output = model(images)
```

---

## GroundingDINO

Open-vocabulary text-guided detection — detect any object described by text.

### Training

```python
from flashdet.models.architectures import GroundingDINO

model = GroundingDINO(
    num_queries=900,
    d_model=256,
    nhead=8,
    num_encoder_layers=3,
    num_decoder_layers=6,
    backbone="resnet50",
    pretrained_backbone=True,
    vocab_size=30522,       # BERT vocabulary
    text_embed_dim=256,
    max_text_len=77,
    text_encoder_depth=3,
)

model.train()
images = torch.randn(2, 3, 512, 512)
text_ids = torch.randint(0, 30522, (2, 10))    # tokenized text prompts
text_mask = torch.ones(2, 10)
gt_meta = {
    "img": images,
    "gt_bboxes": [np.array([[50, 50, 200, 200]], dtype=np.float32),
                  np.array([[100, 100, 300, 300]], dtype=np.float32)],
    "gt_labels": [np.array([0], dtype=np.int64),
                  np.array([5], dtype=np.int64)],
}
output = model(images, text_ids=text_ids, text_mask=text_mask, gt_meta=gt_meta)
output["loss"].backward()
```

### Inference

```python
model.eval()
results = model.predict(images, text_ids=text_ids, text_mask=text_mask, score_thr=0.3)
# results: List[Dict] with "boxes", "scores", "labels"
```

---

## Choosing a Model

| Use Case | Recommended Model | Why |
|----------|-------------------|-----|
| Mobile/Edge deployment | FlashDet-m-0.5x | Smallest, fastest |
| General real-time | FlashDet-m or YOLOv10 | Good speed/accuracy |
| High accuracy | DETR or RT-DETR | Transformer-based |
| No post-processing | YOLOv10 | NMS-free |
| Open-vocabulary | GroundingDINO | Text-guided |
| Custom fine-tuning | FlashDet + LoRA | Efficient adaptation |

---

## Common Training Pattern

All architectures follow the same forward API pattern:

```python
# Training
model.train()
output = model(images, gt_meta=gt_meta)
loss = output["loss"]
loss.backward()

# Inference
model.eval()
output = model(images)
preds = output["preds"]

# Predict (high-level, returns boxes/scores/labels)
results = model.predict(images, score_thr=0.5)
```

---

## Registry System

Register and build models dynamically:

```python
from flashdet.registry import BACKBONES

# All architectures are auto-registered
model = BACKBONES.build("DETR", num_classes=10, num_queries=100)
model = BACKBONES.build("YOLOv10", num_classes=10, use_psa=True)
model = BACKBONES.build("RTDETR", num_classes=10)

# List available architectures
print(BACKBONES.list())
```

---

## OBB (Oriented Bounding Box) Detection

For detecting rotated objects (aerial imagery, document text, etc.):

```python
from flashdet.models.head import OBBHead

obb_head = OBBHead(
    num_classes=15,
    in_channels=96,
    feat_channels=96,
    num_angle_bins=180,
    angle_mode="bins",  # or "direct"
)
```
