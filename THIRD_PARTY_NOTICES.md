# Third-Party Notices

This file documents third-party works that influenced FlashDet's design.

## Important: Licensing Compliance

FlashDet is released under the **MIT License**. All code in this repository is
independently written. No code has been copied from GPL or AGPL licensed sources.

The YOLO model architectures in this project are **clean-room implementations**
based on publicly available academic papers. Neural network architectures
described in papers are not copyrightable — only specific code implementations
are protected by copyright.

---

## Academic References

### YOLOX
- **Paper:** Ge et al., "YOLOX: Exceeding YOLO Series in 2021", arXiv:2107.08430, 2021.
- **Concepts Used:** Anchor-free detection, decoupled head (separate cls/reg/obj branches), Focus stem, CSP blocks with SPP, SimOTA assignment.
- **Note:** Original code at Megvii-BaseDetection/YOLOX is Apache-2.0 licensed. Our implementation is independently written based on the paper.

### YOLOv8
- **Paper/Source:** Jocher et al., "YOLOv8", Ultralytics, 2023.
- **Concepts Used:** C2f blocks (CSP Bottleneck with 2 convolutions), SPPF, PANet neck, DFL head, anchor-free.
- **Note:** Ultralytics' implementation is AGPL-3.0 licensed — we did NOT use their code. Our implementation is a clean-room version based on publicly documented architecture.

### YOLOv9
- **Paper:** Wang et al., "YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information", arXiv:2402.13616, 2024.
- **Concepts Used:** GELAN (Generalized Efficient Layer Aggregation Network), PGI (Programmable Gradient Information).
- **Note:** Our implementation is written from scratch based on the paper. The original authors' code (WongKinYiu/yolov9) is GPL-3.0 licensed — we did NOT use their code.

### YOLOv10
- **Paper:** Wang et al., "YOLOv10: Real-Time End-to-End Object Detection", arXiv:2405.14458, 2024.
- **Concepts Used:** Spatial-Channel Decoupled Downsampling (SCDown), Partial Self-Attention (PSA), NMS-free dual-assignment.
- **Note:** The research code at THU-MIG/yolov10 is Apache-2.0 licensed. Our implementation is independently written based on the paper description.

### YOLO11
- **Paper/Source:** Jocher et al., YOLO11, Ultralytics, 2024.
- **Concepts Used:** C3k2 blocks, C2PSA attention, SPPF pooling.
- **Note:** Ultralytics' implementation is AGPL-3.0 licensed — we did NOT use their code. Our implementation is a clean-room version based on publicly documented architecture.

### RepNeXt / Structural Reparameterization
- **Paper:** "RepNeXt: A Fast Multi-Scale CNN using Structural Reparameterization", arXiv:2406.16004, 2024.
- **Concepts Used:** Multi-scale reparameterizable convolutions, PicoBlock, StrideDown.
- **Note:** Our implementation is independently written based on the paper.

### torchtune (Meta)
- **License:** BSD-3-Clause
- **Source:** https://github.com/pytorch/torchtune
- **Concepts Used:** Optimizer utilities, LoRA/QLoRA helpers, KD loss recipes.
- **Note:** Training infrastructure patterns inspired by torchtune's recipes.

### SORT (Multi-Object Tracking)
- **Paper:** Bewley et al., "Simple Online and Realtime Tracking", ICIP 2016. arXiv:1602.00763
- **Concepts Used:** Kalman filter prediction + Hungarian IoU assignment.
- **Note:** Our SortTracker is an independent implementation based on the paper.

### Deep SORT
- **Paper:** Wojke et al., "Simple Online and Realtime Tracking with a Deep Association Metric", ICIP 2017. arXiv:1703.07402
- **Concepts Used:** Mahalanobis gating, cascade matching, ReID appearance features.
- **Note:** Our DeepSortTracker is independently written based on the paper.

### ByteTrack
- **Paper:** Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box", ECCV 2022. arXiv:2110.06864
- **Concepts Used:** Two-stage association using low-confidence detections.
- **Note:** Our ByteTracker is independently written based on the paper.

### OC-SORT
- **Paper:** Cao et al., "Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking", CVPR 2023. arXiv:2203.14360
- **Concepts Used:** Observation-centric re-update (ORU), velocity consistency, virtual trajectories.
- **Note:** Our OCSortTracker is independently written based on the paper.

### BoT-SORT
- **Paper:** Aharon et al., "BoT-SORT: Robust Associations Multi-Pedestrian Tracking", arXiv:2206.14651, 2022.
- **Concepts Used:** ReID feature matching, camera motion compensation, hybrid cost matrix.
- **Note:** Our BoTSortTracker is independently written based on the paper.

### StrongSORT
- **Paper:** Du et al., "StrongSORT: Make DeepSORT Great Again", IEEE TMM 2023. arXiv:2202.13514
- **Concepts Used:** EMA feature update, ECC camera alignment, NSA Kalman filter.
- **Note:** Our StrongSortTracker is independently written based on the paper.

### Kernelized Correlation Filters (KCF)
- **Paper:** Henriques et al., "High-Speed Tracking with Kernelized Correlation Filters", IEEE TPAMI 2015.
- **Concepts Used:** HOG features, Gaussian kernel ridge regression in Fourier domain for bbox prediction.
- **Note:** Our KCFPredictor is independently written based on the paper.

### Median Flow
- **Paper:** Kalal et al., "Forward-Backward Error: Automatic Detection of Tracking Failures", ICPR 2010.
- **Concepts Used:** Sparse Lucas-Kanade optical flow with forward-backward error filtering.
- **Note:** Our MedianFlowPredictor is independently written based on the paper.

### Extended Kalman Filter (EKF)
- **Concepts Used:** Non-linear state estimation with constant-acceleration model (cx, cy, w, h, vx, vy, vw, vh, ax, ay).
- **Note:** Standard EKF formulation, independently implemented.

### IoU Variants (GIoU / DIoU / CIoU)
- **GIoU:** Rezatofighi et al., "Generalized Intersection over Union", CVPR 2019.
- **DIoU/CIoU:** Zheng et al., "Distance-IoU Loss", AAAI 2020.

### General YOLO Family
- **C2f Block:** Based on "YOLOv8" architecture description (CSP Bottleneck with 2 convolutions).
- **DFL (Distribution Focal Loss):** Li et al., "Generalized Focal Loss V2", 2020.
- **Hungarian Matching:** Kuhn, "The Hungarian Method for the assignment problem", 1955.
- **STAL (Small Target Aware Label Assignment):** Custom implementation for FlashDet.

---

## Runtime Dependencies

All required dependencies use permissive licenses:

| Package | License | Usage |
|---------|---------|-------|
| PyTorch | BSD-3-Clause | Deep learning framework |
| TorchVision | BSD-3-Clause | Vision utilities, NMS |
| NumPy | BSD-3-Clause | Numerical computation |
| OpenCV | Apache-2.0 | Image I/O and processing |
| Pillow | MIT-like (HPND) | Image loading |
| pycocotools | BSD-2-Clause | COCO evaluation |
| PyYAML | MIT | Config parsing |
| tqdm | MIT/MPL-2.0 | Progress bars |

### Optional Dependencies

| Package | License | Usage |
|---------|---------|-------|
| scipy | BSD-3-Clause | Tracker matching (Hungarian) |
| lap | BSD-2-Clause | Linear assignment (trackers) |
| onnx | Apache-2.0 | ONNX model export |
| onnxruntime | MIT | ONNX inference |
| onnxsim | Apache-2.0 | ONNX simplification |
| matplotlib | PSF/BSD | Training plots |
| pandas | BSD-3-Clause | Analytics/CSV |

---

## What This Project Does NOT Include

- No code from Ultralytics (AGPL-3.0)
- No code from WongKinYiu/yolov9 (GPL-3.0)
- No pretrained weights from AGPL/GPL sources
- No bundled third-party model checkpoints

Users who wish to compare with official YOLO implementations should install
those packages separately in their own environment, respecting their respective
licenses.
