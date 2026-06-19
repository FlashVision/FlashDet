# Installation

## From PyPI

```bash
pip install flashdet
```

## With extras

```bash
pip install "flashdet[all]"       # Everything
pip install "flashdet[export]"    # ONNX export
pip install "flashdet[tracker]"   # ByteTracker, SORT, BoTSORT
pip install "flashdet[solutions]" # Counting, speed, heatmaps
pip install "flashdet[analytics]" # Benchmarking, plots
```

## From source

```bash
git clone https://github.com/FlashVision/FlashDet.git
cd FlashDet
pip install -e ".[all]"
```

## Verify

```bash
flashdet check
flashdet version
```

## Requirements

- Python >= 3.8
- PyTorch >= 2.0
- OpenCV >= 4.5
