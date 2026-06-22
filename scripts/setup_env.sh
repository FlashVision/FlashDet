#!/usr/bin/env bash
# ============================================================
#  FlashDet — One-Command Environment Setup
# ============================================================
#
#  Creates a Python virtual environment, installs the correct
#  PyTorch build (CPU or CUDA), and all project dependencies.
#
#  Usage:
#    bash setup_env.sh              # auto-detect GPU
#    bash setup_env.sh --cpu        # force CPU-only
#    bash setup_env.sh --cuda 11.8  # force specific CUDA version
#    bash setup_env.sh --cuda 12.4  # force CUDA 12.4
#
#  After setup, activate with:
#    source venv/bin/activate
#
# ============================================================

set -euo pipefail

VENV_DIR="venv"
PYTHON="${PYTHON:-python3}"
FORCE_CPU=false
FORCE_CUDA=""

# ---- Parse arguments ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu)
            FORCE_CPU=true
            shift
            ;;
        --cuda)
            FORCE_CUDA="$2"
            shift 2
            ;;
        --venv)
            VENV_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: bash setup_env.sh [--cpu] [--cuda VERSION] [--venv DIR]"
            echo ""
            echo "Options:"
            echo "  --cpu           Force CPU-only PyTorch (no GPU)"
            echo "  --cuda VERSION  Force a specific CUDA version (11.8, 12.1, 12.4)"
            echo "  --venv DIR      Virtual environment directory (default: venv)"
            echo ""
            echo "If no flags are given, the script auto-detects GPU availability."
            exit 0
            ;;
        *)
            echo "Unknown option: $1 (use --help)"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "  FlashDet Environment Setup"
echo "=============================================="
echo ""

# ---- Check Python ----
if ! command -v "$PYTHON" &>/dev/null; then
    echo "ERROR: $PYTHON not found. Install Python 3.10+ first."
    exit 1
fi

PY_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$("$PYTHON" -c "import sys; print(sys.version_info.major)")
PY_MINOR=$("$PYTHON" -c "import sys; print(sys.version_info.minor)")

if [[ "$PY_MAJOR" -lt 3 ]] || [[ "$PY_MINOR" -lt 10 ]]; then
    echo "ERROR: Python 3.10+ required (found $PY_VERSION)"
    exit 1
fi
echo "[OK] Python $PY_VERSION"

# ---- Create virtual environment ----
if [[ -d "$VENV_DIR" ]]; then
    echo "[OK] Virtual environment already exists: $VENV_DIR/"
else
    echo "Creating virtual environment: $VENV_DIR/"
    "$PYTHON" -m venv "$VENV_DIR"
fi

# Activate
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
echo "[OK] Activated: $(which python)"

# Upgrade pip
pip install --upgrade pip --quiet

# ---- Detect GPU / select PyTorch index URL ----
TORCH_INDEX=""
TORCH_LABEL=""

if [[ "$FORCE_CPU" == true ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    TORCH_LABEL="CPU-only (forced)"
elif [[ -n "$FORCE_CUDA" ]]; then
    case "$FORCE_CUDA" in
        11.8|11.8.*)
            TORCH_INDEX="https://download.pytorch.org/whl/cu118"
            TORCH_LABEL="CUDA 11.8 (forced)"
            ;;
        12.1|12.1.*)
            TORCH_INDEX="https://download.pytorch.org/whl/cu121"
            TORCH_LABEL="CUDA 12.1 (forced)"
            ;;
        12.4|12.4.*|12.6|12.6.*|12.*)
            TORCH_INDEX="https://download.pytorch.org/whl/cu124"
            TORCH_LABEL="CUDA 12.x (forced)"
            ;;
        *)
            echo "ERROR: Unsupported CUDA version: $FORCE_CUDA"
            echo "  Supported: 11.8, 12.1, 12.4"
            exit 1
            ;;
    esac
else
    # Auto-detect
    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
        CUDA_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
        # Try to get CUDA version from nvidia-smi
        CUDA_RUNTIME=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || echo "")
        if [[ -n "$CUDA_RUNTIME" ]]; then
            CUDA_MAJOR=$(echo "$CUDA_RUNTIME" | cut -d. -f1)
            if [[ "$CUDA_MAJOR" -ge 12 ]]; then
                TORCH_INDEX="https://download.pytorch.org/whl/cu124"
                TORCH_LABEL="CUDA 12.x (auto-detected: $CUDA_RUNTIME)"
            else
                TORCH_INDEX="https://download.pytorch.org/whl/cu118"
                TORCH_LABEL="CUDA 11.8 (auto-detected: $CUDA_RUNTIME)"
            fi
        else
            TORCH_INDEX="https://download.pytorch.org/whl/cu124"
            TORCH_LABEL="CUDA 12.x (GPU detected, assuming 12.x)"
        fi
    else
        TORCH_INDEX="https://download.pytorch.org/whl/cpu"
        TORCH_LABEL="CPU-only (no GPU detected)"
    fi
fi

echo "[OK] PyTorch variant: $TORCH_LABEL"
echo ""

# ---- Install PyTorch ----
echo "Installing PyTorch..."
pip install torch torchvision --index-url "$TORCH_INDEX" --quiet
echo "[OK] $(python -c "import torch; print(f'torch {torch.__version__}  CUDA: {torch.cuda.is_available()}')")"

# ---- Install project dependencies ----
echo ""
echo "Installing project dependencies..."
pip install -r requirements.txt --quiet
echo "[OK] All dependencies installed"

# ---- Verify ----
echo ""
echo "=============================================="
echo "  Verification"
echo "=============================================="

python -c "
import sys, torch, numpy, cv2, PIL, matplotlib, onnx, onnxruntime, PyQt5

print(f'  Python:       {sys.version.split()[0]}')
print(f'  PyTorch:      {torch.__version__}')
print(f'  CUDA:         {torch.version.cuda if torch.cuda.is_available() else \"N/A (CPU)\"}')
if torch.cuda.is_available():
    print(f'  GPU:          {torch.cuda.get_device_name(0)}')
print(f'  NumPy:        {numpy.__version__}')
print(f'  OpenCV:       {cv2.__version__}')
print(f'  Pillow:       {PIL.__version__}')
print(f'  Matplotlib:   {matplotlib.__version__}')
print(f'  ONNX:         {onnx.__version__}')
print(f'  ONNXRuntime:  {onnxruntime.__version__}')
print(f'  PyQt5:        {PyQt5.QtCore.PYQT_VERSION_STR}')
"

echo ""
echo "=============================================="
echo "  Setup Complete!"
echo "=============================================="
echo ""
echo "  Activate the environment:"
echo "    source $VENV_DIR/bin/activate"
echo ""
echo "  Run the UI:"
echo "    python ui/main.py"
echo ""
echo "  Train a model:"
echo "    python train.py --help"
echo ""
