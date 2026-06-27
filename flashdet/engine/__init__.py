"""FlashDet Engine — training, evaluation, inference, and core utilities.

Organized into sub-packages:

- ``engine.training/``   — All training methods (supervised, KD, SSL, few-shot, etc.)
- ``engine.evaluation/`` — Model validation
- ``engine.inference/``  — Post-processing and prediction decoding
- ``engine.core/``       — Shared utilities (EMA, callbacks)
"""

# ── Training methods ──────────────────────────────────────────────────
from flashdet.engine.training import (
    Trainer,
    SSLTrainer,
    FewShotTrainer,
    SemiSupervisedTrainer,
    ActiveLearningTrainer,
)

# ── Evaluation ────────────────────────────────────────────────────────
from flashdet.engine.evaluation import Validator

# ── Inference ─────────────────────────────────────────────────────────
from flashdet.engine.inference import decode_yolo_predictions, decode_detr_predictions, Predictor

# ── Core utilities ────────────────────────────────────────────────────
from flashdet.engine.core import (
    ModelEMA,
    MuSGD,
    build_musgd,
    Callback,
    CallbackList,
    EarlyStopping,
    LRSchedulerCallback,
    CSVLogger,
    TensorBoardCallback,
)

__all__ = [
    # Training
    "Trainer",
    "SSLTrainer",
    "FewShotTrainer",
    "SemiSupervisedTrainer",
    "ActiveLearningTrainer",
    # Evaluation
    "Validator",
    # Inference
    "decode_yolo_predictions",
    "decode_detr_predictions",
    "Predictor",
    # Core
    "ModelEMA",
    "MuSGD",
    "build_musgd",
    "Callback",
    "CallbackList",
    "EarlyStopping",
    "LRSchedulerCallback",
    "CSVLogger",
    "TensorBoardCallback",
]
