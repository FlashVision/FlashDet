"""FlashDet Engine — training, inference, evaluation, and export.

Organized into sub-packages:

- ``engine.training/``   — All training methods (supervised, KD, SSL, few-shot, etc.)
- ``engine.inference/``  — Prediction and post-processing
- ``engine.evaluation/`` — Model validation
- ``engine.export/``     — Model export (ONNX, TorchScript)
- ``engine.core/``       — Shared utilities (EMA, callbacks)

Everything is re-exported here for convenience::

    from flashdet.engine import Trainer, KDTrainer, SSLTrainer
    from flashdet.engine import Predictor, Validator, Exporter
    from flashdet.engine import ModelEMA, Callback
"""

# ── Training methods ──────────────────────────────────────────────────
from flashdet.engine.training import (
    Trainer,
    KDTrainer,
    SSLTrainer,
    FewShotTrainer,
    SemiSupervisedTrainer,
    ActiveLearningTrainer,
)

# ── Inference ─────────────────────────────────────────────────────────
from flashdet.engine.inference import (
    Predictor,
    decode_yolo_predictions,
    decode_detr_predictions,
)

# ── Evaluation ────────────────────────────────────────────────────────
from flashdet.engine.evaluation import Validator

# ── Export ────────────────────────────────────────────────────────────
from flashdet.engine.export import Exporter

# ── Core utilities ────────────────────────────────────────────────────
from flashdet.engine.core import (
    ModelEMA,
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
    "KDTrainer",
    "SSLTrainer",
    "FewShotTrainer",
    "SemiSupervisedTrainer",
    "ActiveLearningTrainer",
    # Inference
    "Predictor",
    "decode_yolo_predictions",
    "decode_detr_predictions",
    # Evaluation
    "Validator",
    # Export
    "Exporter",
    # Core
    "ModelEMA",
    "Callback",
    "CallbackList",
    "EarlyStopping",
    "LRSchedulerCallback",
    "CSVLogger",
    "TensorBoardCallback",
]
