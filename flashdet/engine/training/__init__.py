"""Training methods for object detection.

Available trainers:

- :class:`Trainer` — Standard supervised detection training.
- :class:`KDTrainer` — Knowledge Distillation (teacher → student).
- :class:`SSLTrainer` — Self-Supervised Learning (BYOL / MoCo / SimCLR).
- :class:`FewShotTrainer` — Few-Shot learning from limited examples.
- :class:`SemiSupervisedTrainer` — Semi-Supervised with pseudo-labels.
- :class:`ActiveLearningTrainer` — Active Learning with uncertainty querying.

To add a new training method:
  1. Create ``flashdet/engine/training/my_trainer.py``.
  2. Subclass :class:`Trainer` or write a standalone class.
  3. Import and export it here.
"""

from flashdet.engine.training.trainer import Trainer
from flashdet.engine.training.kd_trainer import KDTrainer
from flashdet.engine.training.ssl_trainer import SSLTrainer
from flashdet.engine.training.few_shot_trainer import FewShotTrainer
from flashdet.engine.training.semi_supervised_trainer import SemiSupervisedTrainer
from flashdet.engine.training.active_learning_trainer import ActiveLearningTrainer

__all__ = [
    "Trainer",
    "KDTrainer",
    "SSLTrainer",
    "FewShotTrainer",
    "SemiSupervisedTrainer",
    "ActiveLearningTrainer",
]
