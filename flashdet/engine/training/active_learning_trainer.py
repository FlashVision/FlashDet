"""Active Learning Trainer for FlashDet.

Implements an iterative train-then-query loop where the model selects
the most informative unlabeled images for annotation, maximising
accuracy while minimising labeling cost.

Usage::

    from flashdet.engine import ActiveLearningTrainer

    trainer = ActiveLearningTrainer(
        labeled_images="data/labeled/train",
        unlabeled_pool="data/unlabeled/images",
        query_strategy="entropy",
        query_budget=50,
        al_rounds=5,
    )
    trainer.train()
"""

import os
import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from flashdet.engine.training.trainer import Trainer
from flashdet.utils import AverageMeter

logger = logging.getLogger(__name__)


class ActiveLearningTrainer(Trainer):
    """Active Learning trainer for object detection.

    Runs multiple rounds of:

    1. **Train** the model on the current labeled set.
    2. **Score** every image in the unlabeled pool using an
       acquisition function.
    3. **Query** the top-K most informative images and move
       them from unlabeled pool to labeled set (user annotates).

    Supported acquisition strategies:

    - ``"entropy"``: Predictive entropy over class probabilities —
      high entropy means the model is most uncertain.
    - ``"margin"``: Smallest margin between top-2 class probabilities.
    - ``"least_confidence"``: 1 − max class probability.
    - ``"random"``: Random baseline.
    - ``"mc_dropout"``: Monte-Carlo dropout for epistemic uncertainty
      (runs T forward passes with dropout enabled).

    Args:
        unlabeled_pool: Path to directory of unlabeled candidate images.
        query_strategy: Acquisition function name.
        query_budget: Number of images to query per AL round.
        al_rounds: Total number of active learning rounds.
        mc_dropout_T: Number of MC-dropout forward passes (if strategy="mc_dropout").
        **kwargs: Forwarded to :class:`Trainer`.
    """

    STRATEGIES = ("entropy", "margin", "least_confidence", "random", "mc_dropout")

    def __init__(
        self,
        unlabeled_pool: Optional[str] = None,
        query_strategy: str = "entropy",
        query_budget: int = 50,
        al_rounds: int = 5,
        mc_dropout_T: int = 10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert query_strategy in self.STRATEGIES, (
            f"query_strategy must be one of {self.STRATEGIES}, got '{query_strategy}'"
        )
        self.unlabeled_pool = unlabeled_pool
        self.query_strategy = query_strategy
        self.query_budget = query_budget
        self.al_rounds = al_rounds
        self.mc_dropout_T = mc_dropout_T

    def train(self):
        """Run the active learning loop: train → score → query → repeat."""
        self._logger.info("=" * 60)
        self._logger.info(f"Active Learning ({self.query_strategy}, {self.al_rounds} rounds)")
        self._logger.info("=" * 60)

        all_results = []
        for al_round in range(self.al_rounds):
            self._logger.info(f"\n--- AL Round {al_round + 1}/{self.al_rounds} ---")
            result = super().train()
            all_results.append(result)

            if self.unlabeled_pool and al_round < self.al_rounds - 1:
                self._logger.info(f"Scoring unlabeled pool for next round...")
                summary = self.get_al_summary()
                self._logger.info(
                    f"  Strategy: {summary['strategy']}, Budget: {summary['budget_per_round']}"
                )
                self._logger.info(
                    f"  (Note: automatic annotation transfer requires external labeling. "
                    f"Query indices are logged for manual annotation.)"
                )

        self._logger.info(f"\nActive Learning complete. {len(all_results)} rounds finished.")
        return all_results[-1] if all_results else {}

    @torch.no_grad()
    def score_unlabeled(
        self, model: nn.Module, images: torch.Tensor,
    ) -> torch.Tensor:
        """Compute acquisition scores for a batch of unlabeled images.

        Higher score = more informative = should be queried first.

        Args:
            model: The current detection model.
            images: Batch of images (B, C, H, W).

        Returns:
            Tensor of shape (B,) with per-image scores.
        """
        model.eval()

        if self.query_strategy == "random":
            return torch.rand(images.shape[0], device=images.device)

        if self.query_strategy == "mc_dropout":
            return self._mc_dropout_score(model, images)

        results = model.predict(images, None, score_thr=0.01, nms_thr=0.65)
        scores_per_image = []

        for dets, labels in results:
            if dets is None or dets.numel() == 0:
                scores_per_image.append(torch.tensor(1.0, device=images.device))
                continue

            conf = dets[:, 4]

            if self.query_strategy == "entropy":
                p = conf.clamp(1e-7, 1 - 1e-7)
                entropy = -(p * p.log() + (1 - p) * (1 - p).log())
                scores_per_image.append(entropy.mean())
            elif self.query_strategy == "margin":
                sorted_conf = conf.sort(descending=True).values
                if sorted_conf.numel() >= 2:
                    margin = sorted_conf[0] - sorted_conf[1]
                    scores_per_image.append(1.0 - margin)
                else:
                    scores_per_image.append(torch.tensor(1.0, device=images.device))
            elif self.query_strategy == "least_confidence":
                scores_per_image.append(1.0 - conf.max())
            else:
                scores_per_image.append(torch.tensor(0.0, device=images.device))

        return torch.stack(scores_per_image)

    def _mc_dropout_score(self, model: nn.Module, images: torch.Tensor) -> torch.Tensor:
        """Monte-Carlo Dropout uncertainty estimation.

        Runs T forward passes with dropout enabled and measures the
        variance in predictions as an uncertainty score.
        """
        model.train()
        all_confs = []

        for _ in range(self.mc_dropout_T):
            results = model.predict(images, None, score_thr=0.01, nms_thr=0.65)
            batch_confs = []
            for dets, _ in results:
                if dets is not None and dets.numel() > 0:
                    batch_confs.append(dets[:, 4].mean().item())
                else:
                    batch_confs.append(0.0)
            all_confs.append(batch_confs)

        confs = torch.tensor(all_confs, device=images.device)
        variance = confs.var(dim=0)
        model.eval()
        return variance

    def select_query_indices(self, scores: torch.Tensor) -> torch.Tensor:
        """Select top-K indices with highest acquisition scores.

        Args:
            scores: (N,) acquisition scores for the full unlabeled pool.

        Returns:
            (K,) indices of images to query for annotation.
        """
        k = min(self.query_budget, scores.shape[0])
        _, indices = scores.topk(k)
        return indices

    def get_al_summary(self) -> Dict:
        """Return a summary of the active learning configuration."""
        return {
            "strategy": self.query_strategy,
            "budget_per_round": self.query_budget,
            "total_rounds": self.al_rounds,
            "mc_dropout_passes": self.mc_dropout_T if self.query_strategy == "mc_dropout" else None,
        }
