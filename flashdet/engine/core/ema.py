"""Exponential Moving Average of model weights.

Single source of truth — imported by ``train.py``, ``Trainer``, etc.
"""

import copy
import math

import torch
import torch.nn as nn


class ModelEMA:
    """Exponential Moving Average of model weights with adaptive decay warmup.

    Uses an adaptive decay schedule so the EMA converges quickly even with
    small datasets (few batches/epoch).  The effective decay ramps from 0
    (EMA = model copy) toward *target_decay* over the first few thousand
    updates, using::

        effective_decay = target_decay * (1 - exp(-n / warmup))

    With the default warmup=2000, the decay is ~0.63 * target after 2000
    updates and ~0.99 * target after 10 000 updates.

    Args:
        model: The model whose weights will be averaged.
        decay: Target EMA decay factor.
        warmup: Number of updates over which to ramp the decay.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9998, warmup: int = 2000):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.target_decay = decay
        self.warmup = warmup
        self.num_updates = 0
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @property
    def decay(self):
        if self.warmup <= 0:
            return self.target_decay
        return self.target_decay * (1.0 - math.exp(-self.num_updates / self.warmup))

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.num_updates += 1
        d = self.decay
        for ema_p, model_p in zip(self.ema.parameters(), model.parameters()):
            ema_p.data.mul_(d).add_(model_p.data, alpha=1.0 - d)
        for ema_b, model_b in zip(self.ema.buffers(), model.buffers()):
            ema_b.copy_(model_b)

    def state_dict(self):
        return {
            "ema_state": self.ema.state_dict(),
            "target_decay": self.target_decay,
            "warmup": self.warmup,
            "num_updates": self.num_updates,
        }

    def load_state_dict(self, state: dict):
        self.ema.load_state_dict(state["ema_state"], strict=False)
        self.target_decay = state.get("target_decay",
                                      state.get("decay", self.target_decay))
        self.warmup = state.get("warmup", self.warmup)
        self.num_updates = state.get("num_updates", 0)
