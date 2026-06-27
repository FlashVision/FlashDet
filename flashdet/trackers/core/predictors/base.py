"""Base class for bounding-box predictors.

All predictors follow this interface so tracker algorithms can
swap prediction strategies without code changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BasePredictor(ABC):
    """Abstract base for bbox state predictors.

    Subclasses implement ``initiate``, ``predict``, ``update``, and
    optionally ``set_frame`` for image-based methods.
    """

    @abstractmethod
    def initiate(self, bbox: np.ndarray):
        """Initialise internal state from the first detection [x1,y1,x2,y2]."""

    @abstractmethod
    def predict(self) -> np.ndarray:
        """Predict the next bounding box. Returns [x1,y1,x2,y2]."""

    @abstractmethod
    def update(self, bbox: np.ndarray):
        """Correct the internal state with a matched detection [x1,y1,x2,y2]."""

    def set_frame(self, frame: np.ndarray):
        """Provide the current video frame (needed by image-based predictors)."""

    @property
    def needs_frame(self) -> bool:
        """True if this predictor requires ``set_frame`` each tick."""
        return False
