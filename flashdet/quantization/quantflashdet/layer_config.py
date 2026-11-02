"""Layer-group policies for FlashDet quantization.

Detection heads are sensitive to INT8, so the default policy keeps the head in
float and quantizes the backbone/neck. The policy is fully overridable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Tuple

VALID_METHODS = ("none", "static", "dynamic", "weight_only", "qat")


@dataclass
class QuantLayerPolicy:
    backbone: str = "static"
    neck: str = "static"
    head: str = "none"

    def __post_init__(self):
        for group, method in asdict(self).items():
            if method not in VALID_METHODS:
                raise ValueError(
                    f"{group} policy {method!r} is invalid; "
                    f"choose from {VALID_METHODS}"
                )

    def as_dict(self) -> Dict[str, str]:
        return asdict(self)

    def groups_for(self, method: str) -> Tuple[str, ...]:
        return tuple(
            name for name, value in self.as_dict().items() if value == method
        )

    def keep_float(self) -> Tuple[str, ...]:
        return tuple(
            name for name, value in self.as_dict().items() if value == "none"
        )


DEFAULT_POLICY = QuantLayerPolicy()
