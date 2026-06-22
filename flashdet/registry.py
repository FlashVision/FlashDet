"""Registry pattern for pluggable components.

Allows registering and discovering backbones, necks, heads, losses,
datasets, and other components by name — so users can swap them via config
without modifying source code.

Usage:
    from flashdet.registry import BACKBONES, NECKS, HEADS

    @BACKBONES.register("MobileNetV3")
    class MobileNetV3(nn.Module):
        ...

    # Later, build from config
    backbone = BACKBONES.build("MobileNetV3", **kwargs)
"""

from typing import Any, Callable, Dict, Optional


class Registry:
    """A registry that maps names to classes/functions."""

    def __init__(self, name: str):
        self._name = name
        self._registry: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return self._name

    def register(self, name: Optional[str] = None) -> Callable:
        """Register a class or function.

        Can be used as a decorator:
            @BACKBONES.register("MyBackbone")
            class MyBackbone: ...

        Or without arguments (uses class name):
            @BACKBONES.register()
            class MyBackbone: ...
        """
        def decorator(obj):
            key = name or obj.__name__
            if key in self._registry:
                raise KeyError(f"{self._name}: '{key}' is already registered")
            self._registry[key] = obj
            return obj

        if callable(name):
            obj = name
            key = obj.__name__
            self._registry[key] = obj
            return obj

        return decorator

    def build(self, name: str, **kwargs) -> Any:
        """Build a registered component by name."""
        if name not in self._registry:
            available = ", ".join(sorted(self._registry.keys()))
            raise KeyError(
                f"{self._name}: '{name}' not found. Available: [{available}]"
            )
        return self._registry[name](**kwargs)

    def get(self, name: str) -> Any:
        """Get the registered class without instantiating."""
        if name not in self._registry:
            available = ", ".join(sorted(self._registry.keys()))
            raise KeyError(
                f"{self._name}: '{name}' not found. Available: [{available}]"
            )
        return self._registry[name]

    def list(self) -> list:
        """List all registered names."""
        return sorted(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __len__(self) -> int:
        return len(self._registry)

    def __repr__(self) -> str:
        return f"Registry(name={self._name}, items={self.list()})"


BACKBONES = Registry("backbones")
NECKS = Registry("necks")
HEADS = Registry("heads")
LOSSES = Registry("losses")
DETECTORS = Registry("detectors")
DATASETS = Registry("datasets")
TRANSFORMS = Registry("transforms")
TRACKERS = Registry("trackers")
