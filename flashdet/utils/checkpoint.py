"""
Checkpoint utilities for saving and loading models.
"""

import os
import warnings
from typing import Dict

import torch


def _state_dict_for_save(model: torch.nn.Module) -> Dict:
    """Strip torch.compile prefixes so checkpoints remain loadable."""
    state = model.state_dict()
    return {
        k.replace("_orig_mod.", "", 1): v
        for k, v in state.items()
    }


def _unsafe_load_allowed(allow_unsafe) -> bool:
    if allow_unsafe is None:
        return os.environ.get("FLASHDET_ALLOW_UNSAFE_LOAD", "0").lower() in (
            "1", "true", "yes", "on",
        )
    return bool(allow_unsafe)


def safe_torch_load(path, map_location="cpu", allow_unsafe=None):
    """Load a ``.pth`` checkpoint without enabling arbitrary pickle execution.

    PyTorch's default ``weights_only=False`` can execute arbitrary code embedded
    in a checkpoint.  This helper uses ``weights_only=True`` and only falls back
    to the legacy loader when the caller explicitly opts in (argument or the
    ``FLASHDET_ALLOW_UNSAFE_LOAD=1`` environment variable) *and* the file is
    trusted.

    Args:
        path: Checkpoint path.
        map_location: Torch device mapping.
        allow_unsafe: ``True`` to permit the legacy pickle loader. ``None``
            reads ``FLASHDET_ALLOW_UNSAFE_LOAD``; ``False`` forces safe mode.

    Raises:
        RuntimeError: The checkpoint needs the unsafe loader and unsafe loading
            was not explicitly enabled.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as safe_error:
        if not _unsafe_load_allowed(allow_unsafe):
            raise RuntimeError(
                f"Refusing to load untrusted checkpoint {path!r}: the file "
                "contains objects that require PyTorch's unsafe pickle loader. "
                "Use a state-dict checkpoint, or set "
                "FLASHDET_ALLOW_UNSAFE_LOAD=1 only for a file you trust."
            ) from safe_error
        warnings.warn(
            f"Loading {path!r} with weights_only=False because unsafe loading "
            "was explicitly enabled. Only do this for trusted checkpoints.",
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.load(path, map_location=map_location, weights_only=False)


def load_inference_model(model_path: str, device: str = "cpu"):
    """Rebuild an inference model from a safe state-dict checkpoint.

    Supports FlashDet checkpoints and older full-module pickles (the latter
    only when unsafe loading is explicitly enabled).  This keeps analytics
    tools from calling ``torch.load(weights_only=False)`` directly.
    """
    ckpt = safe_torch_load(model_path, map_location=device)
    if isinstance(ckpt, torch.nn.Module):
        return ckpt.to(device).eval()

    if not isinstance(ckpt, dict):
        raise ValueError(f"Unsupported checkpoint payload in {model_path!r}")

    cfg = ckpt.get("config", {}) or {}
    from flashdet.cfg import get_config
    from flashdet.models.detector import build_model

    num_classes = cfg.get("num_classes", 80)
    arch = cfg.get("architecture", "flashdet")
    size = cfg.get("model_size", "n")
    config = get_config(num_classes=num_classes)
    config.model.architecture = arch
    config.model.size = size
    config.model.backbone_type = cfg.get("backbone_type") or "lite"
    config.model.neck_type = cfg.get("neck_type")
    if arch in ("yolov8", "yolov9", "yolov10", "yolov11"):
        config.model.width_mult = cfg.get("width_mult", 1.0)
        config.model.depth_mult = cfg.get("depth_mult", 1.0)

    model = build_model(config, architecture=arch)
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict"))
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint {model_path!r} has no model_state_dict")
    model.load_state_dict(state_dict, strict=False)
    return model.to(device).eval()


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    save_path: str,
    metrics: Dict = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    config: Dict = None,
    ema=None,
) -> str:
    """
    Save training checkpoint.

    Args:
        model: Model to save.
        optimizer: Optimizer state.
        epoch: Current epoch number.
        loss: Current loss value.
        save_path: Destination path.
        metrics: Additional metrics to include (optional).
        scheduler: LR scheduler (optional).
        config: Model configuration dict (optional).
        ema: ModelEMA instance — when provided its state is saved under
            ``"ema_state_dict"`` so training can resume without a separate
            load-then-save round-trip (optional).

    Returns:
        Path to the saved checkpoint.
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": _state_dict_for_save(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "metrics": metrics or {},
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if ema is not None:
        checkpoint["ema_state_dict"] = ema.state_dict()

    if config is not None:
        checkpoint["config"] = config
    elif hasattr(model, "num_classes"):
        checkpoint["config"] = {
            "num_classes": getattr(model, "num_classes", 10),
            "input_size": getattr(model, "input_size", (320, 320)),
        }

    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved: {save_path}")

    return save_path


def save_weights_only(
    model: torch.nn.Module,
    save_path: str,
    config: Dict = None
) -> str:
    """
    Save model weights only (smaller file for deployment).
    
    Args:
        model: Model to save
        save_path: Path to save weights
        config: Model configuration (optional)
        
    Returns:
        Path to saved weights
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    
    checkpoint = {
        "model_state_dict": _state_dict_for_save(model),
    }
    
    if config is not None:
        checkpoint["config"] = config
    elif hasattr(model, "num_classes"):
        checkpoint["config"] = {
            "num_classes": getattr(model, "num_classes", 10),
            "input_size": getattr(model, "input_size", (320, 320)),
        }
    
    torch.save(checkpoint, save_path)
    
    size_mb = os.path.getsize(save_path) / 1e6
    print(f"Weights saved: {save_path} ({size_mb:.2f} MB)")
    
    return save_path


def save_inference_weights(
    model: torch.nn.Module,
    save_path: str,
    config: Dict = None,
    half: bool = False
) -> str:
    """
    Save inference-only weights (excludes aux_head, optionally FP16).
    
    This produces a much smaller file suitable for deployment:
    - Excludes auxiliary head (only used during training)
    - Optionally converts to FP16 for ~50% size reduction
    
    Args:
        model: Model to save (FlashDet)
        save_path: Path to save weights
        config: Model configuration (optional)
        half: If True, save as FP16 (half precision)
        
    Returns:
        Path to saved weights
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    
    state_dict = _state_dict_for_save(model)
    inference_state_dict = {
        k: v for k, v in state_dict.items()
        if not k.startswith(("aux_head.", "aux_heads.", "head.o2m_heads.", "o2m_heads."))
    }
    
    # Convert to FP16 if requested
    if half:
        inference_state_dict = {
            k: v.half() if v.dtype == torch.float32 else v 
            for k, v in inference_state_dict.items()
        }
    
    checkpoint = {
        "model_state_dict": inference_state_dict,
        "half": half,
    }
    
    if config is not None:
        checkpoint["config"] = config
    elif hasattr(model, "num_classes"):
        checkpoint["config"] = {
            "num_classes": getattr(model, "num_classes", 10),
            "input_size": getattr(model, "input_size", (320, 320)),
        }
    
    torch.save(checkpoint, save_path)
    
    size_mb = os.path.getsize(save_path) / 1e6
    precision = "FP16" if half else "FP32"
    print(f"Inference weights saved: {save_path} ({size_mb:.2f} MB, {precision})")
    
    return save_path


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    optimizer: torch.optim.Optimizer = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    device: str = "cuda"
) -> Dict:
    """
    Load training checkpoint.
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        device: Device to load to
        
    Returns:
        Checkpoint dictionary with epoch, loss, metrics
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = safe_torch_load(checkpoint_path, map_location=device)
    
    # Load model weights — always strict=False so checkpoints saved before the
    # aux_fpn was added (or with any architecture change) still resume cleanly.
    if "model_state_dict" in checkpoint:
        missing, unexpected = model.load_state_dict(
            checkpoint["model_state_dict"], strict=False
        )
    elif "state_dict" in checkpoint:
        state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
    else:
        missing, unexpected = model.load_state_dict(checkpoint, strict=False)

    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
    print(f"Model loaded from: {checkpoint_path}")
    
    # Load optimizer state — skip gracefully if the architecture changed and the
    # saved parameter groups no longer match (e.g. aux_fpn was added).
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("Optimizer state loaded")
        except (ValueError, KeyError) as e:
            print(f"  Optimizer state skipped (architecture mismatch: {e}). "
                  "Starting with fresh optimizer.")

    # Load scheduler state — skip gracefully on step-count mismatch too.
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print("Scheduler state loaded")
        except (ValueError, KeyError) as e:
            print(f"  Scheduler state skipped ({e}). Starting fresh.")
    
    return {
        "epoch": checkpoint.get("epoch", 0),
        "loss": checkpoint.get("loss", 0.0),
        "metrics": checkpoint.get("metrics", {})
    }


def save_model(model: torch.nn.Module, save_path: str) -> str:
    """
    Save model weights only.

    Args:
        model: Model to save.
        save_path: Path to save weights.

    Returns:
        Path to saved weights.
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved: {save_path}")
    return save_path


def load_model(model: torch.nn.Module, weight_path: str, device: str = "cuda") -> torch.nn.Module:
    """
    Load model weights.
    
    Args:
        model: Model to load weights into
        weight_path: Path to weights
        device: Device to load to
        
    Returns:
        Model with loaded weights
    """
    state_dict = safe_torch_load(weight_path, map_location=device)
    
    # Handle different checkpoint formats
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    elif "state_dict" in state_dict:
        state_dict = {k.replace("model.", ""): v for k, v in state_dict["state_dict"].items()}
    
    model.load_state_dict(state_dict, strict=False)
    print(f"Weights loaded from: {weight_path}")
    
    return model
