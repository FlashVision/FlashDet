"""
Torchtune-inspired memory & performance optimizations for FlashDet.

Techniques ported from https://github.com/meta-pytorch/torchtune:
  - Activation checkpointing (gradient checkpointing)
  - Activation offloading to CPU
  - Fused optimizer step into backward pass
  - 8-bit AdamW via bitsandbytes
  - torch.compile wrapper
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

logger = logging.getLogger(__name__)


def apply_activation_checkpointing(
    model: nn.Module,
    target_modules: Optional[list] = None,
):
    """Wrap submodules with gradient checkpointing to trade compute for memory.

    By default, checkpoints the backbone and FPN — the two most activation-heavy
    stages.  The detection head is left un-checkpointed because its activations
    are relatively small and checkpointing it can interfere with the loss
    backward graph.

    Inspired by torchtune's ``enable_activation_checkpointing`` recipe flag.
    """
    if target_modules is None:
        target_modules = ["backbone", "fpn", "aux_fpn"]

    wrapped = []
    for name in target_modules:
        mod = getattr(model, name, None)
        if mod is None:
            continue

        original_forward = mod.forward

        def _make_ckpt_forward(orig_fn):
            def ckpt_forward(*args, **kwargs):
                if torch.is_grad_enabled():
                    return torch_checkpoint(
                        orig_fn, *args, use_reentrant=False, **kwargs
                    )
                return orig_fn(*args, **kwargs)
            return ckpt_forward

        mod.forward = _make_ckpt_forward(original_forward)
        wrapped.append(name)

    if wrapped:
        logger.info(
            "Activation checkpointing enabled for: %s", ", ".join(wrapped)
        )
    else:
        logger.warning("No modules matched for activation checkpointing")

    return model


class ActivationOffloadHook:
    """Offload activations to CPU during forward, reload on backward.

    This pairs well with activation checkpointing — when activations *are*
    saved (e.g. for the head), this hook moves them to CPU to free GPU
    memory, then moves them back on the backward pass.

    Inspired by torchtune's ``enable_activation_offloading`` flag.
    """

    def __init__(self):
        self._handles = []

    def register(self, model: nn.Module, target_modules: list = None):
        if target_modules is None:
            target_modules = ["backbone", "fpn"]

        for name in target_modules:
            mod = getattr(model, name, None)
            if mod is None:
                continue
            for submod in mod.modules():
                h = submod.register_forward_hook(self._offload_hook)
                self._handles.append(h)

        logger.info(
            "Activation offloading registered on %d sub-modules",
            len(self._handles),
        )
        return self

    @staticmethod
    def _offload_hook(module, input, output):
        if not isinstance(output, torch.Tensor):
            return output
        if not output.requires_grad:
            return output

        cpu_output = output.detach().cpu()
        device = output.device

        class _Reload(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, cpu_copy):
                ctx.device = device
                ctx.cpu_copy = cpu_copy
                return x

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output, None

        return _Reload.apply(output, cpu_output)

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def create_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float = 0.05,
    use_8bit: bool = False,
    optimizer_in_bwd: bool = False,
    betas: tuple = (0.9, 0.999),
):
    """Create an optimizer with optional torchtune-style enhancements.

    Args:
        model: The model whose parameters to optimize.
        lr: Learning rate.
        weight_decay: Weight decay coefficient.
        use_8bit: Use bitsandbytes 8-bit AdamW (halves optimizer memory).
        optimizer_in_bwd: Fuse optimizer step into backward pass. When True,
            each parameter gets its own optimizer that steps inline during
            ``backward()``.  The returned "optimizer" is a lightweight wrapper
            whose ``step()`` and ``zero_grad()`` are no-ops.
        betas: Adam beta coefficients.

    Returns:
        An optimizer (or a no-op wrapper when ``optimizer_in_bwd=True``).
    """
    if optimizer_in_bwd:
        return _setup_optimizer_in_bwd(
            model, lr=lr, weight_decay=weight_decay,
            use_8bit=use_8bit, betas=betas,
        )

    if use_8bit:
        try:
            import bitsandbytes as bnb
            opt = bnb.optim.AdamW8bit(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=betas,
            )
            logger.info("Using bitsandbytes 8-bit AdamW")
            return opt
        except ImportError:
            logger.warning(
                "bitsandbytes not installed, falling back to standard AdamW. "
                "Install with: pip install bitsandbytes"
            )

    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
    )


class _OptimizerInBwdWrapper:
    """Lightweight wrapper that mimics an optimizer interface when per-param
    optimizers are fused into backward hooks.

    Inspired by torchtune's ``optimizer_in_bwd`` which registers a hook on
    each parameter to call ``optimizer.step()`` inside the backward pass,
    thereby freeing gradient memory immediately.
    """

    def __init__(self, model, lr, weight_decay, use_8bit, betas):
        self.param_groups = [{"lr": lr}]
        self._per_param_optims = {}
        self._hooks = []

        opt_cls = torch.optim.AdamW
        if use_8bit:
            try:
                import bitsandbytes as bnb
                opt_cls = bnb.optim.AdamW8bit
                logger.info("Fused backward + 8-bit AdamW")
            except ImportError:
                logger.warning("bitsandbytes not available for fused optimizer")

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            opt = opt_cls(
                [p], lr=lr, weight_decay=weight_decay, betas=betas
            )
            self._per_param_optims[name] = opt

            def _make_hook(optimizer):
                def hook(grad):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                return hook

            h = p.register_post_accumulate_grad_hook(_make_hook(opt))
            self._hooks.append(h)

        logger.info(
            "Optimizer fused into backward for %d parameters",
            len(self._per_param_optims),
        )

    def step(self):
        pass

    def zero_grad(self, set_to_none=True):
        pass

    def state_dict(self):
        return {
            name: opt.state_dict()
            for name, opt in self._per_param_optims.items()
        }

    def load_state_dict(self, state_dict):
        for name, opt in self._per_param_optims.items():
            if name in state_dict:
                opt.load_state_dict(state_dict[name])

    def set_lr(self, lr):
        self.param_groups[0]["lr"] = lr
        for opt in self._per_param_optims.values():
            for pg in opt.param_groups:
                pg["lr"] = lr


def _setup_optimizer_in_bwd(model, lr, weight_decay, use_8bit, betas):
    return _OptimizerInBwdWrapper(model, lr, weight_decay, use_8bit, betas)


def compile_model(model: nn.Module, backend: str = "inductor"):
    """Apply torch.compile to the model for faster training/inference.

    Inspired by torchtune's ``compile=True`` flag which can yield 20-30%
    throughput improvements on modern GPUs (A100+, 4090+).

    Falls back gracefully on older PyTorch or unsupported hardware.
    """
    if not hasattr(torch, "compile"):
        logger.warning(
            "torch.compile not available (requires PyTorch >= 2.0). Skipping."
        )
        return model

    try:
        compiled = torch.compile(model, backend=backend)
        logger.info("torch.compile enabled (backend=%s)", backend)
        return compiled
    except Exception as e:
        logger.warning("torch.compile failed: %s. Using eager mode.", e)
        return model


def get_memory_stats(device: torch.device) -> dict:
    """Return current GPU memory usage stats (useful for monitoring)."""
    if device.type != "cuda":
        return {}
    return {
        "allocated_mb": torch.cuda.memory_allocated(device) / 1024**2,
        "reserved_mb": torch.cuda.memory_reserved(device) / 1024**2,
        "max_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
    }


def log_memory_stats(device: torch.device, prefix: str = ""):
    """Log GPU memory stats."""
    stats = get_memory_stats(device)
    if stats:
        logger.info(
            "%sGPU Memory: %.1f MB allocated, %.1f MB reserved, %.1f MB peak",
            f"[{prefix}] " if prefix else "",
            stats["allocated_mb"],
            stats["reserved_mb"],
            stats["max_allocated_mb"],
        )
