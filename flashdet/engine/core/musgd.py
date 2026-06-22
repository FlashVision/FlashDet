"""
MuSGD — Hybrid Muon-SGD Optimizer for YOLO26-based FlashDet.

Combines Muon (momentum + Newton-Schulz orthogonalization) with
standard SGD-momentum:
  - Parameters with ndim >= 2 (conv kernels, linear weights): hybrid
    Muon + SGD update with configurable blend weights.
  - Parameters with ndim < 2 (biases, BN scale/shift): pure SGD.

Reference:
    Ultralytics YOLO26 (2026), Section 3.3.1.
    Muon optimizer: Jordan et al., 2024.
"""

import torch
from torch.optim import Optimizer
from typing import Iterable, Dict, Any, List


def _newton_schulz_orthogonalize(
    G: torch.Tensor, steps: int = 5, eps: float = 1e-7
) -> torch.Tensor:
    """Approximate orthogonalization via Newton-Schulz iterations.

    Reshapes to 2D, ensures rows <= cols, then iterates:
        X_{k+1} = (3I - X @ X^T) / 2 @ X

    Works on arbitrary-dim tensors by flattening to 2D and back.
    """
    orig_shape = G.shape
    if G.ndim > 2:
        G = G.reshape(G.shape[0], -1)

    rows, cols = G.shape
    transposed = False
    if rows > cols:
        G = G.T
        rows, cols = cols, rows
        transposed = True

    norm = torch.norm(G, p="fro")
    X = G / (norm + eps)

    I = torch.eye(rows, device=X.device, dtype=X.dtype)
    for _ in range(steps):
        A = X @ X.T  # [rows, rows]
        X = ((3 * I - A) / 2) @ X  # [rows, cols]

    X = X * norm

    if transposed:
        X = X.T

    return X.reshape(orig_shape)


class MuSGD(Optimizer):
    """Hybrid Muon-SGD optimizer.

    For parameter groups with ``use_muon=True`` (default for ndim>=2 params),
    applies a weighted blend of:
      1. Muon update: momentum + Newton-Schulz orthogonalization
      2. SGD update: standard SGD with momentum

    For parameter groups with ``use_muon=False`` (biases, BN), uses pure SGD.

    Args:
        params: Iterable of parameters or param groups.
        lr: Learning rate. Default: 1e-3.
        momentum: Momentum factor. Default: 0.9.
        weight_decay: L2 regularization. Default: 5e-4.
        muon_weight: Blend weight for Muon component. Default: 0.5.
        sgd_weight: Blend weight for SGD component. Default: 0.5.
        nesterov: Use Nesterov momentum for SGD component. Default: False.
        ns_steps: Newton-Schulz iteration count. Default: 5.

    Example::

        muon_params = [p for p in model.parameters() if p.ndim >= 2]
        sgd_params = [p for p in model.parameters() if p.ndim < 2]
        optimizer = MuSGD([
            {"params": muon_params, "use_muon": True},
            {"params": sgd_params, "use_muon": False},
        ], lr=1e-3, momentum=0.9)
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        muon_weight: float = 0.5,
        sgd_weight: float = 0.5,
        nesterov: bool = False,
        ns_steps: int = 5,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            muon_weight=muon_weight,
            sgd_weight=sgd_weight,
            nesterov=nesterov,
            ns_steps=ns_steps,
            use_muon=True,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]
            use_muon = group.get("use_muon", True)
            muon_w = group["muon_weight"]
            sgd_w = group["sgd_weight"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)

                if use_muon and p.ndim >= 2:
                    # Muon: orthogonalize the momentum buffer
                    muon_update = _newton_schulz_orthogonalize(buf.clone(), steps=ns_steps)

                    if nesterov:
                        sgd_update = grad + momentum * buf
                    else:
                        sgd_update = buf

                    # Blended update
                    update = muon_w * muon_update + sgd_w * sgd_update
                    p.add_(update, alpha=-lr)
                else:
                    # Pure SGD for 1D params
                    if nesterov:
                        update = grad + momentum * buf
                    else:
                        update = buf
                    p.add_(update, alpha=-lr)

        return loss


def build_musgd(
    model: torch.nn.Module,
    lr: float = 1e-3,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    muon_weight: float = 0.5,
    sgd_weight: float = 0.5,
    nesterov: bool = False,
    ns_steps: int = 5,
) -> MuSGD:
    """Convenience builder that auto-splits model parameters into
    Muon-eligible (ndim>=2) and SGD-only (ndim<2) groups.

    Args:
        model: The nn.Module to optimize.
        lr: Learning rate.
        momentum: Momentum factor.
        weight_decay: L2 weight decay (only on Muon params).
        muon_weight: Muon blend factor.
        sgd_weight: SGD blend factor.
        nesterov: Nesterov momentum.
        ns_steps: Newton-Schulz iterations.

    Returns:
        Configured MuSGD optimizer.
    """
    muon_params = []
    sgd_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2:
            muon_params.append(p)
        else:
            sgd_params.append(p)

    param_groups = [
        {
            "params": muon_params,
            "use_muon": True,
            "weight_decay": weight_decay,
        },
        {
            "params": sgd_params,
            "use_muon": False,
            "weight_decay": 0.0,
        },
    ]

    return MuSGD(
        param_groups,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        muon_weight=muon_weight,
        sgd_weight=sgd_weight,
        nesterov=nesterov,
        ns_steps=ns_steps,
    )
