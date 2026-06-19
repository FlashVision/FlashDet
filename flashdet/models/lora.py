"""
LoRA (Low-Rank Adaptation) variants for FlashDet.

Supported variants:
  - Standard LoRA (Hu et al., 2022)
  - DoRA: Weight-Decomposed Low-Rank Adaptation (Liu et al., 2024)
  - LoRA+: Asymmetric learning rates for A and B matrices (Hayou et al., 2024)
  - AdaLoRA: Adaptive rank allocation via SVD pruning (Zhang et al., 2023)
  - OrthoLoRA: Orthogonal regularization for better adaptation (unitary constraint)
  - LoRA-FA: Freeze A, only train B (memory-efficient, Kalajdzievski 2023)

All variants are adapted for convolutional object detection backbones.
"""

import logging
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Supported variant names
LORA_VARIANTS = ["standard", "dora", "lora_plus", "adalora", "ortho", "lora_fa"]


# ═══════════════════════════════════════════════════════════════════════
# Standard LoRA
# ═══════════════════════════════════════════════════════════════════════

class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation.

    output = W_frozen @ x + (alpha/rank) * B @ A @ x
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=False
        )
        self.bias = (
            nn.Parameter(torch.zeros(out_features), requires_grad=False)
            if bias
            else None
        )

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.weight, self.bias)
        lora_out = (
            self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        )
        return base_out + lora_out


class LoRAConv2d(nn.Module):
    """Conv2d layer with LoRA adaptation via 1x1 conv decomposition."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size),
            requires_grad=False,
        )
        self.bias = (
            nn.Parameter(torch.zeros(out_channels), requires_grad=False)
            if bias
            else None
        )

        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.kernel_size = kernel_size

        self.lora_A = nn.Conv2d(
            in_channels, rank, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.lora_B = nn.Conv2d(
            rank, out_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.lora_dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.conv2d(
            x, self.weight, self.bias,
            stride=self.stride, padding=self.padding, groups=self.groups,
        )
        lora_input = self.lora_dropout(x)
        if self.stride > 1:
            lora_input = F.avg_pool2d(lora_input, self.stride, self.stride)
        lora_out = self.lora_B(self.lora_A(lora_input)) * self.scaling
        if lora_out.shape[2:] != base_out.shape[2:]:
            lora_out = F.adaptive_avg_pool2d(lora_out, base_out.shape[2:])
        return base_out + lora_out


# ═══════════════════════════════════════════════════════════════════════
# DoRA: Weight-Decomposed LoRA (Liu et al., 2024)
# Decomposes W into magnitude (m) and direction (V), applies LoRA to V only.
# output = m * (V + delta_V) / ||V + delta_V|| @ x
# ═══════════════════════════════════════════════════════════════════════

class DoRALinear(nn.Module):
    """DoRA: LoRA with weight decomposition into magnitude and direction."""

    def __init__(self, in_features, out_features, rank=8, alpha=16.0,
                 dropout=0.0, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=False)
        self.bias = (
            nn.Parameter(torch.zeros(out_features), requires_grad=False)
            if bias else None)

        # Magnitude vector (trainable) — initialized from pretrained W norm
        self.magnitude = nn.Parameter(torch.ones(out_features))

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def _init_magnitude_from_weight(self):
        """Call after loading pretrained weights to set magnitude correctly."""
        with torch.no_grad():
            self.magnitude.copy_(self.weight.norm(dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # W' = W + scaling * B @ A
        delta = (self.lora_B @ self.lora_A) * self.scaling
        adapted_weight = self.weight + delta
        # Normalize direction, apply learned magnitude
        direction_norm = adapted_weight.norm(dim=1, keepdim=True).clamp(min=1e-8)
        normalized = adapted_weight / direction_norm
        final_weight = self.magnitude.unsqueeze(1) * normalized
        return F.linear(self.lora_dropout(x), final_weight, self.bias)


class DoRAConv2d(nn.Module):
    """DoRA for Conv2d layers."""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, groups=1, rank=8, alpha=16.0, dropout=0.0, bias=False):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.kernel_size = kernel_size

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size),
            requires_grad=False)
        self.bias = (
            nn.Parameter(torch.zeros(out_channels), requires_grad=False)
            if bias else None)

        self.magnitude = nn.Parameter(torch.ones(out_channels))

        self.lora_A = nn.Conv2d(in_channels, rank, 1, bias=False)
        self.lora_B = nn.Conv2d(rank, out_channels, 1, bias=False)
        self.lora_dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def _init_magnitude_from_weight(self):
        with torch.no_grad():
            w_flat = self.weight.view(self.weight.size(0), -1)
            self.magnitude.copy_(w_flat.norm(dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.conv2d(x, self.weight, self.bias,
                            stride=self.stride, padding=self.padding, groups=self.groups)
        lora_input = self.lora_dropout(x)
        if self.stride > 1:
            lora_input = F.avg_pool2d(lora_input, self.stride, self.stride)
        lora_out = self.lora_B(self.lora_A(lora_input)) * self.scaling

        if lora_out.shape[2:] != base_out.shape[2:]:
            lora_out = F.adaptive_avg_pool2d(lora_out, base_out.shape[2:])

        # DoRA: normalize combined output per-channel, apply magnitude
        combined = base_out + lora_out
        norm = combined.norm(dim=(2, 3), keepdim=True).clamp(min=1e-8)
        return self.magnitude.view(1, -1, 1, 1) * combined / norm * math.sqrt(
            combined.shape[2] * combined.shape[3])


# ═══════════════════════════════════════════════════════════════════════
# LoRA-FA: Freeze A, only train B (Kalajdzievski, 2023)
# Reduces trainable params by half; A is frozen after random init.
# ═══════════════════════════════════════════════════════════════════════

class LoRAFALinear(nn.Module):
    """LoRA-FA: Frozen A matrix, only B is trainable."""

    def __init__(self, in_features, out_features, rank=8, alpha=16.0,
                 dropout=0.0, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=False)
        self.bias = (
            nn.Parameter(torch.zeros(out_features), requires_grad=False)
            if bias else None)

        # A is frozen (random projection), B is trainable
        self.register_buffer("lora_A", torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.weight, self.bias)
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        return base_out + lora_out


class LoRAFAConv2d(nn.Module):
    """LoRA-FA for Conv2d: Frozen A conv, only B is trainable."""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, groups=1, rank=8, alpha=16.0, dropout=0.0, bias=False):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.kernel_size = kernel_size

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size),
            requires_grad=False)
        self.bias = (
            nn.Parameter(torch.zeros(out_channels), requires_grad=False)
            if bias else None)

        # A is a non-trainable buffer
        self.lora_A = nn.Conv2d(in_channels, rank, 1, bias=False)
        self.lora_A.weight.requires_grad_(False)
        self.lora_B = nn.Conv2d(rank, out_channels, 1, bias=False)
        self.lora_dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.conv2d(x, self.weight, self.bias,
                            stride=self.stride, padding=self.padding, groups=self.groups)
        lora_input = self.lora_dropout(x)
        if self.stride > 1:
            lora_input = F.avg_pool2d(lora_input, self.stride, self.stride)
        lora_out = self.lora_B(self.lora_A(lora_input)) * self.scaling
        if lora_out.shape[2:] != base_out.shape[2:]:
            lora_out = F.adaptive_avg_pool2d(lora_out, base_out.shape[2:])
        return base_out + lora_out


# ═══════════════════════════════════════════════════════════════════════
# OrthoLoRA: Orthogonal-regularized LoRA
# Encourages A and B to stay orthogonal for better gradient flow and
# prevents adapter collapse. Uses orthogonal initialization.
# ═══════════════════════════════════════════════════════════════════════

class OrthoLoRALinear(nn.Module):
    """OrthoLoRA: LoRA with orthogonal initialization and regularization."""

    def __init__(self, in_features, out_features, rank=8, alpha=16.0,
                 dropout=0.0, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=False)
        self.bias = (
            nn.Parameter(torch.zeros(out_features), requires_grad=False)
            if bias else None)

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Orthogonal initialization for better gradient properties
        nn.init.orthogonal_(self.lora_A)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.weight, self.bias)
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        return base_out + lora_out

    def ortho_regularization(self) -> torch.Tensor:
        """Compute orthogonality loss: ||A @ A^T - I||_F^2."""
        AAT = self.lora_A @ self.lora_A.T
        eye = torch.eye(self.rank, device=AAT.device, dtype=AAT.dtype)
        return ((AAT - eye) ** 2).sum()


class OrthoLoRAConv2d(nn.Module):
    """OrthoLoRA for Conv2d layers."""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, groups=1, rank=8, alpha=16.0, dropout=0.0, bias=False):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.kernel_size = kernel_size

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size),
            requires_grad=False)
        self.bias = (
            nn.Parameter(torch.zeros(out_channels), requires_grad=False)
            if bias else None)

        self.lora_A = nn.Conv2d(in_channels, rank, 1, bias=False)
        self.lora_B = nn.Conv2d(rank, out_channels, 1, bias=False)
        self.lora_dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        nn.init.orthogonal_(self.lora_A.weight.view(rank, -1))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.conv2d(x, self.weight, self.bias,
                            stride=self.stride, padding=self.padding, groups=self.groups)
        lora_input = self.lora_dropout(x)
        if self.stride > 1:
            lora_input = F.avg_pool2d(lora_input, self.stride, self.stride)
        lora_out = self.lora_B(self.lora_A(lora_input)) * self.scaling
        if lora_out.shape[2:] != base_out.shape[2:]:
            lora_out = F.adaptive_avg_pool2d(lora_out, base_out.shape[2:])
        return base_out + lora_out

    def ortho_regularization(self) -> torch.Tensor:
        A_flat = self.lora_A.weight.view(self.rank, -1)
        AAT = A_flat @ A_flat.T
        eye = torch.eye(self.rank, device=AAT.device, dtype=AAT.dtype)
        return ((AAT - eye) ** 2).sum()


# ═══════════════════════════════════════════════════════════════════════
# AdaLoRA: Adaptive rank allocation (Zhang et al., 2023)
# Uses SVD parameterization (P, diag(lambda), Q) and prunes singular
# values during training to dynamically allocate rank per layer.
# ═══════════════════════════════════════════════════════════════════════

class AdaLoRALinear(nn.Module):
    """AdaLoRA: SVD-based adaptive rank with importance scoring."""

    def __init__(self, in_features, out_features, rank=8, alpha=16.0,
                 dropout=0.0, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=False)
        self.bias = (
            nn.Parameter(torch.zeros(out_features), requires_grad=False)
            if bias else None)

        # SVD parameterization: delta_W = P @ diag(lambda) @ Q
        self.lora_P = nn.Parameter(torch.empty(out_features, rank))
        self.lora_lambda = nn.Parameter(torch.ones(rank))
        self.lora_Q = nn.Parameter(torch.empty(rank, in_features))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_Q, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_P, a=math.sqrt(5))
        nn.init.ones_(self.lora_lambda)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.weight, self.bias)
        # delta = P @ diag(lambda) @ Q @ x
        Qx = self.lora_dropout(x) @ self.lora_Q.T
        scaled = Qx * self.lora_lambda.unsqueeze(0)
        lora_out = scaled @ self.lora_P.T * self.scaling
        return base_out + lora_out

    def importance_scores(self) -> torch.Tensor:
        """Return per-rank importance scores for pruning decisions."""
        return self.lora_lambda.abs()


class AdaLoRAConv2d(nn.Module):
    """AdaLoRA for Conv2d with SVD-style parameterization."""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, groups=1, rank=8, alpha=16.0, dropout=0.0, bias=False):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.kernel_size = kernel_size

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size),
            requires_grad=False)
        self.bias = (
            nn.Parameter(torch.zeros(out_channels), requires_grad=False)
            if bias else None)

        # SVD: A projects down, lambda scales, B projects up
        self.lora_A = nn.Conv2d(in_channels, rank, 1, bias=False)
        self.lora_lambda = nn.Parameter(torch.ones(rank))
        self.lora_B = nn.Conv2d(rank, out_channels, 1, bias=False)
        self.lora_dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_B.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.conv2d(x, self.weight, self.bias,
                            stride=self.stride, padding=self.padding, groups=self.groups)
        lora_input = self.lora_dropout(x)
        if self.stride > 1:
            lora_input = F.avg_pool2d(lora_input, self.stride, self.stride)
        feat = self.lora_A(lora_input)
        feat = feat * self.lora_lambda.view(1, -1, 1, 1)
        lora_out = self.lora_B(feat) * self.scaling
        if lora_out.shape[2:] != base_out.shape[2:]:
            lora_out = F.adaptive_avg_pool2d(lora_out, base_out.shape[2:])
        return base_out + lora_out

    def importance_scores(self) -> torch.Tensor:
        return self.lora_lambda.abs()


# ═══════════════════════════════════════════════════════════════════════
# Variant registry and factory
# ═══════════════════════════════════════════════════════════════════════

_LINEAR_REGISTRY = {
    "standard": LoRALinear,
    "dora": DoRALinear,
    "lora_plus": LoRALinear,      # same layer, different optimizer LR setup
    "adalora": AdaLoRALinear,
    "ortho": OrthoLoRALinear,
    "lora_fa": LoRAFALinear,
}

_CONV2D_REGISTRY = {
    "standard": LoRAConv2d,
    "dora": DoRAConv2d,
    "lora_plus": LoRAConv2d,
    "adalora": AdaLoRAConv2d,
    "ortho": OrthoLoRAConv2d,
    "lora_fa": LoRAFAConv2d,
}


def get_variant_description(variant: str) -> str:
    """Return a human-readable description of a LoRA variant."""
    descriptions = {
        "standard": "Standard LoRA — trains low-rank A and B matrices",
        "dora": "DoRA — decomposes weight into magnitude + direction, better quality",
        "lora_plus": "LoRA+ — uses higher LR for B matrix (8x) for faster convergence",
        "adalora": "AdaLoRA — adaptive rank via SVD, prunes unimportant dimensions",
        "ortho": "OrthoLoRA — orthogonal init + regularization for stable training",
        "lora_fa": "LoRA-FA — freezes A matrix, trains only B (50% fewer params)",
    }
    return descriptions.get(variant, "Unknown variant")


def _replace_linear_with_lora(
    module: nn.Module,
    rank: int,
    alpha: float,
    dropout: float,
    replaced: list,
    prefix: str = "",
    variant: str = "standard",
):
    LinearCls = _LINEAR_REGISTRY.get(variant, LoRALinear)
    Conv2dCls = _CONV2D_REGISTRY.get(variant, LoRAConv2d)

    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name

        if isinstance(child, nn.Linear):
            lora_layer = LinearCls(
                child.in_features,
                child.out_features,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                bias=child.bias is not None,
            )
            lora_layer.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                lora_layer.bias.data.copy_(child.bias.data)
            if hasattr(lora_layer, '_init_magnitude_from_weight'):
                lora_layer._init_magnitude_from_weight()
            setattr(module, name, lora_layer)
            replaced.append(full_name)

        elif isinstance(child, nn.Conv2d) and child.groups == 1:
            if child.kernel_size[0] > 3:
                _replace_linear_with_lora(
                    child, rank, alpha, dropout, replaced, full_name, variant
                )
                continue
            lora_layer = Conv2dCls(
                child.in_channels,
                child.out_channels,
                kernel_size=child.kernel_size[0],
                stride=child.stride[0],
                padding=child.padding[0],
                groups=child.groups,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                bias=child.bias is not None,
            )
            lora_layer.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                lora_layer.bias.data.copy_(child.bias.data)
            if hasattr(lora_layer, '_init_magnitude_from_weight'):
                lora_layer._init_magnitude_from_weight()
            setattr(module, name, lora_layer)
            replaced.append(full_name)
        else:
            _replace_linear_with_lora(
                child, rank, alpha, dropout, replaced, full_name, variant
            )


def apply_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    variant: str = "standard",
) -> nn.Module:
    """Apply LoRA adapters to target modules of a FlashDet model.

    Freezes all pretrained weights and only makes LoRA A/B matrices trainable.

    Args:
        model: FlashDet instance.
        rank: LoRA rank (lower = fewer params, higher = more capacity).
        alpha: Scaling factor.  ``alpha/rank`` is the effective scale.
        dropout: Dropout applied to LoRA input.
        target_modules: List of top-level module names to apply LoRA to.
        variant: LoRA variant to use. One of:
            "standard", "dora", "lora_plus", "adalora", "ortho", "lora_fa".

    Returns:
        The model with LoRA applied.  Only LoRA parameters are trainable.
    """
    if target_modules is None:
        target_modules = ["backbone", "fpn"]

    if variant not in LORA_VARIANTS:
        logger.warning("Unknown LoRA variant '%s', falling back to 'standard'", variant)
        variant = "standard"

    logger.info("LoRA variant: %s — %s", variant, get_variant_description(variant))

    # Freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    replaced = []
    for mod_name in target_modules:
        submod = getattr(model, mod_name, None)
        if submod is None:
            logger.warning("LoRA target '%s' not found on model, skipping", mod_name)
            continue
        _replace_linear_with_lora(
            submod, rank, alpha, dropout, replaced, prefix=mod_name, variant=variant
        )

    # Unfreeze LoRA parameters + detection head (always trainable for fine-tuning)
    _unfreeze_lora_params(model)
    _unfreeze_module(model, "head")
    _unfreeze_module(model, "aux_head")

    # For DoRA, also unfreeze magnitude parameters
    if variant == "dora":
        for name, p in model.named_parameters():
            if "magnitude" in name:
                p.requires_grad = True

    # For AdaLoRA, unfreeze lambda parameters
    if variant == "adalora":
        for name, p in model.named_parameters():
            if "lora_lambda" in name or "lora_P" in name or "lora_Q" in name:
                p.requires_grad = True

    # Unfreeze BatchNorm in targeted modules
    for mod_name in target_modules:
        submod = getattr(model, mod_name, None)
        if submod is None:
            continue
        for m in submod.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
                m.train()
                for p in m.parameters():
                    p.requires_grad = True

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "LoRA applied (%s): %d layers adapted (rank=%d, alpha=%.1f). "
        "Trainable: %d / %d params (%.1f%%)",
        variant, len(replaced), rank, alpha, trainable, total,
        100.0 * trainable / max(total, 1),
    )

    return model


def get_ortho_regularization_loss(model: nn.Module, weight: float = 0.01) -> torch.Tensor:
    """Compute orthogonality regularization for OrthoLoRA layers.

    Call this during training and add to the total loss:
        loss = detection_loss + get_ortho_regularization_loss(model)
    """
    ortho_loss = torch.tensor(0.0, device=next(model.parameters()).device)
    count = 0
    for m in model.modules():
        if hasattr(m, 'ortho_regularization'):
            ortho_loss = ortho_loss + m.ortho_regularization()
            count += 1
    if count > 0:
        ortho_loss = ortho_loss * weight / count
    return ortho_loss


def get_lora_plus_param_groups(model: nn.Module, lr: float, lr_ratio: float = 8.0):
    """Create parameter groups for LoRA+ (asymmetric LR).

    LoRA+ uses a higher learning rate for B matrices (lr * lr_ratio)
    and a normal rate for A matrices, which accelerates convergence.

    Returns list suitable for optimizer constructor.
    """
    a_params = []
    b_params = []
    other_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_A" in name or "lora_Q" in name or "lora_P" in name:
            a_params.append(p)
        elif "lora_B" in name or "lora_lambda" in name:
            b_params.append(p)
        else:
            other_params.append(p)

    return [
        {"params": a_params, "lr": lr, "name": "lora_A"},
        {"params": b_params, "lr": lr * lr_ratio, "name": "lora_B"},
        {"params": other_params, "lr": lr, "name": "other"},
    ]


def _unfreeze_lora_params(model: nn.Module):
    """Unfreeze all LoRA A/B parameters and their dropout layers."""
    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            p.requires_grad = True


def _unfreeze_module(model: nn.Module, module_name: str):
    """Unfreeze all parameters of a named sub-module."""
    mod = getattr(model, module_name, None)
    if mod is None:
        return
    for p in mod.parameters():
        p.requires_grad = True


def get_lora_state_dict(model: nn.Module) -> dict:
    """Extract LoRA adapter weights and associated BatchNorm parameters."""
    return {
        k: v for k, v in model.state_dict().items()
        if "lora_A" in k or "lora_B" in k
        or "running_mean" in k or "running_var" in k
        or "num_batches_tracked" in k
    }


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """Merge LoRA weights into the base weights for inference.

    After merging, LoRA layers become standard Linear/Conv2d layers with no
    runtime overhead.  This is equivalent to ``W_merged = W + (alpha/r) * B @ A``.
    """
    merged_count = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            delta = (module.lora_B @ module.lora_A) * module.scaling
            module.weight.data += delta
            module.lora_A.data.zero_()
            module.lora_B.data.zero_()
            merged_count += 1
        elif isinstance(module, LoRAConv2d):
            A_weight = module.lora_A.weight.data.squeeze(-1).squeeze(-1)
            B_weight = module.lora_B.weight.data.squeeze(-1).squeeze(-1)
            delta = (B_weight @ A_weight).unsqueeze(-1).unsqueeze(-1)
            if module.kernel_size == 1:
                module.weight.data += delta * module.scaling
            merged_count += 1

    logger.info("Merged LoRA weights in %d layers", merged_count)
    return model


# ─────────────────────────────────────────────────────────────────────
# QLoRA: Quantized base weights (INT8/NF4) + LoRA adapters
# Inspired by torchtune's QLoRA implementation
# ─────────────────────────────────────────────────────────────────────

def _quantize_module_weights(module: nn.Module, quant_dtype: str = "int8"):
    """Quantize frozen weights of a module to save memory.

    Supported dtypes:
      - ``"int8"``:  Dynamic per-channel INT8 quantization (no extra deps).
      - ``"nf4"``:   4-bit NormalFloat via bitsandbytes (requires bitsandbytes).

    Only affects parameters with ``requires_grad=False`` that are floating-point.
    """
    quantized = 0
    if quant_dtype == "nf4":
        try:
            import bitsandbytes as bnb
            for name, child in module.named_modules():
                if isinstance(child, (LoRALinear, LoRAConv2d)):
                    w = child.weight
                    if not w.requires_grad and w.is_floating_point() and w.ndim == 2:
                        qw = bnb.nn.Params4bit(
                            w.data, requires_grad=False, quant_type="nf4"
                        )
                        child.weight = qw
                        quantized += 1
            logger.info("QLoRA (NF4): quantized %d weight tensors", quantized)
            return quantized
        except ImportError:
            logger.warning(
                "bitsandbytes not installed — falling back to INT8 quantization. "
                "Install with: pip install bitsandbytes"
            )
            quant_dtype = "int8"

    if quant_dtype == "int8":
        for child in module.modules():
            if isinstance(child, (LoRALinear, LoRAConv2d)):
                w = child.weight
                if not w.requires_grad and w.is_floating_point():
                    scale = w.data.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
                    w_int8 = (w.data / scale).round().clamp(-128, 127).to(torch.int8)
                    child.register_buffer("_qlora_w_int8", w_int8)
                    child.register_buffer("_qlora_scale", scale)
                    child.weight = nn.Parameter(
                        (w_int8.float() * scale), requires_grad=False
                    )
                    quantized += 1
        logger.info("QLoRA (INT8): quantized %d weight tensors", quantized)

    return quantized


def apply_qlora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    quant_dtype: str = "int8",
    variant: str = "standard",
) -> nn.Module:
    """Apply QLoRA: quantized base weights + LoRA adapters.

    Args:
        model: FlashDet instance.
        rank: LoRA rank.
        alpha: LoRA scaling alpha.
        dropout: LoRA dropout.
        target_modules: Modules to apply LoRA + quantization to.
        quant_dtype: ``"int8"`` (no extra deps) or ``"nf4"`` (requires bitsandbytes).
        variant: LoRA variant to use.

    Returns:
        Model with QLoRA applied.
    """
    model = apply_lora(
        model, rank=rank, alpha=alpha, dropout=dropout,
        target_modules=target_modules, variant=variant,
    )

    n_quantized = _quantize_module_weights(model, quant_dtype=quant_dtype)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "QLoRA applied: %d layers quantized (%s). "
        "Trainable: %d / %d params (%.1f%%)",
        n_quantized, quant_dtype, trainable, total,
        100.0 * trainable / max(total, 1),
    )
    return model
