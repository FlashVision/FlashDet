"""Neural network building blocks and training techniques.

Central hub for all reusable NN components. Users can import from here:

    from flashdet.nn import ConvBNSiLU, C3k2, SPPF, ConvModule
    from flashdet.nn import apply_lora, apply_qlora

To add a new building block:
  1. Add the module in ``flashdet/models/layers/``.
  2. Export it here so it's accessible as ``flashdet.nn.MyBlock``.
"""

# ── Building blocks (layers) ──────────────────────────────────────────
from flashdet.models.layers import (
    ConvBNSiLU,
    DownSample,
    ConvModule,
    DepthwiseConvModule,
    RepConv,
    RepVGGBlock,
    C2f,
    C3k2,
    CSPRepLayer,
    ELAN,
    GELAN,
    Bottleneck,
    SPPF,
    PSA,
    C2PSA,
    SCDown,
)

# ── LoRA / QLoRA ──────────────────────────────────────────────────────
from flashdet.models.lora import (
    apply_lora,
    apply_qlora,
    merge_lora_weights,
    get_lora_state_dict,
    LORA_VARIANTS,
    get_variant_description,
    get_ortho_regularization_loss,
    get_lora_plus_param_groups,
)

__all__ = [
    # Layers / blocks
    "ConvBNSiLU",
    "DownSample",
    "ConvModule",
    "DepthwiseConvModule",
    "RepConv",
    "RepVGGBlock",
    "C2f",
    "C3k2",
    "CSPRepLayer",
    "ELAN",
    "GELAN",
    "Bottleneck",
    "SPPF",
    "PSA",
    "C2PSA",
    "SCDown",
    # LoRA
    "apply_lora",
    "apply_qlora",
    "merge_lora_weights",
    "get_lora_state_dict",
    "LORA_VARIANTS",
    "get_variant_description",
    "get_ortho_regularization_loss",
    "get_lora_plus_param_groups",
]
