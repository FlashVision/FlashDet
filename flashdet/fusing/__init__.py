"""Fusing package: layer fusion, kernel fusion, graph passes.

Example — fuse a checkpoint for deployment and get a change report:

    from flashdet.cfg import get_config
    from flashdet.fusing import run_fusions, default_passes

    model = ...                      # small FlashDet-P model
    gm, report = run_fusions(model, default_passes(), example_input=torch.randn(1,3,128,128))
    print(report)

``run_fusions`` deep-copies the input model, applies every pass, and verifies
numeric parity against a checkpoint-free copy before returning.
"""

from flashdet.fusing.base import (
    FusionPass,
    FusionReport,
    module_op_counts,
    param_count,
    run_fusions,
)
from flashdet.fusing.layer_fusing import (
    ConvBnFusionPass,
    LinearBnFusionPass,
    LinearPairFusionPass,
    RepFusionPass,
    fold_bn_into_conv,
    fold_bn_into_linear,
    merge_linears,
)
from flashdet.fusing.kernel_fusing import (
    ConvBnActFusionPass,
    FusedConvBnAct,
    FusedDWPointwise,
    fused_residual_act,
)

__all__ = [
    "FusionPass",
    "FusionReport",
    "module_op_counts",
    "param_count",
    "run_fusions",
    "default_passes",
    "ConvBnFusionPass",
    "LinearBnFusionPass",
    "LinearPairFusionPass",
    "RepFusionPass",
    "ConvBnActFusionPass",
    "FusedConvBnAct",
    "FusedDWPointwise",
    "fused_residual_act",
    "fold_bn_into_conv",
    "fold_bn_into_linear",
    "merge_linears",
]


def default_passes(example_input=None) -> list[FusionPass]:
    """Recommended pass order for deployment.

    Structural re-parameterization first (blocks collapse to simple convs so
    the graph passes find clean conv→BN patterns), then kernel fusion and
    layer folding.
    """
    return [
        RepFusionPass(),
        ConvBnActFusionPass(example_input=example_input),
        ConvBnFusionPass(example_input=example_input),
        LinearPairFusionPass(example_input=example_input),
        LinearBnFusionPass(example_input=example_input),
    ]
