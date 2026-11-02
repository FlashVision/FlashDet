"""Tests for the fusing package: layer folding, kernel fusion, graph passes."""


import pytest
import torch
import torch.nn as nn

from flashdet.fusing import (
    ConvBnActFusionPass,
    ConvBnFusionPass,
    FusedConvBnAct,
    LinearBnFusionPass,
    LinearPairFusionPass,
    RepFusionPass,
    default_passes,
    fold_bn_into_conv,
    fold_bn_into_linear,
    merge_linears,
    module_op_counts,
    param_count,
    run_fusions,
)


# ---------------------------------------------------------------------------
# Fold math (weight-level equivalence)
# ---------------------------------------------------------------------------


def test_fold_bn_into_conv_matches_sequential():
    torch.manual_seed(0)
    conv = nn.Conv2d(6, 10, 3, padding=1, bias=False)
    bn = nn.BatchNorm2d(10)
    x = torch.randn(2, 6, 12, 12)

    conv.eval()
    bn.eval()
    with torch.no_grad():
        expected = bn(conv(x))
        fold_bn_into_conv(conv, bn)
        actual = conv(x)
    assert torch.allclose(actual, expected, atol=1e-5)


def test_fold_bn_into_conv_existing_bias_and_depthwise():
    torch.manual_seed(0)
    conv = nn.Conv2d(8, 8, 3, padding=1, groups=8, bias=True)
    bn = nn.BatchNorm2d(8)
    x = torch.randn(2, 8, 10, 10)

    conv.eval()
    bn.eval()
    with torch.no_grad():
        expected = bn(conv(x))
        fold_bn_into_conv(conv, bn)
        actual = conv(x)
    assert torch.allclose(actual, expected, atol=1e-5)


def test_fold_bn_into_linear_matches_sequential():
    torch.manual_seed(0)
    linear = nn.Linear(16, 8, bias=False)
    bn = nn.BatchNorm1d(8)
    x = torch.randn(5, 16)

    linear.eval()
    bn.eval()
    with torch.no_grad():
        expected = bn(linear(x))
        fold_bn_into_linear(linear, bn)
        actual = linear(x)
    assert torch.allclose(actual, expected, atol=1e-5)


def test_merge_linears_matches_sequential():
    torch.manual_seed(0)
    first = nn.Linear(8, 6, bias=True)
    second = nn.Linear(6, 4, bias=True)
    x = torch.randn(3, 8)

    merged = merge_linears(first, second)
    with torch.no_grad():
        expected = second(first(x))
        actual = merged(x)
    assert torch.allclose(actual, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Graph passes on small hand-written blocks
# ---------------------------------------------------------------------------


class _Triple(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(8)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class _SharedAct(nn.Module):
    """One activation module used at two call sites (like StrideDown)."""

    def __init__(self):
        super().__init__()
        self.conv_a = nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.bn_a = nn.BatchNorm2d(8)
        self.conv_b = nn.Conv2d(8, 8, 3, padding=1, bias=False)
        self.bn_b = nn.BatchNorm2d(8)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        y = self.act(self.bn_a(self.conv_a(x)))
        z = self.act(self.bn_b(self.conv_b(y)))
        return z + y


def test_conv_bn_pass_parity():
    model = _Triple()
    x = torch.randn(2, 3, 16, 16)
    fused, report = run_fusions(
        model, [ConvBnFusionPass(example_input=x)], example_input=x)

    assert report.parity_ok, report.parity_max_abs_diff
    assert report.passes[0]["fused"] == 1
    assert module_op_counts(fused).get("batchnorm", 0) == 0
    assert param_count(fused) < param_count(model)


def test_conv_bn_act_fuses_triple_into_one_module():
    model = _Triple()
    x = torch.randn(2, 3, 16, 16)
    fused, report = run_fusions(
        model, [ConvBnActFusionPass(example_input=x)], example_input=x)

    assert report.parity_ok, report.parity_max_abs_diff
    assert report.passes[0]["fused"] == 1
    assert isinstance(fused.conv, FusedConvBnAct)
    assert isinstance(fused.bn, nn.Identity)
    assert isinstance(fused.act, nn.Identity)
    assert sum(1 for m in fused.modules() if isinstance(m, FusedConvBnAct)) == 1


def test_shared_activation_is_not_corrupted():
    """A module reused at two call sites must be left untouched by fusion."""
    model = _SharedAct()
    x = torch.randn(2, 3, 16, 16)
    fused, report = run_fusions(
        model, [ConvBnActFusionPass(example_input=x)], example_input=x)

    assert report.parity_ok, report.parity_max_abs_diff
    # The activation is shared, so it cannot be absorbed; conv+BN still folds.
    assert isinstance(fused.act, nn.LeakyReLU)
    assert module_op_counts(fused).get("batchnorm", 0) == 0
    assert not any(isinstance(m, FusedConvBnAct) for m in fused.modules())


def test_aliased_module_is_not_corrupted():
    """One activation object registered under two names counts as shared."""

    class _AliasedAct(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, 3, padding=1, bias=False)
            self.bn = nn.BatchNorm2d(8)
            self.act = nn.LeakyReLU(0.1, inplace=True)
            self.act2 = self.act  # alias: same module, second path

        def forward(self, x):
            z = self.act(self.bn(self.conv(x)))  # triple candidate
            return z + self.act2(z)              # same object, second path

    model = _AliasedAct()
    x = torch.randn(2, 3, 16, 16)
    fused, report = run_fusions(
        model, [ConvBnActFusionPass(example_input=x)], example_input=x)

    assert report.parity_ok, report.parity_max_abs_diff
    assert isinstance(fused.act, nn.LeakyReLU)
    assert fused.act2 is fused.act
    assert not any(isinstance(m, FusedConvBnAct) for m in fused.modules())
    assert module_op_counts(fused).get("batchnorm", 0) == 0


def test_linear_passes_merge_stacked_linears():
    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 6)
            self.bn = nn.BatchNorm1d(6)
            self.fc2 = nn.Linear(6, 4)

        def forward(self, x):
            return self.fc2(self.bn(self.fc1(x)))

    model = Head()
    x = torch.randn(5, 8)
    fused, report = run_fusions(
        model,
        [LinearBnFusionPass(example_input=x),
         LinearPairFusionPass(example_input=x)],
        example_input=x,
    )
    assert report.parity_ok, report.parity_max_abs_diff
    assert isinstance(fused.fc1, nn.Linear)
    assert isinstance(fused.fc2, nn.Identity)
    assert fused.fc1.in_features == 8 and fused.fc1.out_features == 4


# ---------------------------------------------------------------------------
# Whole-detector behaviour
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_flashdet():
    from flashdet.cfg import get_config
    from flashdet.models.detector import build_model

    config = get_config(num_classes=4)
    config.model.size = "p"
    model = build_model(config, architecture="flashdet").eval()
    return model, torch.randn(1, 3, 96, 96)


def test_default_passes_keep_detector_parity(tiny_flashdet):
    model, x = tiny_flashdet
    fused, report = run_fusions(
        model, default_passes(example_input=x), example_input=x)

    assert report.parity_ok, report.parity_max_abs_diff
    assert report.parity_max_abs_diff < 5e-3
    folded = sum(p.get("fused", 0) for p in report.passes)
    assert folded > 0
    assert module_op_counts(fused).get("batchnorm", 0) < module_op_counts(model).get("batchnorm", 0)


def test_run_fusions_does_not_mutate_input(tiny_flashdet):
    model, x = tiny_flashdet
    before = {k: v.clone() for k, v in model.state_dict().items()}
    run_fusions(model, default_passes(example_input=x), example_input=x)

    after = model.state_dict()
    assert set(before) == set(after)
    for key, value in before.items():
        assert torch.equal(value, after[key]), key


def test_rep_fusion_on_reparameterizable_block():
    from flashdet.models.layers.reparam import FusedDWConv

    block = FusedDWConv(8).eval()
    x = torch.randn(1, 8, 12, 12)
    fused, report = run_fusions(
        block, [RepFusionPass()], example_input=x)

    assert report.parity_ok, report.parity_max_abs_diff
    assert report.passes[0]["fused"] == 1
    assert not hasattr(fused, "conv_1x3")


# ---------------------------------------------------------------------------
# Fused checkpoint round-trip
# ---------------------------------------------------------------------------


def test_save_rejects_unknown_pass(tmp_path):
    from flashdet.fusing.base import FusionReport
    from flashdet.fusing.checkpoint import save_fused_checkpoint

    report = FusionReport(passes=[{"name": "not_a_pass", "fused": 1}])
    with pytest.raises(ValueError, match="unknown pass"):
        save_fused_checkpoint(
            nn.Linear(2, 2), str(tmp_path / "bad.pth"),
            config={}, input_size=32, report=report,
        )


def test_fused_checkpoint_roundtrip(tmp_path):
    from flashdet.deployment.onnx_export import build_model_from_checkpoint
    from flashdet.fusing.checkpoint import save_fused_checkpoint
    from flashdet.utils.checkpoint import safe_torch_load

    from flashdet.cfg import get_config
    from flashdet.models.detector import build_model

    config = get_config(num_classes=4)
    config.model.size = "p"
    model = build_model(config, architecture="flashdet").eval()
    cfg = {"architecture": "flashdet", "num_classes": 4,
           "model_size": "p", "input_size": 96, "backbone_type": "lite"}
    ckpt_path = tmp_path / "raw.pth"
    torch.save({"model_state_dict": model.state_dict(), "config": cfg},
               ckpt_path)

    raw = build_model_from_checkpoint(str(ckpt_path), fuse=False)
    probe = torch.randn(1, 3, 96, 96)
    fused, report = run_fusions(
        raw, default_passes(example_input=probe), example_input=probe)
    assert report.parity_ok, report.parity_max_abs_diff

    out_path = tmp_path / "fused.pth"
    save_fused_checkpoint(fused, str(out_path), config=cfg,
                          input_size=96, report=report)

    payload = safe_torch_load(str(out_path))
    assert payload["fusion"]["input_size"] == 96
    assert payload["fusion"]["passes"] == [p["name"] for p in report.passes]

    reloaded = build_model_from_checkpoint(str(out_path))
    saved_state = payload["model_state_dict"]
    reloaded_state = reloaded.state_dict()
    # Training-only heads are intentionally not saved.
    assert set(saved_state) <= set(reloaded_state)
    assert not any(k.startswith(("aux_head.", "aux_heads."))
                   for k in saved_state)
    for key in saved_state:
        assert torch.equal(saved_state[key], reloaded_state[key]), key

    with torch.no_grad():
        a, b = fused(probe), reloaded(probe)
    if isinstance(a, dict):
        keys = [k for k in a if torch.is_tensor(a[k])]
        assert keys
        for key in keys:
            assert torch.allclose(a[key], b[key], atol=1e-6), key
    else:
        assert torch.allclose(a, b, atol=1e-6)

    # And the input checkpoint is still a plain, unfused model.
    fresh = build_model_from_checkpoint(str(ckpt_path), fuse=False)
    assert module_op_counts(fresh).get("batchnorm", 0) > 0
