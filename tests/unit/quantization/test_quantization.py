"""Tests for the quantization toolkit and FlashDet layer policies."""

import torch
import torch.nn as nn

from flashdet.models.architectures.flashdet import FlashDet
from flashdet.quantization.static_quantization.weight_utils import (
    dequantize_weight_per_group,
)
from flashdet.quantization import (
    Int8WeightOnlyConv2d,
    LSQFakeQuantize,
    MinMaxObserver,
    PACTActivation,
    QuantConfig,
    AWQQuantizer,
    GPTQQuantizer,
    PTQQuantizer,
    QATQuantizer,
    QuantFlashDet,
    QuantLayerPolicy,
    apply_smooth_scale,
    compute_smooth_scale,
    convert_qat,
    fake_quantize,
    prepare_qat,
    quantize_awq,
    quantize_dynamic,
    quantize_gptq,
    quantize_static,
    quantize_weight_only,
)


def _model():
    model = FlashDet(num_classes=3, size="p").eval()
    return model, torch.randn(1, 3, 64, 64)


def test_minmax_observer_quantizes_with_bounded_error():
    cfg = QuantConfig(bits=8, symmetric=False)
    observer = MinMaxObserver(cfg)
    x = torch.linspace(-2, 2, 1000)
    observer.observe(x)
    scale, zero_point = observer.calculate_qparams()
    dq = fake_quantize(x, scale, zero_point, cfg.qmin, cfg.qmax)
    assert scale.item() > 0
    assert torch.max(torch.abs(x - dq)).item() < 0.1


def test_per_channel_weight_only_conv():
    conv = nn.Conv2d(4, 6, 3, padding=1)
    qconv = Int8WeightOnlyConv2d(conv)
    x = torch.randn(2, 4, 16, 16)
    y = qconv(x)
    assert y.shape == (2, 6, 16, 16)
    assert qconv.weight_int8.dtype == torch.int8
    assert qconv.weight_int8.shape == conv.weight.shape


def test_dynamic_and_weight_only_flashdet_forward():
    model, x = _model()
    for quantize in (quantize_weight_only, quantize_dynamic):
        qmodel = quantize(model)
        with torch.no_grad():
            out = qmodel(x)
        assert out["o2o_cls"].shape[0] == 1


def test_static_calibrated_flashdet_forward():
    model, x = _model()
    calib = [torch.randn(1, 3, 64, 64) for _ in range(2)]
    qmodel = quantize_static(
        model,
        calib,
        layer_prefixes=("backbone", "neck"),
        keep_float_prefixes=("head",),
        num_batches=2,
    )
    with torch.no_grad():
        out = qmodel(x)
    assert out["o2o_cls"].shape[0] == 1
    assert torch.isfinite(out["o2o_cls"]).all()


def test_awq_quantizes_flashdet():
    model, x = _model()
    calib = [torch.randn(1, 3, 64, 64) for _ in range(2)]
    qmodel = quantize_awq(
        model, calib, bits=4, group_size=32,
        layer_prefixes=("backbone", "neck"), num_batches=2,
    )
    with torch.no_grad():
        out = qmodel(x)
    assert out["o2o_cls"].shape[0] == 1
    assert torch.isfinite(out["o2o_cls"]).all()


def test_gptq_quantizes_linear_and_returns_finite_output():
    model = nn.Sequential(
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 4),
    ).eval()
    calib = [torch.randn(8, 16) for _ in range(4)]
    qmodel = quantize_gptq(model, calib, bits=8, group_size=8, num_batches=4)
    x = torch.randn(3, 16)
    with torch.no_grad():
        y = qmodel(x)
    assert y.shape == (3, 4)
    assert torch.isfinite(y).all()


def test_awq_8bit_reconstruction_error_is_small():
    from flashdet.quantization import awq_quantize_weight

    weight = torch.randn(32, 64)
    activation = torch.rand(64) * 2
    q, scales, zeros, input_scale, gs = awq_quantize_weight(
        weight, activation, bits=8, group_size=32,
    )
    dq = dequantize_weight_per_group(q, scales, zeros, tuple(weight.shape), gs)
    dq = dq / input_scale.view(1, -1)
    rel = (dq - weight).norm() / weight.norm()
    assert rel.item() < 0.1


def test_gptq_8bit_reconstruction_error_is_small():
    from flashdet.quantization import gptq_quantize_weight

    weight = torch.randn(16, 32)
    q, scales, zeros, gs = gptq_quantize_weight(
        weight, torch.eye(32), bits=8, group_size=32,
    )
    dq = dequantize_weight_per_group(q, scales, zeros, tuple(weight.shape), gs)
    rel = (dq - weight).norm() / weight.norm()
    assert rel.item() < 0.1


def test_gptq_quantizes_flashdet_conv_paths():
    model, x = _model()
    calib = [torch.randn(1, 3, 64, 64) for _ in range(2)]
    qmodel = quantize_gptq(
        model, calib, bits=8, group_size=-1,
        layer_prefixes=("backbone", "neck"), num_batches=2,
    )
    with torch.no_grad():
        out = qmodel(x)
    assert out["o2o_cls"].shape[0] == 1
    assert torch.isfinite(out["o2o_cls"]).all()


def test_quantflashdet_layer_policy_report():
    model, x = _model()
    calib = [torch.randn(1, 3, 64, 64) for _ in range(2)]
    quantizer = QuantFlashDet(
        model,
        policy=QuantLayerPolicy(backbone="static", neck="static", head="none"),
    )
    qmodel = quantizer.quantize(calib_loader=calib, num_batches=2)
    report = quantizer.report(qmodel)
    assert report["policy"]["head"] == "none"
    assert report["quantized_modules"]["backbone"] > 0
    with torch.no_grad():
        assert qmodel(x)["o2o_cls"].shape[0] == 1


def test_qat_prepare_and_fake_quant_convert():
    model, x = _model()
    prepared = prepare_qat(
        model,
        layer_prefixes=("backbone", "neck"),
        keep_float_prefixes=("head",),
    )
    prepared.eval()
    with torch.no_grad():
        assert prepared(x)["o2o_cls"].shape[0] == 1
    converted = convert_qat(prepared)
    with torch.no_grad():
        assert converted(x)["o2o_cls"].shape[0] == 1


def test_lsq_and_pact_are_differentiable():
    x = torch.randn(2, 8, 8, 8, requires_grad=True)
    lsq = LSQFakeQuantize(QuantConfig(bits=8))
    loss = lsq(x).sum()
    loss.backward()
    assert x.grad is not None

    pact = PACTActivation(init_alpha=2.0)
    out = pact(torch.randn(2, 4) * 5)
    assert out.max().item() <= 2.01
    out.sum().backward()
    assert pact.alpha.grad is not None


def test_class_based_quantizer_interfaces():
    linear = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 4)).eval()
    calib_linear = [torch.randn(8, 16) for _ in range(4)]

    awq_model = AWQQuantizer(linear, bits=4, group_size=8).quantize(
        calib_linear, num_batches=4
    )
    with torch.no_grad():
        assert torch.isfinite(awq_model(torch.randn(2, 16))).all()

    gptq_model = GPTQQuantizer(linear, bits=8, group_size=8).quantize(
        calib_linear, num_batches=4
    )
    with torch.no_grad():
        assert torch.isfinite(gptq_model(torch.randn(2, 16))).all()

    ptq = PTQQuantizer(linear)
    with torch.no_grad():
        assert torch.isfinite(ptq.quantize(calib_linear, num_batches=4)(torch.randn(2, 16))).all()

    qat = QATQuantizer(linear)
    qat.prepare(layer_prefixes=("",), keep_float_prefixes=())
    qat.model.eval()
    with torch.no_grad():
        assert torch.isfinite(qat.model(torch.randn(2, 16))).all()
    converted = qat.convert()
    with torch.no_grad():
        assert torch.isfinite(converted(torch.randn(2, 16))).all()


def test_smoothquant_scaling_preserves_conv_output():
    conv = nn.Conv2d(4, 6, 3, padding=1)
    x = torch.randn(1, 4, 16, 16)
    before = conv(x).detach()
    scale = compute_smooth_scale(conv.weight, x.abs().amax(dim=(0, 2, 3)))
    apply_smooth_scale(conv, scale)
    after = conv(x / scale.view(1, -1, 1, 1)).detach()
    assert torch.allclose(before, after, atol=1e-4)
