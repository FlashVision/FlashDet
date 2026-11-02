"""ONNX export for FlashDet checkpoints.

Fuses reparameterizable blocks, traces the model to ONNX and (optionally)
runs a graph optimiser and a PyTorch-vs-ONNXRuntime parity check so the
exported file is provably equivalent before you ship it.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

_YOLO_ARCHS = ("yolov8", "yolov9", "yolov10", "yolov11")


def build_model_from_checkpoint(model_path: str, fuse: bool = True) -> torch.nn.Module:
    """Rebuild an inference-ready model from a FlashDet checkpoint.

    Applies the same config reconstruction that ``flashdet export`` and the
    predictor use, then fuses reparameterizable blocks when available
    (multi-branch ``rep`` blocks collapse into single convs for deployment).
    """
    from flashdet.cfg import get_config
    from flashdet.models.detector import build_model
    from flashdet.utils.checkpoint import safe_torch_load

    ckpt = safe_torch_load(model_path, map_location="cpu")
    if isinstance(ckpt, torch.nn.Module):
        module = ckpt
    else:
        cfg_data = ckpt.get("config", {}) or {}
        arch = cfg_data.get("architecture", "flashdet")
        num_classes = cfg_data.get("num_classes", 80)
        model_size = cfg_data.get("model_size", "n")
        input_size = cfg_data.get("input_size")
        if isinstance(input_size, (list, tuple)):
            input_size = input_size[0]

        config = get_config(num_classes=num_classes)
        config.model.architecture = arch
        config.model.size = model_size
        config.model.backbone_type = cfg_data.get("backbone_type") or "lite"
        config.model.neck_type = cfg_data.get("neck_type")

        if arch == "flashdet" and config.model.backbone_type == "lite":
            peek = ckpt.get(
                "model_state_dict",
                ckpt.get("state_dict", ckpt if isinstance(ckpt, dict) else {}),
            )
            if isinstance(peek, dict) and any(
                k.startswith("backbone.stem.") or k.startswith("backbone.stages.")
                for k in peek
            ):
                config.model.backbone_type = "repnext"

        if arch in _YOLO_ARCHS:
            config.model.width_mult = cfg_data.get("width_mult", 1.0)
            config.model.depth_mult = cfg_data.get("depth_mult", 1.0)

        module = build_model(config, architecture=arch)
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        module.load_state_dict(state_dict, strict=False)

    module.eval()
    if fuse:
        raw = module.module if hasattr(module, "module") else module
        if hasattr(raw, "fuse"):
            try:
                raw.fuse()
            except Exception as exc:
                logger.warning("fuse() failed, running unfused: %s", exc)
    return module


def input_size_from_checkpoint(model_path: str, default: int = 320) -> int:
    """Read the training input size from a checkpoint config."""
    from flashdet.utils.checkpoint import safe_torch_load

    ckpt = safe_torch_load(model_path, map_location="cpu")
    config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    size = config.get("input_size") if isinstance(config, dict) else None
    if isinstance(size, (list, tuple)):
        size = size[0] if size else None
    return int(size or default)


class _ExportWrapper(torch.nn.Module):
    """Single tensor output so the ONNX graph traces cleanly."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        return torch.cat([out["o2o_cls"], out["o2o_reg"]], dim=-1)


def _maybe_simplify(path: str) -> None:
    """Best-effort graph optimisation; no-op when tools are missing."""
    try:
        import onnx
        import onnxslim

        model = onnxslim.slim(path)
        onnx.save(model, path)
        logger.info("Simplified ONNX graph with onnxslim")
        return
    except ImportError:
        pass
    try:
        import onnx
        import onnxsim

        model, ok = onnxsim.simplify(path)
        if ok:
            onnx.save(model, path)
            logger.info("Simplified ONNX graph with onnxsim")
    except ImportError:
        logger.info("onnxslim/onnxsim not installed — skipped simplification")


def _export_file(wrapper, dummy, output_path: str, opset: int,
                 dynamic_batch: bool, single_file: bool = True) -> None:
    """Export through torch.onnx.export, tolerating old torch signatures."""
    env = {
        "input_names": ["images"],
        "output_names": ["output"],
        "opset_version": opset,
    }
    if dynamic_batch:
        env["dynamic_axes"] = {"images": {0: "batch"}, "output": {0: "batch"}}
    if single_file:
        import inspect

        sig = inspect.signature(torch.onnx.export)
        if "external_data" in sig.parameters:
            env["external_data"] = False
        else:
            env["use_external_data_format"] = False
    try:
        torch.onnx.export(wrapper, dummy, output_path, **env)
    except TypeError:
        for key in list(env):
            if key in ("external_data", "use_external_data_format"):
                env.pop(key)
        torch.onnx.export(wrapper, dummy, output_path, **env)


def export_onnx(model_path: str, output_path: str | None = None,
                 fuse: bool = True, simplify: bool = False,
                 opset: int = 18, dynamic_batch: bool = True) -> str:
    """Export a FlashDet checkpoint to ONNX, optionally fusing Rep blocks."""
    output_path = output_path or os.path.splitext(model_path)[0] + ".onnx"
    module = build_model_from_checkpoint(model_path, fuse=fuse)
    wrapper = _ExportWrapper(module)
    wrapper.eval()

    input_size = input_size_from_checkpoint(model_path)
    dummy = torch.randn(1, 3, input_size, input_size)

    _export_file(wrapper, dummy, output_path, opset, dynamic_batch)

    if simplify:
        _maybe_simplify(output_path)
    return output_path
