"""FlashDet Exporter — export models to ONNX (and other formats)."""

import os
import logging
from typing import Optional, Tuple

import torch

from flashdet.cfg import get_config
from flashdet.models import FlashDet

logger = logging.getLogger(__name__)


class Exporter:
    """Export a FlashDet model to ONNX format.

    Example::

        from flashdet import Exporter

        exporter = Exporter(model_path="workspace/model_best_inference.pth")
        exporter.export_onnx("model.onnx")
    """

    def __init__(
        self,
        model_path: str,
        input_size: Optional[Tuple[int, int]] = None,
    ):
        self.model_path = model_path
        self._input_size_override = input_size

        cfg = get_config()
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        backbone_size = cfg.model.backbone_size
        num_classes = cfg.model.num_classes
        fpn_channels = cfg.model.fpn_out_channels
        inp_size = cfg.model.input_size

        if "config" in checkpoint:
            ckpt_cfg = checkpoint["config"]
            backbone_size = ckpt_cfg.get("backbone_size", backbone_size)
            num_classes = ckpt_cfg.get("num_classes", num_classes)
            fpn_channels = ckpt_cfg.get("fpn_channels", fpn_channels)
            inp_size = ckpt_cfg.get("input_size", inp_size)

        if input_size is not None:
            inp_size = input_size

        self.input_size = inp_size
        self.num_classes = num_classes

        self.model = FlashDet(
            num_classes=num_classes,
            input_size=inp_size,
            backbone_size=backbone_size,
            fpn_channels=fpn_channels,
            pretrained=False,
            use_aux_head=False,
        )

        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        elif "state_dict" in checkpoint:
            sd = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
            self.model.load_state_dict(sd, strict=False)
        else:
            self.model.load_state_dict(checkpoint, strict=False)

        self.model.eval()
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model loaded: {total_params:,} parameters")

    def export(
        self,
        output: str = "model.onnx",
        simplify: bool = True,
        **kwargs,
    ) -> str:
        """Export model (convenience alias for export_onnx)."""
        return self.export_onnx(output_path=output, simplify=simplify, **kwargs)

    def export_onnx(
        self,
        output_path: str = "model.onnx",
        opset_version: int = 11,
        simplify: bool = True,
        dynamic_batch: bool = True,
    ) -> str:
        """Export model to ONNX format.

        Args:
            output_path: Path for the output .onnx file.
            opset_version: ONNX opset version.
            simplify: Whether to run onnxsim simplification.
            dynamic_batch: Whether to use dynamic batch axis.

        Returns:
            Path to the exported ONNX file.
        """
        inp_h, inp_w = self.input_size if isinstance(self.input_size, tuple) else (self.input_size, self.input_size)
        dummy_input = torch.randn(1, 3, inp_h, inp_w)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        dynamic_axes = None
        if dynamic_batch:
            dynamic_axes = {
                "data": {0: "batch"},
                "output": {0: "batch"},
            }

        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            opset_version=opset_version,
            input_names=["data"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            keep_initializers_as_inputs=True,
        )
        logger.info(f"ONNX exported: {output_path}")

        if simplify:
            try:
                import onnx
                from onnxsim import simplify as onnx_simplify

                onnx_model = onnx.load(output_path)
                simplified, _ = onnx_simplify(onnx_model)
                onnx.save(simplified, output_path)
                logger.info("ONNX model simplified successfully")
            except ImportError:
                logger.warning("onnxsim not installed, skipping simplification")

        file_size = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(f"Output: {output_path} ({file_size:.2f} MB)")
        return output_path

    def export_tensorrt(
        self,
        output_path: str = "model.engine",
        fp16: bool = True,
        int8: bool = False,
        max_batch_size: int = 1,
        workspace_mb: int = 4096,
    ) -> str:
        """Export model to TensorRT engine via ONNX.

        Tries ``torch_tensorrt`` first (direct PyTorch → TRT compilation).
        Falls back to ``tensorrt`` ONNX parser if ``torch_tensorrt`` is
        unavailable. Requires an intermediate ONNX export in the fallback
        path.

        Args:
            output_path: Path for the output TensorRT engine file.
            fp16: Enable FP16 precision.
            int8: Enable INT8 precision (requires calibration data).
            max_batch_size: Maximum batch size for the engine.
            workspace_mb: TensorRT workspace size in MB.

        Returns:
            Path to the exported TensorRT engine file.
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        inp_h, inp_w = self.input_size if isinstance(self.input_size, tuple) else (self.input_size, self.input_size)

        # Strategy 1: torch_tensorrt (preferred)
        try:
            import torch_tensorrt

            example_input = torch.randn(max_batch_size, 3, inp_h, inp_w).to("cuda")
            self.model.to("cuda")

            enabled_precisions = {torch.float32}
            if fp16:
                enabled_precisions.add(torch.float16)
            if int8:
                enabled_precisions.add(torch.int8)

            trt_model = torch_tensorrt.compile(
                self.model,
                inputs=[torch_tensorrt.Input(
                    min_shape=(1, 3, inp_h, inp_w),
                    opt_shape=(max_batch_size, 3, inp_h, inp_w),
                    max_shape=(max_batch_size, 3, inp_h, inp_w),
                    dtype=torch.float32,
                )],
                enabled_precisions=enabled_precisions,
                workspace_size=workspace_mb * (1024 ** 2),
                truncate_long_and_double=True,
            )

            torch_tensorrt.save(trt_model, output_path, output_format="torchscript")
            logger.info(f"TensorRT engine exported via torch_tensorrt: {output_path}")
            file_size = os.path.getsize(output_path) / (1024 * 1024)
            logger.info(f"Output: {output_path} ({file_size:.2f} MB)")
            return output_path

        except ImportError:
            logger.info("torch_tensorrt not available, trying native TensorRT ONNX path")

        # Strategy 2: TensorRT via ONNX
        try:
            import tensorrt as trt

            onnx_path = output_path.replace(".engine", ".onnx")
            self.export_onnx(onnx_path, simplify=True, dynamic_batch=False)

            trt_logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(trt_logger)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, trt_logger)

            with open(onnx_path, "rb") as f:
                if not parser.parse(f.read()):
                    for i in range(parser.num_errors):
                        logger.error(f"TensorRT ONNX parse error: {parser.get_error(i)}")
                    raise RuntimeError("Failed to parse ONNX model for TensorRT")

            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * (1024 ** 2))

            if fp16 and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                logger.info("TensorRT FP16 mode enabled")
            if int8 and builder.platform_has_fast_int8:
                config.set_flag(trt.BuilderFlag.INT8)
                logger.info("TensorRT INT8 mode enabled")

            profile = builder.create_optimization_profile()
            profile.set_shape(
                "data",
                (1, 3, inp_h, inp_w),
                (max_batch_size, 3, inp_h, inp_w),
                (max_batch_size, 3, inp_h, inp_w),
            )
            config.add_optimization_profile(profile)

            engine_bytes = builder.build_serialized_network(network, config)
            if engine_bytes is None:
                raise RuntimeError("TensorRT engine build failed")

            with open(output_path, "wb") as f:
                f.write(engine_bytes)

            file_size = os.path.getsize(output_path) / (1024 * 1024)
            logger.info(f"TensorRT engine exported via ONNX: {output_path} ({file_size:.2f} MB)")
            return output_path

        except ImportError:
            raise ImportError(
                "Neither torch_tensorrt nor tensorrt is installed. "
                "Install one of: pip install torch-tensorrt  OR  pip install tensorrt"
            )
