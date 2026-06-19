#!/usr/bin/env python3
"""
Convert PyTorch model to ONNX format.

Usage:
    python scripts/convert_pth_to_onnx.py --model checkpoint.pth --output model.onnx
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from flashdet.cfg import get_config
from flashdet.models import FlashDet
from flashdet.utils import load_checkpoint


def convert_to_onnx(
    model_path: str,
    output_path: str,
    input_size: tuple = (320, 320),
    opset_version: int = 11,
    simplify: bool = True
):
    """
    Convert PyTorch model to ONNX.
    
    Args:
        model_path: Path to PyTorch checkpoint
        output_path: Output ONNX path
        input_size: Input size (width, height)
        opset_version: ONNX opset version
        simplify: Whether to simplify ONNX model
    """
    print("=" * 60)
    print("PyTorch to ONNX Conversion")
    print("=" * 60)
    
    config = get_config()
    
    # Load checkpoint to detect model config
    checkpoint = torch.load(model_path, map_location="cpu")
    
    # Auto-detect from checkpoint metadata
    backbone_size = config.model.backbone_size
    num_classes = config.model.num_classes
    fpn_channels = config.model.fpn_out_channels
    
    if "config" in checkpoint:
        ckpt_config = checkpoint["config"]
        backbone_size = ckpt_config.get("backbone_size", backbone_size)
        num_classes = ckpt_config.get("num_classes", num_classes)
        fpn_channels = ckpt_config.get("fpn_channels", fpn_channels)
        print(f"   Detected from checkpoint: backbone={backbone_size}, classes={num_classes}")
    
    # Build model
    print("\n1. Building model...")
    model = FlashDet(
        num_classes=num_classes,
        input_size=input_size,
        backbone_size=backbone_size,
        fpn_channels=fpn_channels,
        pretrained=False
    )
    
    # Load weights
    print(f"2. Loading weights: {model_path}")
    # strict=False to ignore aux_head keys from training checkpoints
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    elif "state_dict" in checkpoint:
        state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params:,}")
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, input_size[1], input_size[0])
    print(f"   Input shape: {list(dummy_input.shape)}")
    
    # Export
    print(f"\n3. Exporting to ONNX...")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    # Match common FlashDet ONNX convention (input/output tensor names)
    # Input: "data", Output: "output" (single fused tensor)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=["data"],  # Same as typical FlashDet ONNX exports
        output_names=["output"],  # Single output (cls + reg fused)
        dynamic_axes={
            "data": {0: "batch"},
            "output": {0: "batch"}
        },
        keep_initializers_as_inputs=True  # Match official
    )
    print(f"   Saved: {output_path}")
    
    # Simplify
    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify
            
            print("\n4. Simplifying ONNX model...")
            onnx_model = onnx.load(output_path)
            simplified, _ = onnx_simplify(onnx_model)
            onnx.save(simplified, output_path)
            print("   Simplification complete!")
        except ImportError:
            print("   onnxsim not installed. Skipping simplification.")
            print("   Install: pip install onnxsim")
    
    # Print info
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n" + "=" * 60)
    print(f"Conversion Complete!")
    print(f"  Output: {output_path}")
    print(f"  Size: {file_size:.2f} MB")
    print("=" * 60)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch to ONNX")
    parser.add_argument("--model", "-m", required=True, help="PyTorch checkpoint")
    parser.add_argument("--output", "-o", default="models/ppe_detector.onnx", help="Output path")
    parser.add_argument("--input-size", type=int, nargs=2, default=[320, 320], help="Input size [W, H]")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version")
    parser.add_argument("--no-simplify", action="store_true", help="Skip simplification")
    args = parser.parse_args()
    
    convert_to_onnx(
        model_path=args.model,
        output_path=args.output,
        input_size=tuple(args.input_size),
        opset_version=args.opset,
        simplify=not args.no_simplify
    )


if __name__ == "__main__":
    main()
