#!/usr/bin/env python3
"""
Quantize model from FP16 to INT8 for deployment (< 1MB model).

Supports:
- ONNX quantization
- TensorRT INT8 (if available)
- PyTorch dynamic quantization

Usage:
    python scripts/fp16_to_int8_quantize.py --model model.onnx --output model_int8.onnx
    python scripts/fp16_to_int8_quantize.py --model checkpoint.pth --format pytorch
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def quantize_onnx(
    model_path: str,
    output_path: str,
    calibration_data: str = None
):
    """
    Quantize ONNX model to INT8.
    
    Args:
        model_path: Input ONNX model path
        output_path: Output INT8 model path
        calibration_data: Path to calibration data (optional)
    """
    try:
        import onnx
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print("ERROR: onnxruntime not installed")
        print("Install: pip install onnxruntime")
        return None
    
    print("=" * 60)
    print("ONNX INT8 Quantization")
    print("=" * 60)
    
    print(f"\n1. Loading model: {model_path}")
    original_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"   Original size: {original_size:.2f} MB")
    
    print("\n2. Quantizing to INT8...")
    
    # Dynamic quantization (no calibration needed)
    quantize_dynamic(
        model_input=model_path,
        model_output=output_path,
        weight_type=QuantType.QUInt8
    )
    
    quantized_size = os.path.getsize(output_path) / (1024 * 1024)
    reduction = (1 - quantized_size / original_size) * 100
    
    print(f"\n" + "=" * 60)
    print("Quantization Complete!")
    print(f"  Output: {output_path}")
    print(f"  Original: {original_size:.2f} MB")
    print(f"  Quantized: {quantized_size:.2f} MB")
    print(f"  Reduction: {reduction:.1f}%")
    print("=" * 60)
    
    return output_path


def quantize_pytorch(
    model_path: str,
    output_path: str
):
    """
    Quantize PyTorch model using dynamic quantization.
    
    Args:
        model_path: Input checkpoint path
        output_path: Output quantized model path
    """
    import torch
    from flashdet.cfg import get_config
    from flashdet.models import FlashDet
    
    print("=" * 60)
    print("PyTorch Dynamic Quantization")
    print("=" * 60)
    
    config = get_config()
    
    # Load checkpoint to detect model config
    checkpoint = torch.load(model_path, map_location="cpu")
    
    # Auto-detect from checkpoint metadata
    backbone_size = config.model.backbone_size
    num_classes = config.model.num_classes
    fpn_channels = config.model.fpn_out_channels
    input_size = config.model.input_size
    
    if "config" in checkpoint:
        ckpt_config = checkpoint["config"]
        backbone_size = ckpt_config.get("backbone_size", backbone_size)
        num_classes = ckpt_config.get("num_classes", num_classes)
        fpn_channels = ckpt_config.get("fpn_channels", fpn_channels)
        input_size = ckpt_config.get("input_size", input_size)
        print(f"   Detected from checkpoint: backbone={backbone_size}, classes={num_classes}")
    
    print("\n1. Loading model...")
    model = FlashDet(
        num_classes=num_classes,
        input_size=input_size,
        backbone_size=backbone_size,
        fpn_channels=fpn_channels,
        pretrained=False
    )
    
    # Load only model weights (strict=False to ignore aux_head from training)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    elif "state_dict" in checkpoint:
        state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model.eval()
    
    # Original size
    original_path = output_path.replace("_int8", "_fp32")
    torch.save(model.state_dict(), original_path)
    original_size = os.path.getsize(original_path) / (1024 * 1024)
    os.remove(original_path)
    
    print(f"   Original size: {original_size:.2f} MB")
    
    print("\n2. Applying dynamic quantization...")
    
    # Dynamic quantization (works on Linear and Conv layers)
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear, torch.nn.Conv2d},
        dtype=torch.qint8
    )
    
    print("\n3. Saving quantized model...")
    torch.save(quantized_model.state_dict(), output_path)
    
    quantized_size = os.path.getsize(output_path) / (1024 * 1024)
    reduction = (1 - quantized_size / original_size) * 100
    
    print(f"\n" + "=" * 60)
    print("Quantization Complete!")
    print(f"  Output: {output_path}")
    print(f"  Original: {original_size:.2f} MB")
    print(f"  Quantized: {quantized_size:.2f} MB")
    print(f"  Reduction: {reduction:.1f}%")
    print("=" * 60)
    
    return output_path


def convert_to_fp16(model_path: str, output_path: str):
    """
    Convert ONNX model to FP16.
    
    Args:
        model_path: Input ONNX model path
        output_path: Output FP16 model path
    """
    try:
        import onnx
        from onnxconverter_common import float16
    except ImportError:
        print("ERROR: onnxconverter-common not installed")
        print("Install: pip install onnxconverter-common")
        return None
    
    print("=" * 60)
    print("ONNX FP16 Conversion")
    print("=" * 60)
    
    print(f"\n1. Loading model: {model_path}")
    model = onnx.load(model_path)
    original_size = os.path.getsize(model_path) / (1024 * 1024)
    
    print("\n2. Converting to FP16...")
    model_fp16 = float16.convert_float_to_float16(model)
    
    print(f"\n3. Saving: {output_path}")
    onnx.save(model_fp16, output_path)
    
    fp16_size = os.path.getsize(output_path) / (1024 * 1024)
    reduction = (1 - fp16_size / original_size) * 100
    
    print(f"\n" + "=" * 60)
    print("Conversion Complete!")
    print(f"  Original: {original_size:.2f} MB")
    print(f"  FP16: {fp16_size:.2f} MB")
    print(f"  Reduction: {reduction:.1f}%")
    print("=" * 60)
    
    return output_path


def create_tflite_int8(onnx_path: str, output_path: str, calibration_dir: str = None):
    """
    Convert ONNX to TFLite INT8.
    
    Args:
        onnx_path: Input ONNX model
        output_path: Output TFLite path
        calibration_dir: Directory with calibration images
    """
    try:
        import tensorflow as tf
        import onnx
        from onnx_tf.backend import prepare
    except ImportError:
        print("ERROR: Required packages not installed")
        print("Install: pip install tensorflow onnx-tf")
        return None
    
    print("=" * 60)
    print("ONNX to TFLite INT8 Conversion")
    print("=" * 60)
    
    # Step 1: ONNX to TF SavedModel
    print("\n1. Converting ONNX to TensorFlow...")
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)
    
    saved_model_dir = output_path.replace(".tflite", "_saved_model")
    tf_rep.export_graph(saved_model_dir)
    
    # Step 2: TF SavedModel to TFLite INT8
    print("\n2. Converting to TFLite INT8...")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.int8]
    
    # Representative dataset for quantization
    if calibration_dir:
        import numpy as np
        import cv2
        from glob import glob
        
        def representative_dataset():
            images = glob(os.path.join(calibration_dir, "*.jpg"))[:100]
            for img_path in images:
                img = cv2.imread(img_path)
                img = cv2.resize(img, (320, 320))
                img = img.astype(np.float32) / 255.0
                img = np.expand_dims(img, 0)
                yield [img]
        
        converter.representative_dataset = representative_dataset
    
    tflite_model = converter.convert()
    
    print(f"\n3. Saving: {output_path}")
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    
    size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n" + "=" * 60)
    print("Conversion Complete!")
    print(f"  Output: {output_path}")
    print(f"  Size: {size:.2f} MB")
    print("=" * 60)
    
    # Cleanup
    import shutil
    shutil.rmtree(saved_model_dir, ignore_errors=True)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Quantize model to INT8")
    parser.add_argument("--model", "-m", required=True, help="Input model path")
    parser.add_argument("--output", "-o", help="Output path")
    parser.add_argument("--format", choices=["onnx", "pytorch", "fp16", "tflite"],
                        default="onnx", help="Quantization format")
    parser.add_argument("--calibration", help="Calibration data directory")
    args = parser.parse_args()
    
    # Default output path
    if args.output is None:
        base = os.path.splitext(args.model)[0]
        if args.format == "onnx":
            args.output = f"{base}_int8.onnx"
        elif args.format == "pytorch":
            args.output = f"{base}_int8.pth"
        elif args.format == "fp16":
            args.output = f"{base}_fp16.onnx"
        else:
            args.output = f"{base}_int8.tflite"
    
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    
    if args.format == "onnx":
        quantize_onnx(args.model, args.output, args.calibration)
    elif args.format == "pytorch":
        quantize_pytorch(args.model, args.output)
    elif args.format == "fp16":
        convert_to_fp16(args.model, args.output)
    elif args.format == "tflite":
        create_tflite_int8(args.model, args.output, args.calibration)


if __name__ == "__main__":
    main()
