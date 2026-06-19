"""
Export Model to ONNX
=====================

Export a trained FlashDet model to ONNX format for deployment
on edge devices, web, or any ONNX-compatible runtime.
"""

from flashdet import Exporter

exporter = Exporter(model_path="workspace/my_model/best.pth")

output_path = exporter.export(
    output="model.onnx",
    simplify=True,
)

print(f"Model exported to: {output_path}")
print("You can now run this model with ONNX Runtime, TensorRT, OpenVINO, etc.")
