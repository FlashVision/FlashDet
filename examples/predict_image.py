"""
Run Inference on an Image
==========================

Load a trained model and detect objects in an image.
Results are printed and optionally saved with bounding boxes drawn.
"""

from flashdet import Predictor

predictor = Predictor(
    model_path="workspace/my_model/best.pth",
    device="cuda",
    conf_thresh=0.25,
)

results = predictor.predict("test_image.jpg", output_dir="output/")

print(f"Found {len(results)} objects:")
for class_name, score, x1, y1, x2, y2 in results:
    print(f"  {class_name}: {score:.2f} at [{x1}, {y1}, {x2}, {y2}]")
