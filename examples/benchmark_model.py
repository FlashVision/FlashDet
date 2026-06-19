"""
Benchmark Model Performance
=============================

Measure FPS, latency, model size, and parameter count.
Useful for comparing model sizes before deployment.
"""

from flashdet.analytics import Benchmark

bench = Benchmark(
    model_path="workspace/my_model/best.pth",
    device="cuda",
    input_size=320,
)

results = bench.run()

print("=" * 40)
print("        FlashDet Benchmark Results")
print("=" * 40)
print(f"  FPS:        {results['fps']:.1f}")
print(f"  Latency:    {results['latency_ms']:.2f} ms")
print(f"  Parameters: {results['params']:,}")
print(f"  Model size: {results['size_mb']:.2f} MB")
print("=" * 40)
