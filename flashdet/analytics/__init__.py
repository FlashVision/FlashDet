"""Analytics — benchmarking, profiling and visualisation tools for FlashDet."""

from flashdet.analytics.benchmark import Benchmark
from flashdet.analytics.profiler import Profiler
from flashdet.analytics.plots import (
    plot_training_curves,
    plot_pr_curve,
    plot_confusion_matrix,
    plot_map_curve,
)

__all__ = [
    "Benchmark",
    "Profiler",
    "plot_training_curves",
    "plot_pr_curve",
    "plot_confusion_matrix",
    "plot_map_curve",
]
