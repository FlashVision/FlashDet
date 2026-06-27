"""Analytics — benchmarking, profiling, metrics, error analysis and visualisation tools for FlashDet."""

from flashdet.analytics.benchmark import Benchmark
from flashdet.analytics.profiler import Profiler
from flashdet.analytics.flops import FLOPsCounter
from flashdet.analytics.metrics import DetectionMetrics
from flashdet.analytics.dataset_stats import DatasetAnalyzer
from flashdet.analytics.detection_analysis import DetectionErrorAnalyzer
from flashdet.analytics.report import AnalyticsReport
from flashdet.analytics.plots import (
    plot_training_curves,
    plot_pr_curve,
    plot_confusion_matrix,
    plot_map_curve,
)

__all__ = [
    # Core tools
    "Benchmark",
    "Profiler",
    "FLOPsCounter",
    # Evaluation
    "DetectionMetrics",
    "DetectionErrorAnalyzer",
    # Dataset analysis
    "DatasetAnalyzer",
    # Reporting
    "AnalyticsReport",
    # Plotting
    "plot_training_curves",
    "plot_pr_curve",
    "plot_confusion_matrix",
    "plot_map_curve",
]
