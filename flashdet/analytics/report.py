"""Report generation — export analytics results to JSON, text, and HTML summaries."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


class AnalyticsReport:
    """Generate and export comprehensive analytics reports.

    Aggregates results from Benchmark, Profiler, DetectionMetrics,
    DetectionErrorAnalyzer, DatasetAnalyzer and FLOPsCounter into a
    unified report.

    Parameters
    ----------
    title : str
        Report title.
    model_name : str | None
        Name or path of the model being analyzed.
    """

    def __init__(self, title: str = "FlashDet Analytics Report", model_name: Optional[str] = None):
        self.title = title
        self.model_name = model_name
        self._sections: Dict[str, Any] = {}
        self._metadata: Dict[str, Any] = {
            "generated_at": datetime.now().isoformat(),
            "model_name": model_name,
        }

    def add_section(self, name: str, data: Any):
        """Add a named section to the report.

        Parameters
        ----------
        name : str
            Section identifier (e.g. "benchmark", "metrics", "dataset_stats").
        data : dict | str | Any
            Section content — typically a dict from another analytics tool.
        """
        self._sections[name] = data

    def add_benchmark(self, benchmark_results: Dict[str, Any]):
        """Add benchmark results (from Benchmark.run())."""
        self._sections["benchmark"] = benchmark_results

    def add_profiler(self, profiler_results: List[Dict[str, Any]]):
        """Add profiler results (from Profiler.run())."""
        self._sections["profiler"] = profiler_results

    def add_metrics(self, metrics_results: Dict[str, Any]):
        """Add detection metrics (from DetectionMetrics.compute())."""
        self._sections["metrics"] = metrics_results

    def add_error_analysis(self, error_results: Dict[str, Any]):
        """Add error analysis (from DetectionErrorAnalyzer.analyze())."""
        self._sections["error_analysis"] = {
            k: v for k, v in error_results.items() if k != "errors"
        }

    def add_dataset_stats(self, dataset_results: Dict[str, Any]):
        """Add dataset statistics (from DatasetAnalyzer.analyze())."""
        self._sections["dataset_stats"] = dataset_results

    def add_flops(self, flops_results: Dict[str, Any]):
        """Add FLOPs count (from FLOPsCounter.count())."""
        self._sections["flops"] = {
            "total_flops": flops_results["total_flops"],
            "total_macs": flops_results["total_macs"],
            "total_params": flops_results["total_params"],
            "flops_readable": flops_results["flops_readable"],
            "macs_readable": flops_results["macs_readable"],
            "params_readable": flops_results["params_readable"],
        }

    def add_custom(self, key: str, value: Any):
        """Add custom key-value data."""
        self._sections.setdefault("custom", {})[key] = value

    # ------------------------------------------------------------------
    # Export methods
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return the full report as a dictionary."""
        return {
            "title": self.title,
            "metadata": self._metadata,
            "sections": self._sections,
        }

    def to_json(self, path: Optional[Union[str, Path]] = None, indent: int = 2) -> str:
        """Export the report as a JSON string (and optionally save to file).

        Parameters
        ----------
        path : str | Path | None
            If provided, write JSON to this file.
        indent : int
            JSON indentation level.

        Returns
        -------
        str
            JSON representation of the report.
        """
        data = self.to_dict()
        text = json.dumps(data, indent=indent, default=self._json_serializer)
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(text)
        return text

    def to_text(self, path: Optional[Union[str, Path]] = None) -> str:
        """Export a human-readable text report.

        Parameters
        ----------
        path : str | Path | None
            If provided, write text to this file.

        Returns
        -------
        str
            Formatted text report.
        """
        lines = [
            "=" * 70,
            f"  {self.title}",
            "=" * 70,
            f"  Generated: {self._metadata['generated_at']}",
        ]
        if self.model_name:
            lines.append(f"  Model:     {self.model_name}")
        lines.append("")

        if "benchmark" in self._sections:
            bm = self._sections["benchmark"]
            lines.extend([
                "  --- Speed & Size ---",
                f"  FPS:          {bm.get('fps', 'N/A')}",
                f"  Latency:      {bm.get('latency_ms', 'N/A')} ms",
                f"  Parameters:   {bm.get('params', 'N/A'):,}",
                f"  Model size:   {bm.get('model_size_mb', 'N/A')} MB",
                "",
            ])

        if "flops" in self._sections:
            fl = self._sections["flops"]
            lines.extend([
                "  --- Computational Complexity ---",
                f"  FLOPs:  {fl['flops_readable']}",
                f"  MACs:   {fl['macs_readable']}",
                f"  Params: {fl['params_readable']}",
                "",
            ])

        if "metrics" in self._sections:
            m = self._sections["metrics"]
            lines.extend([
                "  --- Detection Metrics ---",
                f"  mAP@[.50:.95]:  {m.get('mAP', 0):.4f}",
                f"  mAP@.50:        {m.get('mAP_50', 0):.4f}",
                f"  mAP@.75:        {m.get('mAP_75', 0):.4f}",
                f"  mAP (small):    {m.get('mAP_small', 0):.4f}",
                f"  mAP (medium):   {m.get('mAP_medium', 0):.4f}",
                f"  mAP (large):    {m.get('mAP_large', 0):.4f}",
                "",
            ])

        if "error_analysis" in self._sections:
            ea = self._sections["error_analysis"]
            s = ea.get("summary", {})
            lines.extend([
                "  --- Error Analysis ---",
                f"  Classification errors:  {s.get('classification', 0)}",
                f"  Localization errors:    {s.get('localization', 0)}",
                f"  Cls + Loc errors:       {s.get('cls_and_loc', 0)}",
                f"  Duplicate detections:   {s.get('duplicate', 0)}",
                f"  Background FPs:         {s.get('background', 0)}",
                f"  Missed GTs:             {s.get('missed', 0)}",
                "",
            ])

        if "dataset_stats" in self._sections:
            ds = self._sections["dataset_stats"]
            lines.extend([
                "  --- Dataset Statistics ---",
                f"  Images:       {ds.get('num_images', 0):,}",
                f"  Annotations:  {ds.get('num_annotations', 0):,}",
                f"  Classes:      {ds.get('num_classes', 0)}",
            ])
            opi = ds.get("objects_per_image", {})
            if opi:
                lines.append(f"  Obj/Image:    mean={opi.get('mean', 0):.1f}, "
                             f"max={opi.get('max', 0)}")
            lines.append("")

        if "profiler" in self._sections:
            prof = self._sections["profiler"]
            if isinstance(prof, list) and prof:
                lines.append("  --- Top-10 Slowest Layers ---")
                sorted_layers = sorted(prof, key=lambda x: -x.get("time_ms", 0))[:10]
                for layer in sorted_layers:
                    lines.append(
                        f"    {layer['name']:<40} "
                        f"{layer.get('time_ms', 0):.3f}ms "
                        f"({layer.get('time_pct', 0):.1f}%)"
                    )
                lines.append("")

        lines.append("=" * 70)
        text = "\n".join(lines)

        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(text)
        return text

    def to_html(self, path: Optional[Union[str, Path]] = None) -> str:
        """Export an HTML report.

        Parameters
        ----------
        path : str | Path | None
            If provided, write HTML to this file.

        Returns
        -------
        str
            HTML string.
        """
        sections_html = []

        if "benchmark" in self._sections:
            bm = self._sections["benchmark"]
            sections_html.append(self._html_table(
                "Speed & Size",
                [("FPS", bm.get("fps")), ("Latency (ms)", bm.get("latency_ms")),
                 ("Parameters", f"{bm.get('params', 0):,}"),
                 ("Model Size (MB)", bm.get("model_size_mb"))],
            ))

        if "flops" in self._sections:
            fl = self._sections["flops"]
            sections_html.append(self._html_table(
                "Computational Complexity",
                [("FLOPs", fl["flops_readable"]), ("MACs", fl["macs_readable"]),
                 ("Parameters", fl["params_readable"])],
            ))

        if "metrics" in self._sections:
            m = self._sections["metrics"]
            sections_html.append(self._html_table(
                "Detection Metrics",
                [("mAP@[.50:.95]", f"{m.get('mAP', 0):.4f}"),
                 ("mAP@.50", f"{m.get('mAP_50', 0):.4f}"),
                 ("mAP@.75", f"{m.get('mAP_75', 0):.4f}"),
                 ("mAP (small)", f"{m.get('mAP_small', 0):.4f}"),
                 ("mAP (medium)", f"{m.get('mAP_medium', 0):.4f}"),
                 ("mAP (large)", f"{m.get('mAP_large', 0):.4f}")],
            ))

        if "error_analysis" in self._sections:
            ea = self._sections["error_analysis"]
            s = ea.get("summary", {})
            sections_html.append(self._html_table(
                "Error Analysis",
                [("Classification", s.get("classification", 0)),
                 ("Localization", s.get("localization", 0)),
                 ("Cls + Loc", s.get("cls_and_loc", 0)),
                 ("Duplicate", s.get("duplicate", 0)),
                 ("Background", s.get("background", 0)),
                 ("Missed", s.get("missed", 0))],
            ))

        if "dataset_stats" in self._sections:
            ds = self._sections["dataset_stats"]
            sections_html.append(self._html_table(
                "Dataset Statistics",
                [("Images", f"{ds.get('num_images', 0):,}"),
                 ("Annotations", f"{ds.get('num_annotations', 0):,}"),
                 ("Classes", ds.get("num_classes", 0))],
            ))

        body = "\n".join(sections_html)
        html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{self.title}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       max-width: 900px; margin: 40px auto; padding: 0 20px; background: #f8f9fa; }}
h1 {{ color: #1a1a2e; border-bottom: 2px solid #16213e; padding-bottom: 10px; }}
h2 {{ color: #16213e; margin-top: 30px; }}
table {{ border-collapse: collapse; width: 100%; margin: 15px 0; background: white;
         box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-radius: 6px; overflow: hidden; }}
th, td {{ padding: 10px 16px; text-align: left; border-bottom: 1px solid #eee; }}
th {{ background: #16213e; color: white; font-weight: 500; }}
tr:hover {{ background: #f0f4ff; }}
.meta {{ color: #666; font-size: 0.9em; margin-bottom: 20px; }}
</style>
</head>
<body>
<h1>{self.title}</h1>
<div class="meta">
  Generated: {self._metadata['generated_at']}<br>
  {"Model: " + self.model_name if self.model_name else ""}
</div>
{body}
</body>
</html>"""
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(html)
        return html

    # ------------------------------------------------------------------
    # Convenience: comparison report
    # ------------------------------------------------------------------

    @staticmethod
    def compare_models(results: List[Dict[str, Any]], model_names: Optional[List[str]] = None) -> str:
        """Generate a side-by-side text comparison of multiple model results.

        Parameters
        ----------
        results : list[dict]
            Each dict should contain keys like "fps", "latency_ms", "mAP", etc.
        model_names : list[str] | None
            Names for each model column.

        Returns
        -------
        str
            Formatted comparison table.
        """
        if not results:
            return "No results to compare."

        names = model_names or [f"Model_{i}" for i in range(len(results))]
        metrics_keys = ["fps", "latency_ms", "mAP", "mAP_50", "params", "model_size_mb",
                        "total_flops", "flops_readable"]

        col_w = max(len(n) for n in names) + 2
        header = f"{'Metric':<20}" + "".join(f"{n:>{col_w}}" for n in names)
        lines = [header, "-" * len(header)]

        for key in metrics_keys:
            values = []
            has_any = False
            for r in results:
                v = r.get(key)
                if v is not None:
                    has_any = True
                    if isinstance(v, float):
                        values.append(f"{v:.4f}")
                    elif isinstance(v, int):
                        values.append(f"{v:,}")
                    else:
                        values.append(str(v))
                else:
                    values.append("-")
            if has_any:
                row = f"{key:<20}" + "".join(f"{v:>{col_w}}" for v in values)
                lines.append(row)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _html_table(title: str, rows: List[tuple]) -> str:
        trs = "".join(f"<tr><td>{k}</td><td><strong>{v}</strong></td></tr>" for k, v in rows)
        return f"<h2>{title}</h2>\n<table><tbody>{trs}</tbody></table>"

    @staticmethod
    def _json_serializer(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return str(obj)
