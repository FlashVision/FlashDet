#!/usr/bin/env python
"""Render a benchmark_pipeline report as a deployment dashboard.

    python scripts/benchmark_pipeline.py --model best.pth --val-images data/valid \
        --input-size 128 --output workspace/pipeline/report.json
    python scripts/plot_pipeline.py --report workspace/pipeline/report.json \
        --output docs/images/optimization-pipeline.png

The figure has six panels: forward latency, artifact size, mAP@0.5, decoded
box IoU against FP32, a size/speed trade-off plot, and a scorecard comparing
the best stage with the FP32 baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


FAMILY_COLORS = {
    "FP32": "#5B7DB1",
    "FP16": "#8E7CC3",
    "ONNX": "#55A868",
    "INT8": "#DD8452",
}


def _short(label: str) -> str:
    return (label.replace(" PyTorch", "")
                 .replace(" (fused)", " fused")
                 .replace("FP32 ONNX", "ONNX")
                 .replace("INT8 ONNX", "INT8"))


def _color(short: str) -> str:
    base = FAMILY_COLORS.get(short.split(" ")[0], "#8C8C8C")
    if short.endswith("fused"):
        return tuple(c * 0.78 for c in mcolors.to_rgb(base))
    return base


def _style(ax, title: str) -> None:
    ax.set_title(title, loc="left")
    ax.grid(axis="x", alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _bars(ax, rows, value, title, unit, xlabel, fmt="{:.2f}",
          extra=None, xmax_pad=1.35):
    labels = [_short(row["label"]) for row in rows]
    values = [row[value] for row in rows]
    bars = ax.barh(labels, values,
                   color=[_color(lbl) for lbl in labels], height=0.62)
    ax.invert_yaxis()
    upper = max(values) if values else 1.0
    ax.set_xlim(0, upper * xmax_pad)
    for bar, row in zip(bars, rows):
        text = fmt.format(row[value])
        if extra is not None:
            text += "  ·  " + extra.format(**row)
        ax.text(bar.get_width() + upper * 0.02,
                bar.get_y() + bar.get_height() / 2, text,
                va="center", fontsize=8.5)
    _style(ax, title)
    ax.set_xlabel(xlabel)


def _lollipops(ax, rows, value, title, xlabel, xlim, fmt="{:.3f}",
               color_by_row=False):
    labels = [_short(row["label"]) for row in rows]
    ys = list(range(len(rows)))
    values = [row[value] for row in rows]
    ax.hlines(ys, xlim[0], values,
              color=[_color(lbl) for lbl in labels], linewidth=2, alpha=0.65)
    ax.scatter(values, ys,
               color=[_color(lbl) for lbl in labels], s=55, zorder=3)
    for y, row in zip(ys, rows):
        ax.text(row[value] + (xlim[1] - xlim[0]) * 0.012, y,
                fmt.format(row[value]), va="center", fontsize=8.5)
    ax.set_yticks(ys, labels)
    ax.invert_yaxis()
    ax.set_xlim(*xlim)
    _style(ax, title)
    ax.set_xlabel(xlabel)


_TRADEOFF_OFFSETS = {
    "FP32": (-38, 8),
    "FP32 fused": (8, 6),
    "ONNX": (-40, 4),
    "ONNX fused": (8, -13),
    "INT8": (8, 6),
    "INT8 fused": (10, 10),
}


def _tradeoff(ax, rows):
    for row in rows:
        short = _short(row["label"])
        ax.scatter(row["size_mb"], row["latency_ms"], s=150,
                   color=_color(short), edgecolor="white", linewidth=1.2,
                   zorder=3)
        ax.annotate(short, (row["size_mb"], row["latency_ms"]),
                    textcoords="offset points",
                    xytext=_TRADEOFF_OFFSETS.get(short, (6, 6)),
                    fontsize=8.5)
    # Arrow from the baseline to the fastest artifact.
    base = rows[0]
    fastest = min(rows, key=lambda row: row["latency_ms"])
    ax.annotate(
        "", xy=(fastest["size_mb"], fastest["latency_ms"]),
        xytext=(base["size_mb"], base["latency_ms"]),
        arrowprops=dict(arrowstyle="->", color="#999999", linestyle="--",
                        linewidth=1.3), zorder=1,
    )
    speed = base["latency_ms"] / fastest["latency_ms"]
    shrink = base["size_mb"] / fastest["size_mb"]
    ax.text(0.78, 1.55, f"{speed:.1f}× faster\n{shrink:.1f}× smaller",
            fontsize=8.5, color="#666666", ha="center")
    ax.set_yscale("log")
    ax.set_yticks([0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0])
    ax.set_yticklabels(["0.6", "0.8", "1", "1.5", "2", "3", "5"])
    ax.minorticks_off()
    ax.set_xlim(0.40, 1.47)
    ax.set_ylim(0.52, 6.2)
    ax.set_xlabel("artifact size (MB, lower is better)")
    ax.set_ylabel("forward latency (ms, log scale, lower is better)")
    _style(ax, "Trade-off (lower-left is better)")
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)


def _scorecard(ax, base, best):
    ax.axis("off")
    speed = base["latency_ms"] / best["latency_ms"]
    shrink = base["size_mb"] / best["size_mb"]
    params = (best["params"] - base["params"]) / base["params"] * 100
    map_drop = (best["map50"] - base["map50"]) * 100

    cards = [
        (f"{speed:.1f}×", "faster", f"{best['latency_ms']:.2f} ms vs "
                                   f"{base['latency_ms']:.2f} ms", "#2E7D4F"),
        (f"{shrink:.1f}×", "smaller", f"{best['size_mb']:.2f} MB vs "
                                     f"{base['size_mb']:.2f} MB", "#2E7D4F"),
        (f"{params:+.1f}%", "parameters", f"{best['params']:,} vs "
                                         f"{base['params']:,}", "#2E7D4F"),
        (f"{map_drop:+.1f}", "mAP points", f"{best['map50']:.3f} vs "
                                          f"{base['map50']:.3f}", "#B03A2E"),
    ]
    ax.set_title("Best artifact vs FP32 baseline", loc="left")
    positions = [(0.03, 0.72), (0.53, 0.72), (0.03, 0.30), (0.53, 0.30)]
    for (x, y), (big, small, detail, color) in zip(positions, cards):
        ax.text(x, y, big, transform=ax.transAxes, fontsize=26,
                fontweight="bold", color=color, va="center")
        ax.text(x, y - 0.10, small, transform=ax.transAxes, fontsize=10,
                color="#333333", va="center")
        ax.text(x, y - 0.18, detail, transform=ax.transAxes, fontsize=8.5,
                color="#666666", va="center")
    ax.text(0.03, 0.02,
            f"best artifact: {best['label']}", transform=ax.transAxes,
            fontsize=9.5, style="italic", color="#333333")


def render(report: dict, output_path: str) -> str:
    rows = report["rows"]
    speed_rows = [row for row in rows
                  if not _short(row["label"]).startswith("FP16")]
    base = rows[0]
    best = min(speed_rows, key=lambda row: row["latency_ms"])

    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    fig.suptitle(
        "FlashDet optimization pipeline — "
        f"{report.get('model', 'checkpoint').split('/')[-1]} "
        f"({base['params']:,} params, "
        f"{report.get('input_size', '?')}×{report.get('input_size', '?')} input)",
        fontsize=13, fontweight="bold",
    )

    _bars(axes[0, 0], speed_rows, "latency_ms",
          "Speed (forward pass)", "ms", "latency (ms, lower is better)",
          extra="{fps:.0f} FPS")
    _bars(axes[0, 1], rows, "size_mb",
          "Artifact size", "MB", "file size (MB, lower is better)")
    _lollipops(axes[0, 2], rows, "map50",
               "Accuracy (mAP@0.5)", "mAP@0.5", (0.55, 0.615))
    _lollipops(axes[1, 0], rows, "box_iou",
               "Boxes vs FP32 (mean IoU)", "mean matched IoU", (0.85, 1.01))
    _tradeoff(axes[1, 1], speed_rows)
    _scorecard(axes[1, 2], base, best)

    fig.text(
        0.5, -0.02,
        "Latency is forward-only (decode/NMS excluded), median of 100 runs "
        "after 20 warmups. FP16 CPU rows are a storage format and are omitted "
        "from the speed panels. Lower is better for latency and size.",
        ha="center", fontsize=8, color="#666666",
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True,
                        help="JSON written by scripts/benchmark_pipeline.py")
    parser.add_argument("--output", default="docs/images/optimization-pipeline.png")
    args = parser.parse_args()

    with open(args.report, encoding="utf-8") as handle:
        report = json.load(handle)
    path = render(report, args.output)
    print(f"Saved: {path}")


if __name__ == "__main__":
    sys.exit(main())
