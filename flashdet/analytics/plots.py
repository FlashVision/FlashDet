"""Plotting utilities — training curves, PR curves, mAP and confusion matrices."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Union

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def _get_plt():
    """Lazy-import matplotlib to avoid hard dependency at module level."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ------------------------------------------------------------------
# Training curves
# ------------------------------------------------------------------

def plot_training_curves(
    log: Dict[str, List[float]],
    keys: Optional[Sequence[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Training Curves",
) -> "Figure":
    """Plot one or more scalar metrics from a training log dict.

    Parameters
    ----------
    log : dict[str, list[float]]
        ``{"loss": [...], "lr": [...], "mAP": [...], ...}``
    keys : sequence of str | None
        Which keys to plot.  *None* plots everything.
    save_path : str | Path | None
        If given, save figure to this path.
    title : str
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _get_plt()
    keys = keys or list(log.keys())
    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    axes = axes.flatten()

    for ax, key in zip(axes, keys):
        values = log[key]
        ax.plot(values, linewidth=1.5)
        ax.set_title(key)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

    return fig


# ------------------------------------------------------------------
# Precision-Recall curve
# ------------------------------------------------------------------

def plot_pr_curve(
    precisions: np.ndarray,
    recalls: np.ndarray,
    ap: Optional[float] = None,
    class_name: str = "all",
    save_path: Optional[Union[str, Path]] = None,
) -> "Figure":
    """Plot a Precision-Recall curve.

    Parameters
    ----------
    precisions, recalls : np.ndarray
        1-D arrays of matched length.
    ap : float | None
        Average Precision value (shown in legend when provided).
    class_name : str
        Label for the curve.
    save_path : str | Path | None
        Optional file path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(6, 5))
    label = f"{class_name}"
    if ap is not None:
        label += f" (AP={ap:.3f})"
    ax.plot(recalls, precisions, linewidth=1.5, label=label)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

    return fig


# ------------------------------------------------------------------
# mAP over IoU thresholds
# ------------------------------------------------------------------

def plot_map_curve(
    iou_thresholds: np.ndarray,
    map_values: np.ndarray,
    save_path: Optional[Union[str, Path]] = None,
) -> "Figure":
    """Plot mAP at different IoU thresholds.

    Parameters
    ----------
    iou_thresholds : np.ndarray
        1-D array of IoU thresholds (e.g. 0.50, 0.55, …, 0.95).
    map_values : np.ndarray
        Corresponding mAP values.
    save_path : str | Path | None
        Optional save path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(iou_thresholds, map_values, width=0.03, color="steelblue", edgecolor="white")
    ax.set_xlabel("IoU Threshold")
    ax.set_ylabel("mAP")
    ax.set_title("mAP @ IoU Thresholds")
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

    return fig


# ------------------------------------------------------------------
# Confusion matrix
# ------------------------------------------------------------------

def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Confusion Matrix",
) -> "Figure":
    """Plot a confusion matrix as a heatmap.

    Parameters
    ----------
    matrix : np.ndarray
        Square confusion matrix of shape ``(n_classes, n_classes)``.
    class_names : list[str] | None
        Tick labels.  Auto-generated indices when *None*.
    normalize : bool
        Row-normalise the matrix before plotting.
    save_path : str | Path | None
        Optional save path.
    title : str
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _get_plt()
    n = matrix.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n)]

    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        matrix = matrix.astype(np.float64) / row_sums

    fig, ax = plt.subplots(figsize=(max(6, n * 0.6), max(5, n * 0.5)))
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    thresh = matrix.max() / 2
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            text = f"{val:.2f}" if normalize else f"{int(val)}"
            ax.text(
                j, i, text, ha="center", va="center",
                color="white" if val > thresh else "black", fontsize=7,
            )

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

    return fig
