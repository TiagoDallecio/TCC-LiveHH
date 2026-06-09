"""
Matplotlib plots for the evaluation harness. All plots are thesis-ready:
no chart junk, clear axis labels, consistent color palette.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from poker_vision.inference.evaluation.metrics import (
    ACTION_KINDS,
    AggregateMetrics,
)

_PALETTE = {
    "primary": "#2E5EAA",
    "secondary": "#E07A1F",
    "accent": "#3C8D40",
    "neutral": "#6C757D",
    "grid": "#E5E5E5",
}


def _style_axes(ax) -> None:
    ax.grid(True, color=_PALETTE["grid"], linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_reliability_diagram(metrics: AggregateMetrics, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    bins = metrics.calibration.bins
    centers = [(b.lo + b.hi) / 2 for b in bins]
    accuracies = [b.accuracy for b in bins]
    counts = [b.count for b in bins]
    widths = [(b.hi - b.lo) * 0.9 for b in bins]

    ax.bar(
        centers,
        accuracies,
        width=widths,
        color=_PALETTE["primary"],
        alpha=0.85,
        edgecolor="white",
        label="Empirical accuracy",
    )
    ax.plot([0, 1], [0, 1], color=_PALETTE["neutral"], linestyle="--", linewidth=1.2, label="Perfect calibration")

    # Annotate counts above each bar
    for c, a, n in zip(centers, accuracies, counts):
        if n > 0:
            ax.text(c, min(a + 0.03, 1.02), f"n={n}", ha="center", va="bottom", fontsize=8, color=_PALETTE["neutral"])

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(
        f"Reliability Diagram  "
        f"(ECE={metrics.calibration.expected_calibration_error:.3f}, "
        f"Brier={metrics.calibration.brier_score:.3f})"
    )
    ax.legend(loc="upper left", frameon=False)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(metrics: AggregateMetrics, out_path: Path) -> None:
    matrix = np.array([[metrics.confusion_matrix[e][p] for p in ACTION_KINDS] for e in ACTION_KINDS], dtype=float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, where=row_sums > 0, out=np.zeros_like(matrix))

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(ACTION_KINDS)))
    ax.set_yticks(range(len(ACTION_KINDS)))
    ax.set_xticklabels(ACTION_KINDS, rotation=45, ha="right")
    ax.set_yticklabels(ACTION_KINDS)
    ax.set_xlabel("Predicted action")
    ax.set_ylabel("Expected action")
    ax.set_title("Action Confusion Matrix (row-normalized)")

    for i in range(len(ACTION_KINDS)):
        for j in range(len(ACTION_KINDS)):
            raw = int(matrix[i, j])
            norm = normalized[i, j]
            if raw == 0:
                continue
            color = "white" if norm > 0.5 else "black"
            ax.text(j, i, f"{raw}\n{norm:.0%}", ha="center", va="center", color=color, fontsize=8)

    fig.colorbar(im, ax=ax, shrink=0.85, label="Row fraction")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_accuracy_by_complexity(metrics: AggregateMetrics, out_path: Path) -> None:
    order = ["trivial", "simple", "moderate", "complex"]
    values = [metrics.accuracy_by_complexity.get(k, 0.0) for k in order]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(order, values, color=_PALETTE["primary"], alpha=0.85)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.1%}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Sequence accuracy")
    ax.set_xlabel("Window complexity")
    ax.set_title("Sequence Accuracy by Window Complexity")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_accuracy_by_num_villains(metrics: AggregateMetrics, out_path: Path) -> None:
    items = sorted(metrics.accuracy_by_num_villains.items())
    xs = [k for k, _ in items]
    ys = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(xs, ys, marker="o", linewidth=2, color=_PALETTE["primary"], markersize=7)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.1%}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Number of villains in window")
    ax.set_ylabel("Sequence accuracy")
    ax.set_title("Sequence Accuracy by Window Size")
    if xs:
        ax.set_xticks(xs)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


__all__ = [
    "plot_accuracy_by_complexity",
    "plot_accuracy_by_num_villains",
    "plot_confusion_matrix",
    "plot_reliability_diagram",
]
