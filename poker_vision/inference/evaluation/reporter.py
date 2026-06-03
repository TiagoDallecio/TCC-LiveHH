"""
Report writers: JSON, Markdown, and per-case CSV.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from poker_vision.inference.evaluation.metrics import (
    ACTION_KINDS,
    AggregateMetrics,
    CaseResult,
)


def write_json_report(metrics: AggregateMetrics, out_path: Path) -> None:
    payload = {
        "summary": {
            "total_cases": metrics.total_cases,
            "total_actions": metrics.total_actions,
            "trivial_cases": metrics.trivial_cases,
            "sequence_accuracy": metrics.sequence_accuracy,
            "action_accuracy": metrics.action_accuracy,
            "player_action_accuracy": metrics.player_action_accuracy,
            "amount_mae": metrics.amount_mae,
            "amount_mape": metrics.amount_mape,
            "false_positive_rate_trivial": metrics.false_positive_rate_trivial,
        },
        "calibration": {
            "expected_calibration_error": metrics.calibration.expected_calibration_error,
            "max_calibration_error": metrics.calibration.max_calibration_error,
            "brier_score": metrics.calibration.brier_score,
            "bins": [asdict(b) for b in metrics.calibration.bins],
        },
        "confusion_matrix": metrics.confusion_matrix,
        "stratified": {
            "by_complexity": metrics.accuracy_by_complexity,
            "by_num_villains": {str(k): v for k, v in metrics.accuracy_by_num_villains.items()},
            "by_street": metrics.accuracy_by_street,
        },
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_markdown_report(metrics: AggregateMetrics, out_path: Path, chart_paths: dict[str, Path]) -> None:
    lines: list[str] = []
    lines.append("# Opponent Action Inferencer — Evaluation Report\n")
    lines.append("## Summary\n")
    lines.append(f"- **Total cases:** {metrics.total_cases:,}")
    lines.append(f"- **Total actions:** {metrics.total_actions:,}")
    lines.append(f"- **Trivial cases:** {metrics.trivial_cases:,}\n")

    lines.append("## Coarse Metrics\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Sequence accuracy | {metrics.sequence_accuracy:.3%} |")
    lines.append(f"| Action accuracy | {metrics.action_accuracy:.3%} |")
    lines.append(f"| Player+Action accuracy | " f"{metrics.player_action_accuracy:.3%} |")
    lines.append(f"| Amount MAE | {metrics.amount_mae:.3f} |")
    lines.append(f"| Amount MAPE | {metrics.amount_mape:.3%} |")
    lines.append(f"| **False-positive rate (trivial windows)** | " f"**{metrics.false_positive_rate_trivial:.3%}** |\n")

    lines.append("## Calibration\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| ECE (Expected Calibration Error) | " f"{metrics.calibration.expected_calibration_error:.4f} |")
    lines.append(f"| MCE (Max Calibration Error) | " f"{metrics.calibration.max_calibration_error:.4f} |")
    lines.append(f"| Brier Score | " f"{metrics.calibration.brier_score:.4f} |\n")

    if "reliability" in chart_paths:
        lines.append(f"![Reliability Diagram]({chart_paths['reliability'].name})\n")

    lines.append("## Confusion Matrix\n")
    header = "| Expected \\ Predicted | " + " | ".join(ACTION_KINDS) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(ACTION_KINDS) + 1))
    for e in ACTION_KINDS:
        row = [f"**{e}**"] + [str(metrics.confusion_matrix[e][p]) for p in ACTION_KINDS]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    if "confusion" in chart_paths:
        lines.append(f"![Confusion Matrix]({chart_paths['confusion'].name})\n")

    lines.append("## Stratified Accuracy\n")
    lines.append("### By Complexity\n")
    lines.append("| Complexity | Sequence accuracy |")
    lines.append("|---|---|")
    for k in ("trivial", "simple", "moderate", "complex"):
        v = metrics.accuracy_by_complexity.get(k, 0.0)
        lines.append(f"| {k} | {v:.3%} |")
    lines.append("")
    if "complexity" in chart_paths:
        lines.append(f"![Accuracy by Complexity]" f"({chart_paths['complexity'].name})\n")

    lines.append("### By Number of Villains in Window\n")
    lines.append("| # villains | Sequence accuracy |")
    lines.append("|---|---|")
    for k, v in metrics.accuracy_by_num_villains.items():
        lines.append(f"| {k} | {v:.3%} |")
    lines.append("")
    if "villains" in chart_paths:
        lines.append(f"![Accuracy by Window Size]" f"({chart_paths['villains'].name})\n")

    lines.append("### By Street\n")
    lines.append("| Street | Sequence accuracy |")
    lines.append("|---|---|")
    for k in ("preflop", "flop", "turn", "river"):
        v = metrics.accuracy_by_street.get(k, 0.0)
        lines.append(f"| {k} | {v:.3%} |")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_per_case_csv(results: list[CaseResult], out_path: Path) -> None:
    fieldnames = [
        "case_id",
        "complexity",
        "street",
        "num_villains",
        "sequence_exact_match",
        "window_confidence",
        "num_actions_expected",
        "num_actions_correct",
        "is_trivial_window",
        "false_positive",
        "amount_mae_in_case",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            n_expected = sum(1 for c in r.per_action if c.expected is not None)
            n_correct = sum(1 for c in r.per_action if c.fully_correct)
            errs = [float(e) for e in r.amount_abs_errors]
            mae = sum(errs) / len(errs) if errs else 0.0
            is_false_pos = bool(
                r.is_trivial_window
                and getattr(r, "predicted_has_action", False)
                and not getattr(r, "expected_has_action", False)
            )
            writer.writerow(
                {
                    "case_id": r.case_id,
                    "complexity": r.complexity,
                    "street": r.street,
                    "num_villains": r.num_villains,
                    "sequence_exact_match": int(r.sequence_exact_match),
                    "window_confidence": f"{r.window_confidence:.4f}",
                    "num_actions_expected": n_expected,
                    "num_actions_correct": n_correct,
                    "is_trivial_window": int(r.is_trivial_window),
                    "false_positive": int(is_false_pos),
                    "amount_mae_in_case": f"{mae:.4f}",
                }
            )


__all__ = [
    "write_json_report",
    "write_markdown_report",
    "write_per_case_csv",
]
