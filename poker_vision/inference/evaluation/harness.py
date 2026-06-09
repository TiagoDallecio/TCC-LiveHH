"""
Top-level harness: runs the inferencer over a corpus and produces all outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Optional

from poker_vision.inference.evaluation.metrics import (
    AggregateMetrics,
    CaseResult,
    aggregate,
    evaluate_case,
)
from poker_vision.inference.evaluation.plots import (
    plot_accuracy_by_complexity,
    plot_accuracy_by_num_villains,
    plot_confusion_matrix,
    plot_reliability_diagram,
)
from poker_vision.inference.evaluation.reporter import (
    write_json_report,
    write_markdown_report,
    write_per_case_csv,
)
from poker_vision.inference.opponent_action_inferencer import (
    InferencerConfig,
    OpponentActionInferencer,
)
from poker_vision.inference.test_corpus import WindowTestCase, load_corpus


@dataclass
class HarnessOutputs:
    metrics: AggregateMetrics
    per_case: list[CaseResult]
    artifact_paths: dict[str, Path]


def run_evaluation(
    corpus: list[WindowTestCase],
    output_dir: Path,
    config: Optional[InferencerConfig] = None,
    default_big_blind: Decimal = Decimal("1.0"),
    relative_tolerance: float = 0.05,
) -> HarnessOutputs:
    output_dir.mkdir(parents=True, exist_ok=True)

    per_case: list[CaseResult] = []
    for case in corpus:
        inf = OpponentActionInferencer(config)
        inf.on_anchor(case.anchor_start, case.ctx_before)
        predicted = inf.on_anchor(
            case.anchor_end, case.ctx_before, non_folders=frozenset(getattr(case, "non_folders", []))
        )

        bb = Decimal(str(case.metadata.get("big_blind", default_big_blind)))
        per_case.append(
            evaluate_case(
                case,
                predicted,
                big_blind=bb,
                relative_tolerance=relative_tolerance,
            )
        )

    metrics = aggregate(per_case)

    chart_paths = {
        "reliability": output_dir / "reliability_diagram.png",
        "confusion": output_dir / "confusion_matrix_run2.png",
        "complexity": output_dir / "accuracy_by_complexity.png",
        "villains": output_dir / "accuracy_by_num_villains.png",
    }
    plot_reliability_diagram(metrics, chart_paths["reliability"])
    plot_confusion_matrix(metrics, chart_paths["confusion"])
    plot_accuracy_by_complexity(metrics, chart_paths["complexity"])
    plot_accuracy_by_num_villains(metrics, chart_paths["villains"])

    json_path = output_dir / "report.json"
    md_path = output_dir / "report.md"
    csv_path = output_dir / "per_case.csv"
    write_json_report(metrics, json_path)
    write_markdown_report(metrics, md_path, chart_paths)
    write_per_case_csv(per_case, csv_path)

    artifact_paths = {
        "report_json": json_path,
        "report_md": md_path,
        "per_case_csv": csv_path,
        **chart_paths,
    }
    return HarnessOutputs(metrics=metrics, per_case=per_case, artifact_paths=artifact_paths)


def run_from_corpus_file(
    corpus_path: Path,
    output_dir: Path,
    config: Optional[InferencerConfig] = None,
) -> HarnessOutputs:
    corpus = load_corpus(corpus_path)
    return run_evaluation(corpus, output_dir, config=config)


__all__ = ["HarnessOutputs", "run_evaluation", "run_from_corpus_file"]
