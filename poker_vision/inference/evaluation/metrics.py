"""
Metric computation for OpponentActionInferencer evaluation.

All metrics operate on (predicted_sequence, expected_sequence) pairs and
aggregate across a corpus of WindowTestCase results.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional

from poker_vision.inference.opponent_action_inferencer import (
    ActionKind,
    InferredAction,
)
from poker_vision.inference.test_corpus import Complexity, WindowTestCase

# ---------------------------------------------------------------------------
# 1. Per-case evaluation result
# ---------------------------------------------------------------------------

ACTION_KINDS: tuple[ActionKind, ...] = (
    "fold",
    "check",
    "call",
    "bet",
    "raise",
    "all_in",
)


@dataclass
class ActionComparison:
    expected: Optional[InferredAction]
    predicted: Optional[InferredAction]
    player_correct: bool
    action_type_correct: bool
    amount_correct: bool
    fully_correct: bool


@dataclass
class CaseResult:
    case_id: str
    complexity: Complexity
    street: str
    num_villains: int
    big_blind: Decimal
    sequence_exact_match: bool
    per_action: list[ActionComparison]
    window_confidence: float  # average across predicted actions
    amount_abs_errors: list[Decimal]  # only on action-type-correct, non-fold/check
    is_trivial_window: bool
    predicted_has_action: bool  # any non-fold/check predicted
    expected_has_action: bool
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 2. Amount tolerance check
# ---------------------------------------------------------------------------


def amount_within_tolerance(
    expected: Decimal,
    predicted: Decimal,
    big_blind: Decimal,
    relative_tolerance: float = 0.05,
) -> bool:
    """±1 BB OR ±5% (whichever larger). Exact match required for zero amounts."""
    if expected == 0 and predicted == 0:
        return True
    if expected == 0 or predicted == 0:
        return False
    abs_tol = max(big_blind, Decimal(str(relative_tolerance)) * abs(expected))
    return abs(expected - predicted) <= abs_tol


# ---------------------------------------------------------------------------
# 3. Sequence alignment
# ---------------------------------------------------------------------------


def _align_by_player(
    expected: list[InferredAction],
    predicted: list[InferredAction],
) -> list[tuple[Optional[InferredAction], Optional[InferredAction]]]:
    """
    Align two action sequences positionally. The inferencer always emits
    actions in turn order matching the expected order, so positional pairing
    is correct. Length mismatches yield None placeholders.
    """
    n = max(len(expected), len(predicted))
    out: list[tuple[Optional[InferredAction], Optional[InferredAction]]] = []
    for i in range(n):
        e = expected[i] if i < len(expected) else None
        p = predicted[i] if i < len(predicted) else None
        out.append((e, p))
    return out


# ---------------------------------------------------------------------------
# 4. Per-case evaluation
# ---------------------------------------------------------------------------


def evaluate_case(
    case: WindowTestCase,
    predicted: list[InferredAction],
    big_blind: Decimal,
    relative_tolerance: float = 0.05,
) -> CaseResult:
    pairs = _align_by_player(case.expected_actions, predicted)
    comparisons: list[ActionComparison] = []
    amount_errors: list[Decimal] = []
    exact_match = len(predicted) == len(case.expected_actions)

    for exp, pred in pairs:
        if exp is None or pred is None:
            comparisons.append(
                ActionComparison(
                    expected=exp,
                    predicted=pred,
                    player_correct=False,
                    action_type_correct=False,
                    amount_correct=False,
                    fully_correct=False,
                )
            )
            exact_match = False
            continue

        player_ok = exp.player_id == pred.player_id
        action_ok = exp.action == pred.action
        amount_ok = amount_within_tolerance(exp.amount, pred.amount, big_blind, relative_tolerance)
        fully_ok = player_ok and action_ok and amount_ok
        if action_ok and exp.action in ("call", "bet", "raise", "all_in"):
            amount_errors.append(abs(exp.amount - pred.amount))
        if not fully_ok:
            exact_match = False
        comparisons.append(
            ActionComparison(
                expected=exp,
                predicted=pred,
                player_correct=player_ok,
                action_type_correct=action_ok,
                amount_correct=amount_ok,
                fully_correct=fully_ok,
            )
        )

    avg_conf = sum(p.confidence for p in predicted) / len(predicted) if predicted else 1.0

    is_trivial = case.complexity == "trivial"
    predicted_has_action = any(p.action not in ("fold", "check") for p in predicted)
    expected_has_action = any(e.action not in ("fold", "check") for e in case.expected_actions)

    return CaseResult(
        case_id=case.case_id,
        complexity=case.complexity,
        street=case.metadata.get("street_at_window_end", "unknown"),
        num_villains=case.num_villains_in_window,
        big_blind=big_blind,
        sequence_exact_match=exact_match,
        per_action=comparisons,
        window_confidence=avg_conf,
        amount_abs_errors=amount_errors,
        is_trivial_window=is_trivial,
        predicted_has_action=predicted_has_action,
        expected_has_action=expected_has_action,
        metadata=dict(case.metadata),
    )


# ---------------------------------------------------------------------------
# 5. Aggregated metrics
# ---------------------------------------------------------------------------


@dataclass
class ReliabilityBin:
    lo: float
    hi: float
    count: int
    mean_confidence: float
    accuracy: float


@dataclass
class CalibrationMetrics:
    bins: list[ReliabilityBin]
    expected_calibration_error: float  # ECE
    max_calibration_error: float  # MCE
    brier_score: float


@dataclass
class AggregateMetrics:
    sequence_accuracy: float
    action_accuracy: float
    player_action_accuracy: float
    amount_mae: float
    amount_mape: float

    calibration: CalibrationMetrics

    confusion_matrix: dict[str, dict[str, int]]  # expected -> predicted -> count
    accuracy_by_complexity: dict[str, float]
    accuracy_by_num_villains: dict[int, float]
    accuracy_by_street: dict[str, float]
    false_positive_rate_trivial: float

    total_cases: int
    total_actions: int
    trivial_cases: int


def _equal_width_bins(
    confidences: list[float],
    correctness: list[bool],
    num_bins: int = 10,
) -> CalibrationMetrics:
    if not confidences:
        return CalibrationMetrics(bins=[], expected_calibration_error=0.0, max_calibration_error=0.0, brier_score=0.0)

    n = len(confidences)
    bins: list[ReliabilityBin] = []
    ece = 0.0
    mce = 0.0
    for i in range(num_bins):
        lo = i / num_bins
        hi = (i + 1) / num_bins
        if i < num_bins - 1:
            members = [(c, k) for c, k in zip(confidences, correctness) if lo <= c < hi]
        else:
            members = [(c, k) for c, k in zip(confidences, correctness) if lo <= c <= hi]
        if not members:
            bins.append(ReliabilityBin(lo, hi, 0, 0.0, 0.0))
            continue
        mean_conf = sum(c for c, _ in members) / len(members)
        acc = sum(1 for _, k in members if k) / len(members)
        bins.append(ReliabilityBin(lo, hi, len(members), mean_conf, acc))
        weight = len(members) / n
        gap = abs(mean_conf - acc)
        ece += weight * gap
        mce = max(mce, gap)

    brier = sum((c - (1.0 if k else 0.0)) ** 2 for c, k in zip(confidences, correctness)) / n

    return CalibrationMetrics(
        bins=bins,
        expected_calibration_error=ece,
        max_calibration_error=mce,
        brier_score=brier,
    )


def aggregate(results: list[CaseResult]) -> AggregateMetrics:
    if not results:
        raise ValueError("Cannot aggregate empty result set")

    total_cases = len(results)
    seq_correct = sum(1 for r in results if r.sequence_exact_match)

    all_comparisons: list[ActionComparison] = [c for r in results for c in r.per_action if c.expected is not None]
    total_actions = len(all_comparisons)
    fully = sum(1 for c in all_comparisons if c.fully_correct)
    player_action = sum(1 for c in all_comparisons if c.player_correct and c.action_type_correct)

    all_amount_errors = [float(e) for r in results for e in r.amount_abs_errors]
    amount_mae = sum(all_amount_errors) / len(all_amount_errors) if all_amount_errors else 0.0

    mape_terms: list[float] = []
    for r in results:
        for c in r.per_action:
            if (
                c.expected is not None
                and c.predicted is not None
                and c.action_type_correct
                and c.expected.action in ("call", "bet", "raise", "all_in")
                and c.expected.amount > 0
            ):
                err = abs(float(c.expected.amount - c.predicted.amount))
                mape_terms.append(err / float(c.expected.amount))
    amount_mape = sum(mape_terms) / len(mape_terms) if mape_terms else 0.0

    conf_points: list[float] = []
    correct_points: list[bool] = []
    for r in results:
        for c in r.per_action:
            if c.predicted is None:
                continue
            conf_points.append(c.predicted.confidence)
            correct_points.append(c.fully_correct)
    calibration = _equal_width_bins(conf_points, correct_points, num_bins=10)

    confusion: dict[str, dict[str, int]] = {e: {p: 0 for p in ACTION_KINDS} for e in ACTION_KINDS}
    for c in all_comparisons:
        if c.predicted is None:
            continue
        confusion[c.expected.action][c.predicted.action] += 1

    by_complexity_n: dict[str, list[bool]] = defaultdict(list)
    by_villains_n: dict[int, list[bool]] = defaultdict(list)
    by_street_n: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        by_complexity_n[r.complexity].append(r.sequence_exact_match)
        by_villains_n[r.num_villains].append(r.sequence_exact_match)
        by_street_n[r.street].append(r.sequence_exact_match)

    def _mean(xs: list[bool]) -> float:
        return sum(1 for x in xs if x) / len(xs) if xs else 0.0

    accuracy_by_complexity = {k: _mean(v) for k, v in by_complexity_n.items()}
    accuracy_by_num_villains = {k: _mean(v) for k, v in by_villains_n.items()}
    accuracy_by_street = {k: _mean(v) for k, v in by_street_n.items()}

    trivials = [r for r in results if r.is_trivial_window]
    if trivials:
        fp = sum(1 for r in trivials if r.predicted_has_action and not r.expected_has_action)
        fpr_trivial = fp / len(trivials)
    else:
        fpr_trivial = 0.0

    return AggregateMetrics(
        sequence_accuracy=seq_correct / total_cases,
        action_accuracy=fully / total_actions if total_actions else 0.0,
        player_action_accuracy=(player_action / total_actions if total_actions else 0.0),
        amount_mae=amount_mae,
        amount_mape=amount_mape,
        calibration=calibration,
        confusion_matrix=confusion,
        accuracy_by_complexity=accuracy_by_complexity,
        accuracy_by_num_villains=dict(sorted(accuracy_by_num_villains.items())),
        accuracy_by_street=accuracy_by_street,
        false_positive_rate_trivial=fpr_trivial,
        total_cases=total_cases,
        total_actions=total_actions,
        trivial_cases=len(trivials),
    )


__all__ = [
    "ACTION_KINDS",
    "AggregateMetrics",
    "ActionComparison",
    "CalibrationMetrics",
    "CaseResult",
    "ReliabilityBin",
    "aggregate",
    "amount_within_tolerance",
    "evaluate_case",
]
