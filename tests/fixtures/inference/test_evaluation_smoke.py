from decimal import Decimal
from pathlib import Path

from poker_vision.inference.evaluation.harness import run_evaluation
from poker_vision.inference.opponent_action_inferencer import (
    AnchorEvent,
    AnchorType,
    HeroAction,
    InferredAction,
    TableContext,
)
from poker_vision.inference.test_corpus import WindowTestCase


def _make_case() -> WindowTestCase:
    ctx = TableContext(
        num_players=4,
        button_seat=0,
        hero_seat=3,
        seat_order=["V1", "V2", "V3", "Hero"],
        active_players=["V1", "V2", "V3", "Hero"],
        current_street="flop",
        current_bet=Decimal("0"),
        last_raise_size=Decimal("0"),
        turn_pointer="V1",
        pot=Decimal("30"),
        contributions_this_street={},
        hero_id="Hero",
    )
    a1 = AnchorEvent(AnchorType.STREET_START, 1.0, "flop", Decimal("30"), Decimal("30"), board=("As", "Kh", "7d"))
    a2 = AnchorEvent(
        AnchorType.HERO_ACTION,
        5.0,
        "flop",
        Decimal("30"),
        Decimal("30"),
        board=("As", "Kh", "7d"),
        hero_action=HeroAction("check", Decimal("0")),
    )
    return WindowTestCase(
        case_id="smoke::001",
        ctx_before=ctx,
        anchor_start=a1,
        anchor_end=a2,
        expected_actions=[
            InferredAction("V1", "check", Decimal("0"), confidence=1.0, is_inferred=False),
            InferredAction("V2", "check", Decimal("0"), confidence=1.0, is_inferred=False),
            InferredAction("V3", "check", Decimal("0"), confidence=1.0, is_inferred=False),
        ],
        metadata={"complexity": "trivial", "street_at_window_end": "flop", "big_blind": "1.0"},
    )


def test_harness_smoke(tmp_path: Path) -> None:
    out = run_evaluation([_make_case()], tmp_path)
    assert out.metrics.total_cases == 1
    assert out.artifact_paths["report_md"].exists()
    assert out.artifact_paths["reliability"].exists()
