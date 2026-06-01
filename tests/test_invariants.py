from decimal import Decimal

import pytest

from poker_vision.logic.invariants import InvariantViolationError, check_invariants
from poker_vision.logic.models import HandState, PlayerState


def _make_clean_hand_state() -> HandState:
    return HandState(
        street="preflop",
        pot=Decimal("30"),
        current_bet_to_match=Decimal("10"),
        action_on_seat=0,
        players={
            "Hero": PlayerState(seat=0, stack=Decimal("100"), current_bet=Decimal("10")),
            "Villain_1": PlayerState(seat=1, stack=Decimal("100"), current_bet=Decimal("10")),
            "Villain_2": PlayerState(seat=2, stack=Decimal("100"), current_bet=Decimal("10"), has_folded=True),
        },
    )


def test_check_invariants_raises_for_negative_stack() -> None:
    hand_state = _make_clean_hand_state()
    hand_state.players["Hero"].stack = Decimal("-1")

    with pytest.raises(InvariantViolationError, match="negative stack"):
        check_invariants(hand_state)

    assert hand_state.quality.needs_review is True


def test_check_invariants_passes_for_clean_state() -> None:
    hand_state = _make_clean_hand_state()

    check_invariants(hand_state)

    assert hand_state.quality.needs_review is False
