"""Tests for the sub-classifier. We verify that the structural and
action-pair categorizations correctly route known cases."""

from __future__ import annotations

import pytest

from poker_vision.corpus.ambiguity_subclassifier import (
    ConfusedActionPair,
    ErrorMode,
    HandStructure,
    build_3d_crosstab,
    classify_action_pair,
    classify_structure,
)

# =========================================================================
# Action-pair classification — the unambiguous part
# =========================================================================


@pytest.mark.parametrize(
    "predicted,actual,expected",
    [
        ("fold", "check", ConfusedActionPair.FOLD_VS_CHECK),
        ("check", "fold", ConfusedActionPair.FOLD_VS_CHECK),  # order-insensitive
        ("call", "raise", ConfusedActionPair.CALL_VS_RAISE),
        ("bet", "check", ConfusedActionPair.CHECK_VS_BET),
        ("raise", "bet", ConfusedActionPair.BET_VS_RAISE),
        ("fold", "raise", ConfusedActionPair.FOLD_VS_RAISE),  # cross-magnitude pair
    ],
)
def test_action_pair_classification(predicted, actual, expected):
    assert classify_action_pair(predicted_kind=predicted, actual_kind=actual) == expected


# =========================================================================
# Structural classification — the judgment calls
# =========================================================================


def test_preflop_heads_up_fold_to_open():
    s = classify_structure(
        street="preflop",
        num_active_players=2,
        current_bet_before_window=3,  # someone raised
        is_terminal_fold=False,
        is_first_fold_of_street=True,
    )
    assert s == HandStructure.PREFLOP_FOLD_TO_OPEN


def test_preflop_multiway():
    s = classify_structure(
        street="preflop",
        num_active_players=4,
        current_bet_before_window=3,
        is_terminal_fold=False,
        is_first_fold_of_street=False,
    )
    assert s == HandStructure.PREFLOP_MULTIWAY


def test_flop_check_around():
    s = classify_structure(
        street="flop",
        num_active_players=3,
        current_bet_before_window=0,
        is_terminal_fold=False,
        is_first_fold_of_street=False,
    )
    assert s == HandStructure.FLOP_CHECK_AROUND


def test_river_terminal_fold():
    s = classify_structure(
        street="river",
        num_active_players=2,
        current_bet_before_window=20,
        is_terminal_fold=True,
        is_first_fold_of_street=True,
    )
    assert s == HandStructure.LATE_STREET_FOLD


# =========================================================================
# Cross-tab rendering — smoke test
# =========================================================================


def test_crosstab_builds_correctly():
    from poker_vision.corpus.ambiguity_subclassifier import SubclassifiedError

    errors = [
        SubclassifiedError(
            hand_id="h1",
            window_id="w1",
            structure=HandStructure.PREFLOP_FOLD_TO_OPEN,
            action_pair=ConfusedActionPair.FOLD_VS_CALL,
            predicted_kind="fold",
            actual_kind="call",
            num_active_players=2,
            street="preflop",
            error_mode=ErrorMode.SAME_PLAYER_MAGNITUDE,
            predicted_player="p1",
            actual_player="p1",
        ),
        SubclassifiedError(
            hand_id="h2",
            window_id="w1",
            structure=HandStructure.PREFLOP_FOLD_TO_OPEN,
            action_pair=ConfusedActionPair.FOLD_VS_CALL,
            predicted_kind="fold",
            actual_kind="call",
            num_active_players=2,
            street="preflop",
            error_mode=ErrorMode.SAME_PLAYER_MAGNITUDE,
            predicted_player="p1",
            actual_player="p1",
        ),
        SubclassifiedError(
            hand_id="h3",
            window_id="w1",
            structure=HandStructure.POSTFLOP_HEADS_UP,
            action_pair=ConfusedActionPair.CALL_VS_RAISE,
            predicted_kind="call",
            actual_kind="raise",
            num_active_players=2,
            street="flop",
            error_mode=ErrorMode.SAME_PLAYER_MAGNITUDE,
            predicted_player="p1",
            actual_player="p1",
        ),
    ]

    crosstab = build_3d_crosstab(errors)
    assert crosstab[ErrorMode.SAME_PLAYER_MAGNITUDE][(HandStructure.PREFLOP_FOLD_TO_OPEN, ConfusedActionPair.FOLD_VS_CALL)] == 2
    assert crosstab[ErrorMode.SAME_PLAYER_MAGNITUDE][(HandStructure.POSTFLOP_HEADS_UP, ConfusedActionPair.CALL_VS_RAISE)] == 1