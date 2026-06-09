"""Tests for synthetic FSM state computation.

These tests are CRITICAL because the synthetic FSM is the ground truth
we'll use to validate hard-constraint pruning. If the synthetic state is
wrong, every accuracy number we measure in 4.5.8.4c will be a lie.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from poker_vision.corpus.synthetic_fsm import (
    HistoryAction,
    compute_synthetic_state,
)

BB = Decimal("10")
SB = Decimal("5")


# =========================================================================
# Preflop turn order
# =========================================================================


def test_preflop_6max_first_to_act_is_utg():
    """In 6-max, UTG is 3 seats left of button. After blinds post, UTG
    acts first."""
    snap = compute_synthetic_state(
        seat_order=["sb", "bb", "utg", "mp", "co", "btn"],
        button_player_id="btn",
        big_blind=BB,
        small_blind=SB,
        actions_so_far=[],
        current_street="preflop",
    )
    assert snap.turn_pointer == "utg"
    assert snap.action_order == ("utg", "mp", "co", "btn", "sb", "bb")


def test_preflop_heads_up_button_acts_first():
    """In heads-up, button is SB and acts first preflop."""
    snap = compute_synthetic_state(
        seat_order=["btn", "bb"],
        button_player_id="btn",
        big_blind=BB,
        small_blind=SB,
        actions_so_far=[],
        current_street="preflop",
    )
    assert snap.turn_pointer == "btn"


def test_preflop_after_one_fold_advances_correctly():
    snap = compute_synthetic_state(
        seat_order=["sb", "bb", "utg", "mp", "co", "btn"],
        button_player_id="btn",
        big_blind=BB,
        small_blind=SB,
        actions_so_far=[HistoryAction("utg", "fold")],
        current_street="preflop",
    )
    assert snap.turn_pointer == "mp"
    assert "utg" not in snap.action_order


# =========================================================================
# Postflop turn order
# =========================================================================


def test_postflop_first_to_act_is_sb_when_active():
    """On the flop, first active player left of button acts. Replay a
    preflop where everyone limps so we reach the flop with all players."""
    actions = [
        HistoryAction("utg", "call", BB),
        HistoryAction("mp", "call", BB),
        HistoryAction("co", "call", BB),
        HistoryAction("btn", "call", BB),
        HistoryAction("sb", "call", BB - SB),
        HistoryAction("bb", "check"),
    ]
    snap = compute_synthetic_state(
        seat_order=["sb", "bb", "utg", "mp", "co", "btn"],
        button_player_id="btn",
        big_blind=BB,
        small_blind=SB,
        actions_so_far=actions,
        current_street="flop",
    )
    assert snap.turn_pointer == "sb"


def test_postflop_skips_folded_player():
    actions = [
        HistoryAction("utg", "fold"),
        HistoryAction("mp", "call", BB),
        HistoryAction("co", "fold"),
        HistoryAction("btn", "call", BB),
        HistoryAction("sb", "fold"),
        HistoryAction("bb", "check"),
    ]
    snap = compute_synthetic_state(
        seat_order=["sb", "bb", "utg", "mp", "co", "btn"],
        button_player_id="btn",
        big_blind=BB,
        small_blind=SB,
        actions_so_far=actions,
        current_street="flop",
    )
    # SB folded, so BB (next left of button among actives) acts first
    assert snap.turn_pointer == "bb"
    assert set(snap.action_order) == {"bb", "mp", "btn"}


# =========================================================================
# Legal action sets
# =========================================================================


def test_legal_actions_facing_bet():
    actions = [HistoryAction("utg", "raise", Decimal("30"))]
    snap = compute_synthetic_state(
        seat_order=["sb", "bb", "utg", "mp", "co", "btn"],
        button_player_id="btn",
        big_blind=BB,
        small_blind=SB,
        actions_so_far=actions,
        current_street="preflop",
    )
    assert snap.legal_actions_per_player["mp"] == frozenset({"fold", "call", "raise"})


def test_legal_actions_no_bet_postflop():
    """First to act postflop with no bet: can check or bet."""
    actions = [
        HistoryAction("utg", "call", BB),
        HistoryAction("mp", "fold"),
        HistoryAction("co", "fold"),
        HistoryAction("btn", "fold"),
        HistoryAction("sb", "fold"),
        HistoryAction("bb", "check"),
    ]
    snap = compute_synthetic_state(
        seat_order=["sb", "bb", "utg", "mp", "co", "btn"],
        button_player_id="btn",
        big_blind=BB,
        small_blind=SB,
        actions_so_far=actions,
        current_street="flop",
    )
    assert snap.legal_actions_per_player["bb"] == frozenset({"check", "bet"})


# =========================================================================
# Inconsistency detection
# =========================================================================


def test_replay_detects_street_mismatch():
    """If the action stream doesn't reach the claimed street, raise."""
    with pytest.raises(ValueError, match="inconsistent with anchor"):
        compute_synthetic_state(
            seat_order=["sb", "bb", "utg", "mp", "co", "btn"],
            button_player_id="btn",
            big_blind=BB,
            small_blind=SB,
            actions_so_far=[HistoryAction("utg", "fold")],
            current_street="flop",  # but we're still preflop!
        )
