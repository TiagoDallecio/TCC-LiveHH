"""Tests for future-knowledge reconciliation.

These tests verify the generalization: every case the showdown reconciler
catches must also be caught here, plus the new cases we expect to catch."""

from __future__ import annotations

from dataclasses import dataclass

from poker_vision.corpus.future_knowledge import compute_future_knowledge


@dataclass(frozen=True)
class MockAction:
    player_id: str
    kind: str
    street: str = "preflop"


# =========================================================================
# Generalization: must subsume existing showdown logic
# =========================================================================


def test_showdown_participant_marked_non_folder():
    """Every showdown participant must appear in non_folders, even if
    they took no recorded action after the window. This preserves
    backward compatibility with the existing reconciler."""
    knowledge = compute_future_knowledge(
        window_id="w1",
        window_anchor_index=0,
        all_actions=[MockAction("hero", "raise")],
        showdown_participants=frozenset({"hero", "v2"}),
    )
    assert "v2" in knowledge.non_folders
    assert knowledge.evidence["v2"] == "appeared at showdown"


# =========================================================================
# New cases: future action before showdown
# =========================================================================


def test_player_acting_on_later_street_marked_non_folder():
    """If v1 acts on the flop, v1 did not fold preflop."""
    actions = [
        MockAction("hero", "raise", "preflop"),  # anchor 0: preflop window
        MockAction("v1", "call", "preflop"),
        MockAction("v1", "call", "flop"),
        MockAction("hero", "bet", "flop"),
    ]
    knowledge = compute_future_knowledge(
        window_id="preflop_w1",
        window_anchor_index=0,
        all_actions=actions,
        showdown_participants=frozenset(),  # no showdown
    )
    assert "v1" in knowledge.non_folders
    assert "took action" in knowledge.evidence["v1"]


def test_player_who_eventually_folds_still_marked_for_earlier_windows():
    """The 'pattern 1' case from our hypothesis: v1 acts preflop, then
    folds on flop. The preflop window must still flag v1 as non-folder."""
    actions = [
        MockAction("hero", "raise", "preflop"),  # anchor 0
        MockAction("v1", "call", "preflop"),
        MockAction("v1", "fold", "flop"),
    ]
    knowledge = compute_future_knowledge(
        window_id="preflop_w1",
        window_anchor_index=0,
        all_actions=actions,
        showdown_participants=frozenset(),
    )
    assert "v1" in knowledge.non_folders


# =========================================================================
# Cases that correctly remain unflagged
# =========================================================================


def test_player_with_no_future_action_not_marked():
    """If v1 takes no future action and doesn't reach showdown, we cannot
    flag them as non-folder — they may have genuinely folded."""
    actions = [
        MockAction("hero", "raise", "preflop"),
        MockAction("v1", "fold", "preflop"),
    ]
    knowledge = compute_future_knowledge(
        window_id="preflop_w1",
        window_anchor_index=0,
        all_actions=actions,
        showdown_participants=frozenset(),
    )
    assert "v1" not in knowledge.non_folders


# =========================================================================
# Determinism (golden-fixture property)
# =========================================================================


def test_repeated_calls_produce_identical_knowledge():
    actions = [
        MockAction("hero", "raise"),
        MockAction("v1", "call"),
        MockAction("v2", "call"),
    ]
    k1 = compute_future_knowledge(
        window_id="w",
        window_anchor_index=0,
        all_actions=actions,
        showdown_participants=frozenset({"hero"}),
    )
    k2 = compute_future_knowledge(
        window_id="w",
        window_anchor_index=0,
        all_actions=actions,
        showdown_participants=frozenset({"hero"}),
    )
    assert k1.non_folders == k2.non_folders
    assert k1.evidence == k2.evidence
