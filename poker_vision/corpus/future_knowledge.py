"""
Computes the generalized non_folders set from a ground-truth hand history.

A player is a 'non-folder' for window W if, at any point after W ends,
that player takes an observable action: betting, calling, checking,
raising, or appearing at showdown.

This is a strict generalization of the existing showdown-based non_folders
logic. Every player flagged by showdown reconciliation will also be flagged
by this — but additionally, players who fold *eventually* but acted at
least once after window W will be flagged for W.

Example: hero raises preflop, v1 calls preflop, flop comes, v1 folds to a
hero c-bet. The existing reconciler only knows v1 didn't fold at showdown
(because no showdown happened). This module knows v1 didn't fold *preflop*
because v1 called preflop AND folded on the flop, which is a future
observable action relative to the preflop window.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WindowFutureKnowledge:
    """Per-window non_folders evidence derived from the future action stream."""

    window_id: str
    non_folders: frozenset[str]
    evidence: dict[str, str]
    """player_id -> human-readable justification, for thesis defense and
    debug visualization. E.g., {'v1': 'acted on flop (call)',
    'v2': 'appeared at showdown'}."""


def compute_future_knowledge(
    *,
    window_id: str,
    window_anchor_index: int,  # position of this window in the action stream
    all_actions: list,  # full ground-truth action stream for the hand
    showdown_participants: frozenset[str],
) -> WindowFutureKnowledge:
    """Compute the non_folders set for one window, using all information
    that becomes available *after* the window's anchor.

    The function is intentionally pure: same inputs always produce same
    outputs. No FSM state, no side effects. This is corpus-generation
    scaffolding mirroring the synthetic_fsm pattern."""

    non_folders: set[str] = set()
    evidence: dict[str, str] = {}

    # Signal 1: any player who acts after the window did not fold in the window
    future_actions = all_actions[window_anchor_index + 1 :]
    for action in future_actions:
        action_kind = getattr(action, "action", getattr(action, "kind", "unknown"))

        if action_kind not in ("fold", "unknown"):
            non_folders.add(action.player_id)
            evidence[action.player_id] = f"took action '{action_kind}' after window"

    # Signal 2: showdown participants (subset of Signal 1 in most cases,
    # but explicit because showdown can occur after all action ends)
    for player in showdown_participants:
        if player not in non_folders:
            non_folders.add(player)
            evidence[player] = "appeared at showdown"

    return WindowFutureKnowledge(
        window_id=window_id,
        non_folders=frozenset(non_folders),
        evidence=evidence,
    )


def compute_for_all_windows(
    *,
    hand_id: str,
    all_actions: list,
    window_anchors: list,  # ordered list of (window_id, anchor_index) pairs
    showdown_participants: frozenset[str],
) -> dict[str, WindowFutureKnowledge]:
    """Convenience batch-mode for a whole hand. Returns window_id -> knowledge."""
    return {
        window_id: compute_future_knowledge(
            window_id=window_id,
            window_anchor_index=anchor_idx,
            all_actions=all_actions,
            showdown_participants=showdown_participants,
        )
        for window_id, anchor_idx in window_anchors
    }
