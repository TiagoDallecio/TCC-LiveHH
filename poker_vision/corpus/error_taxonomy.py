"""
Classifies inferencer errors into categories that map to specific
architectural remedies.

The categorization answers one question per error: 'What information,
if available at inference time, would have resolved this ambiguity?'

The answer determines which architectural component to build next.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class ErrorCategory(Enum):
    """Each category corresponds to a specific architectural remedy."""

    FUTURE_ACTION_AVAILABLE = "future_action_available"
    """A player misattributed as folding takes an action in a later round.
    Resolved by: generalized non_folders reconciliation (4.5.9a)."""

    POT_COLLECTOR_KNOWN = "pot_collector_known"
    """A player misattributed as folding ultimately collects the pot.
    Resolved by: pot-collection reconciliation (4.5.9b)."""

    SHOWDOWN_REVEAL_AVAILABLE = "showdown_reveal_available"
    """A player misattributed as folding appears at showdown. Should
    have been caught by existing non_folders logic — if this category
    is non-empty, there's a bug in the current reconciler."""

    GENUINELY_AMBIGUOUS = "genuinely_ambiguous"
    """No future information disambiguates this error. The misattributed
    fold *did* fold; the inferencer simply picked the wrong player from
    among multiple legally-equivalent candidates. Not resolvable by any
    temporal constraint — would require a different signal entirely
    (e.g., gesture detection, occupancy heuristics)."""


@dataclass(frozen=True)
class ClassifiedError:
    hand_id: str
    window_id: str
    predicted_action: str  # what the inferencer said (e.g., "v1 folded")
    actual_action: str  # ground truth (e.g., "v1 raised")
    misattributed_player: str
    category: ErrorCategory
    evidence: str  # human-readable justification for the category


def classify_error(
    *,
    hand_id: str,
    window_id: str,
    misattributed_player: str,
    predicted_kind: str,
    actual_kind: str,
    hand_actions_after_window: Iterable,  # all ground-truth actions after this window
    hand_pot_collector: str | None,  # who collected the pot at hand end
    hand_showdown_participants: frozenset[str],  # who reached showdown
) -> ClassifiedError:
    """Assign one error to one category. Categories are checked in priority
    order — the first match wins, because the categories are not mutually
    exclusive but the remedy with the broadest applicability should claim
    the error (so we don't over-attribute fixes to narrow remedies).

    Priority: future_action > pot_collector > showdown > genuinely_ambiguous.
    Rationale: future_action is the broadest signal; pot_collector and
    showdown are special cases of 'this player did something after the
    window.' We want to count an error against the broadest remedy that
    would resolve it, because that's the remedy we'd build first."""

    # Priority 1: Did the misattributed player take any action after this window?
    future_actions = [a for a in hand_actions_after_window if a.player_id == misattributed_player]
    if future_actions:
        return ClassifiedError(
            hand_id=hand_id,
            window_id=window_id,
            predicted_action=f"{misattributed_player} {predicted_kind}",
            actual_action=f"{misattributed_player} {actual_kind}",
            misattributed_player=misattributed_player,
            category=ErrorCategory.FUTURE_ACTION_AVAILABLE,
            evidence=(
                f"{misattributed_player} took {len(future_actions)} actions "
                f"after this window, including {future_actions[0].action} on "
                f"{future_actions[0].street}"
            ),
        )

    # Priority 2: Did the misattributed player collect the pot?
    if hand_pot_collector == misattributed_player:
        return ClassifiedError(
            hand_id=hand_id,
            window_id=window_id,
            predicted_action=f"{misattributed_player} {predicted_kind}",
            actual_action=f"{misattributed_player} {actual_kind}",
            misattributed_player=misattributed_player,
            category=ErrorCategory.POT_COLLECTOR_KNOWN,
            evidence=f"{misattributed_player} collected the pot at hand end",
        )

    # Priority 3: Did the misattributed player reach showdown?
    if misattributed_player in hand_showdown_participants:
        return ClassifiedError(
            hand_id=hand_id,
            window_id=window_id,
            predicted_action=f"{misattributed_player} {predicted_kind}",
            actual_action=f"{misattributed_player} {actual_kind}",
            misattributed_player=misattributed_player,
            category=ErrorCategory.SHOWDOWN_REVEAL_AVAILABLE,
            evidence=(
                f"{misattributed_player} appeared at showdown but was "
                f"misattributed despite existing non_folders logic"
            ),
        )

    # Priority 4: No future information disambiguates this.
    return ClassifiedError(
        hand_id=hand_id,
        window_id=window_id,
        predicted_action=f"{misattributed_player} {predicted_kind}",
        actual_action=f"{misattributed_player} {actual_kind}",
        misattributed_player=misattributed_player,
        category=ErrorCategory.GENUINELY_AMBIGUOUS,
        evidence=(
            f"{misattributed_player} took no further action, did not collect "
            f"pot, did not reach showdown. Likely a true fold attributed to "
            f"the wrong player among legally-equivalent candidates."
        ),
    )


def summarize_categories(
    errors: Iterable[ClassifiedError],
) -> dict[ErrorCategory, int]:
    """Count errors per category. Used to decide which remedy to build."""
    counts = {cat: 0 for cat in ErrorCategory}
    for err in errors:
        counts[err.category] += 1
    return counts
