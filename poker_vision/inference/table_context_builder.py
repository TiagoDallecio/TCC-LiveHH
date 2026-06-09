"""
Anti-corruption layer between the FSM and the inferencer.

Translates the FSM's mutable, event-driven HandPhase into an immutable-by-
convention TableContext snapshot. Enforces the partition between derived
state (hard constraints) and measured state (soft priors), and refuses to
produce a context that would violate TableContext's invariants.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Protocol

from poker_vision.inference.table_context import ActionKind, Street, TableContext

if TYPE_CHECKING:
    # Forward references so this module doesn't take a hard dependency on
    # the FSM implementation — we only need its *shape*. This keeps the
    # builder testable with mock FSM states.
    pass


# =========================================================================
# FSM SHAPE PROTOCOLS
# =========================================================================
# We define the FSM's shape as Protocols rather than importing concrete
# classes. This lets us:
#   1. Test the builder with mock FSM states (no FSM dependency)
#   2. Swap FSM implementations later without changing the builder
#   3. Document exactly what the builder needs from the FSM


class PlayerStateLike(Protocol):
    """Minimum interface the builder needs from a player's FSM state."""

    player_id: str
    is_active: bool  # has not folded
    stack: Decimal  # last-known chip stack
    stack_measurement_age_frames: int
    contribution_this_street: Decimal


class HandStateLike(Protocol):
    """Minimum interface the builder needs from the hand-level FSM state."""

    pot: Decimal
    current_bet: Decimal
    last_raise_size: Decimal
    big_blind: Decimal
    street: Street
    turn_pointer: str | None
    action_order: tuple[str, ...]  # players still to act this street
    players: dict[str, PlayerStateLike]  # all seated players

    def legal_actions_for(self, player_id: str) -> frozenset[ActionKind]: ...


# =========================================================================
# BUILDER
# =========================================================================


@dataclass(frozen=True)
class BuilderConfig:
    """Tunable parameters for the builder. Defaults are conservative."""

    strict: bool = True
    """If True, the builder raises on any FSM inconsistency (recommended
    for production and tests). If False, the builder logs warnings and
    drops the offending field — useful for replay of historical hands
    that may have minor inconsistencies."""

    require_fsm_state: bool = False
    """If True, the builder refuses to produce a context without
    turn_pointer / action_order. If False, it produces a 'measured-only'
    context with no derived-state fields populated (legitimate during
    the walking-skeleton phase before the FSM stage is wired in)."""


class TableContextBuilder:
    """Translates FSM state into a validated TableContext snapshot.

    Usage::

        builder = TableContextBuilder()
        ctx = builder.build(
            hand_state=fsm.current_state,
            non_folders=reconciler.non_folders_for_window(window_id),
        )
    """

    def __init__(self, config: BuilderConfig | None = None) -> None:
        self.cfg = config or BuilderConfig()

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def build(
        self,
        hand_state: HandStateLike,
        *,
        non_folders: frozenset[str] = frozenset(),
        non_folders_source: str = "showdown_reveals",
    ) -> TableContext:
        """Build a TableContext from the current FSM state.

        Parameters
        ----------
        hand_state
            The current hand-level FSM state. Must satisfy ``HandStateLike``.
        non_folders
            Players known via future-knowledge reconciliation to have not
            folded during the upcoming window. Typically empty for live
            inference and populated for evaluation against the test corpus.
        non_folders_source
            Human-readable provenance for the non_folders set. Examples:
            ``"showdown_reveals"``, ``"future_fold_events"``,
            ``"synthetic_corpus"``.

        Returns
        -------
        TableContext
            A snapshot satisfying all TableContext invariants.

        Raises
        ------
        ValueError
            If the FSM state is inconsistent (e.g., turn_pointer references
            an inactive player) and ``config.strict=True``.
        """
        # Step 1: extract measured state (always required).
        measured = self._extract_measured_state(hand_state)

        # Step 2: extract derived state (optional, depending on config).
        derived = self._extract_derived_state(hand_state)

        # Step 3: validate the non_folders constraint against the FSM's
        # active-player set. This catches the case where reconciliation
        # claims a player is a non-folder but the FSM has them as folded.
        validated_non_folders = self._validate_non_folders(non_folders, measured["active_players"])

        # Step 4: build the provenance map.
        provenance = self._build_provenance(
            has_derived=derived is not None,
            non_folders_source=non_folders_source if validated_non_folders else None,
        )

        # Step 5: construct the context. TableContext.__post_init__ will
        # run all I1–I7 invariant checks; if anything is wrong, we get a
        # clear error before the context ever reaches the inferencer.
        try:
            return TableContext(
                # --- measured ---
                pot=measured["pot"],
                current_bet=measured["current_bet"],
                last_raise_size=measured["last_raise_size"],
                big_blind=measured["big_blind"],
                contributions_this_street=measured["contributions"],
                street=measured["street"],
                active_players=measured["active_players"],
                player_stacks=measured["stacks"],
                stack_measurement_age_frames=measured["stack_ages"],
                # --- derived (may be None/empty) ---
                turn_pointer=derived["turn_pointer"] if derived else None,
                action_order=derived["action_order"] if derived else (),
                legal_actions_per_player=(derived["legal_actions"] if derived else {}),
                # --- future-knowledge ---
                non_folders=validated_non_folders,
                # --- provenance ---
                constraint_sources=provenance,
            )
        except ValueError as e:
            # Re-raise with context about which FSM state caused it.
            raise ValueError(
                f"TableContextBuilder failed to produce a valid context "
                f"from FSM state (street={measured['street']}, "
                f"pot={measured['pot']}, turn={derived['turn_pointer'] if derived else None}): {e}"
            ) from e

    # ---------------------------------------------------------------------
    # Extraction helpers
    # ---------------------------------------------------------------------

    def _extract_measured_state(self, hand: HandStateLike) -> dict:
        """Pull all measured-state fields out of the FSM. These are
        always required and never optional."""
        active = [pid for pid, p in hand.players.items() if p.is_active]
        # Sort for deterministic ordering — matters for reproducible
        # DFS results and golden-fixture tests.
        active.sort()

        contributions = {pid: hand.players[pid].contribution_this_street for pid in active}
        stacks = {pid: hand.players[pid].stack for pid in active}
        stack_ages = {pid: hand.players[pid].stack_measurement_age_frames for pid in active}

        return {
            "pot": hand.pot,
            "current_bet": hand.current_bet,
            "last_raise_size": hand.last_raise_size,
            "big_blind": hand.big_blind,
            "contributions": contributions,
            "street": hand.street,
            "active_players": active,
            "stacks": stacks,
            "stack_ages": stack_ages,
        }

    def _extract_derived_state(self, hand: HandStateLike) -> dict | None:
        """Pull derived-state fields out of the FSM if available.

        Returns None if the FSM has not established turn order yet.
        In strict mode + require_fsm_state, raises instead of returning None.
        """
        if hand.turn_pointer is None:
            if self.cfg.require_fsm_state and self.cfg.strict:
                raise ValueError(
                    "FSM has no turn_pointer but require_fsm_state=True. "
                    "Either initialize the FSM before building context or "
                    "disable require_fsm_state."
                )
            return None

        # Build legal-actions map only for active players. The FSM owns
        # the per-player legal logic; we just snapshot the result.
        legal_actions: dict[str, frozenset[ActionKind]] = {}
        for pid, player in hand.players.items():
            if not player.is_active:
                continue
            legal = hand.legal_actions_for(pid)
            # Empty set would mean "no legal action" which is nonsense —
            # the FSM should have transitioned. Catch this as a bug.
            if not legal and self.cfg.strict:
                raise ValueError(
                    f"FSM returned empty legal_actions for active player "
                    f"{pid!r}. This indicates an FSM state-machine bug."
                )
            legal_actions[pid] = legal

        return {
            "turn_pointer": hand.turn_pointer,
            "action_order": tuple(hand.action_order),  # ensure immutable
            "legal_actions": legal_actions,
        }

    # ---------------------------------------------------------------------
    # Validation helpers
    # ---------------------------------------------------------------------

    def _validate_non_folders(
        self,
        non_folders: frozenset[str],
        active_players: list[str],
    ) -> frozenset[str]:
        """Validate that the reconciliation's non_folders set is consistent
        with the FSM's view of active players.

        If a player is claimed as a non-folder but the FSM has them as
        folded, we have a contradiction:
        - In strict mode: raise (FSM is the source of truth, reconciliation
          must be wrong)
        - In lenient mode: drop the offending players and log
        """
        active_set = set(active_players)
        contradictions = non_folders - active_set
        if not contradictions:
            return non_folders

        if self.cfg.strict:
            raise ValueError(
                f"non_folders contains players the FSM has as folded: "
                f"{sorted(contradictions)}. The FSM is the source of truth "
                f"for active-player status. This indicates a reconciliation "
                f"bug or an FSM-event-application bug upstream."
            )

        # Lenient: drop the contradictions.
        import logging

        logging.getLogger(__name__).warning(
            "Dropping non_folders entries inconsistent with FSM: %s",
            sorted(contradictions),
        )
        return non_folders - contradictions

    def _build_provenance(
        self,
        *,
        has_derived: bool,
        non_folders_source: str | None,
    ) -> dict[str, str]:
        """Build the provenance map for the resulting context. Used by
        the debug visualizer and the thesis-defense replay tool."""
        sources: dict[str, str] = {
            "pot": "fsm_ledger",
            "current_bet": "fsm_ledger",
            "contributions_this_street": "fsm_ledger",
            "active_players": "fsm_derived",
            "player_stacks": "chip_detector_via_fsm",
        }
        if has_derived:
            sources["turn_pointer"] = "fsm_derived"
            sources["action_order"] = "fsm_derived"
            sources["legal_actions_per_player"] = "fsm_derived"
        if non_folders_source:
            sources["non_folders"] = non_folders_source
        return sources
