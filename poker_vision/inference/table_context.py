from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal

ActionKind = Literal["fold", "check", "call", "bet", "raise", "all_in"]
Street = Literal["preflop", "flop", "turn", "river"]

_CONSTRAINT_FIELDS: frozenset[str] = frozenset(
    {
        "turn_pointer",
        "action_order",
        "legal_actions_per_player",
        "non_folders",
        "active_players",
        "big_blind",
    }
)


@dataclass(frozen=False)
class TableContext:
    """
    Snapshot of table state at the moment an inference window opens.

    Fields are partitioned into three epistemic categories:

    1. MEASURED STATE — values derived from CV observations (pot, stacks).
       These are noisy and the inferencer treats them as soft priors.
    2. DERIVED STATE — values computed deterministically by the FSM from
       prior events (turn pointer, legal actions). These are theorems and
       the inferencer treats them as hard constraints.
    3. FUTURE-KNOWLEDGE CONSTRAINTS — values inferred from post-hoc
       reconciliation (showdown reveals, future folds). Treated as hard
       constraints, identical mechanism to derived state.

    The partition is enforced *by convention and by the docstrings on each
    field*, not by the type system, but the integration tests verify the
    inferencer respects the partition.
    """

    # =========================================================================
    # MEASURED STATE — soft priors, may be noisy
    # =========================================================================

    pot: Decimal = Decimal("0")
    current_bet: Decimal = Decimal("0")
    last_raise_size: Decimal = Decimal("0")
    big_blind: Decimal = Decimal("1")  # <-- O salvador da pátria!

    contributions_this_street: dict[str, Decimal] = field(default_factory=dict)
    street: Street = "preflop"
    active_players: list[str] = field(default_factory=list)
    player_stacks: dict[str, Decimal] = field(default_factory=dict)
    stack_measurement_age_frames: dict[str, int] = field(default_factory=dict)

    # =========================================================================
    # DERIVED STATE — hard constraints from FSM (theorems, not measurements)
    # =========================================================================

    turn_pointer: str | None = None
    """The player whose turn it currently is, per FSM. None if the FSM
    has not established turn order yet (e.g., new street, no actions yet)."""

    action_order: tuple[str, ...] = ()
    """Ordered tuple of players to act in the current window, starting
    from turn_pointer. Empty tuple if FSM has not established it.

    Frozen as a tuple (not a list) because this is a hard constraint and
    must not be mutated post-construction."""

    legal_actions_per_player: dict[str, frozenset[ActionKind]] = field(default_factory=dict)
    """Per-player legal action set derived by the FSM from prior actions
    in the current street. Empty dict for a player = no constraint
    (treat as 'any action legal')."""

    # =========================================================================
    # FUTURE-KNOWLEDGE CONSTRAINTS — hard constraints from reconciliation
    # =========================================================================

    non_folders: frozenset[str] = field(default_factory=frozenset)
    """Players mathematically known to have NOT folded during this window.
    Populated by post-hoc reconciliation from showdown reveals or future
    fold events. Hard constraint — DFS must not generate fold for these
    players."""

    # =========================================================================
    # PROVENANCE — for debugging and thesis defense
    # =========================================================================

    constraint_sources: dict[str, str] = field(default_factory=dict)
    """Map from constraint field name → human-readable source.
    Example: {'non_folders': 'showdown_reveals',
              'turn_pointer': 'fsm_derived',
              'action_order': 'fsm_derived'}.
    Used by the debug visualizer and replay tool."""

    # =========================================================================
    # TOPOLOGY & LEGACY STATE (Backward Compatibility para a FSM)
    # =========================================================================
    num_players: int = 0
    button_seat: int = 0
    hero_seat: int = 0
    seat_order: list[str] = field(default_factory=list)
    hero_id: str = "Hero"
    current_street: Street = "preflop"

    # =========================================================================
    # INVARIANTS
    # =========================================================================

    def __post_init__(self) -> None:
        """Validate the partition invariants.

        These are runtime checks because the type system can't enforce
        cross-field consistency. They run on every construction — keep
        them cheap.
        """
        # I1: turn_pointer, if set, must be in active_players
        if self.turn_pointer is not None:
            if self.turn_pointer not in self.active_players:
                raise ValueError(
                    f"turn_pointer={self.turn_pointer!r} not in " f"active_players={self.active_players!r}"
                )

        # I2: action_order, if non-empty, must start with turn_pointer
        if self.action_order:
            if self.turn_pointer is None:
                raise ValueError("action_order is non-empty but turn_pointer is None")
            if self.action_order[0] != self.turn_pointer:
                raise ValueError(
                    f"action_order[0]={self.action_order[0]!r} does not " f"match turn_pointer={self.turn_pointer!r}"
                )
            # I3: every player in action_order must be active
            unknown = set(self.action_order) - set(self.active_players)
            if unknown:
                raise ValueError(f"action_order contains non-active players: {unknown}")

        # I4: non_folders must be a subset of active_players
        unknown_non_folders = self.non_folders - set(self.active_players)
        if unknown_non_folders:
            raise ValueError(f"non_folders contains non-active players: " f"{unknown_non_folders}")

        # I5: legal_actions_per_player keys must be active
        unknown_legal = set(self.legal_actions_per_player) - set(self.active_players)
        if unknown_legal:
            raise ValueError(f"legal_actions_per_player has entries for non-active " f"players: {unknown_legal}")

        # I6: non-negative monetary quantities
        for name, val in [
            ("pot", self.pot),
            ("current_bet", self.current_bet),
            ("last_raise_size", self.last_raise_size),
            ("big_blind", self.big_blind),
        ]:
            if val < 0:
                raise ValueError(f"{name} must be non-negative, got {val}")

        # I7: big_blind must be strictly positive (zero would break
        # normalization in the scorer)
        if self.big_blind <= 0:
            raise ValueError(f"big_blind must be > 0, got {self.big_blind}")

    # =========================================================================
    # CONVENIENCE ACCESSORS
    # =========================================================================

    @property
    def has_fsm_state(self) -> bool:
        """True iff the FSM has populated derived-state fields.

        Useful for the inferencer to know whether to apply hard turn-order
        constraints. False during the walking-skeleton phase or when the
        FSM stage is disabled."""
        return self.turn_pointer is not None

    def is_action_legal(self, player: str, action: ActionKind) -> bool:
        """Check whether an action is FSM-legal for a player.

        Returns True if no constraint is set for the player (lenient
        default) or if the action is in the player's legal set."""
        legal = self.legal_actions_per_player.get(player)
        if legal is None:
            return True
        return action in legal

    def stack_prior_weight(self, player: str, max_age_frames: int = 90) -> float:
        """Return a weight in [0, 1] for how much to trust this player's
        stack as a prior. Returns 0 if no measurement, decays linearly
        with age."""
        if player not in self.player_stacks:
            return 0.0
        age = self.stack_measurement_age_frames.get(player, 0)
        if age >= max_age_frames:
            return 0.0
        return 1.0 - (age / max_age_frames)

    def snapshot_constraints(self) -> tuple:
        """Return a hashable snapshot of all constraint fields.
        Used by the inferencer to detect mutation across an inference call."""
        # Se legal_actions_per_player for None, garanta que seja um dicionário vazio
        legal = self.legal_actions_per_player or {}
        return (
            self.turn_pointer,
            self.action_order,
            tuple(sorted(legal.items())),
            self.non_folders,
            tuple(self.active_players),
            self.big_blind,
        )

    def assert_constraints_match(self, snapshot: tuple, *, where: str) -> None:
        """Assert that constraint fields have not been mutated since snapshot."""
        current = self.snapshot_constraints()
        if current != snapshot:
            raise RuntimeError(
                f"TableContext constraint fields mutated during {where}. "
                f"This is a programming error: constraint fields "
                f"({sorted(_CONSTRAINT_FIELDS)}) must be immutable across "
                f"an inference call."
            )
