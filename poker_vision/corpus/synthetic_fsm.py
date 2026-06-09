"""
Computes synthetic FSM state (turn_pointer, action_order, legal_actions)
from ground-truth hand histories.

This module exists for ONE purpose: to let us validate the value of FSM-
derived hard constraints in the inferencer BEFORE writing the actual FSM
stage. By replaying the known-correct action stream from a hand history
and computing what the FSM *would* have produced at each anchor point,
we can inject these constraints into the corpus and measure the
inferencer's accuracy improvement.

If the predicted improvement materializes, we proceed with the real FSM
implementation. If it doesn't, we've saved ourselves weeks of work and
need to rethink the architecture.

This module is NOT used in live inference. It is corpus-generation
scaffolding only.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable, cast

from poker_vision.inference.opponent_action_inferencer import ActionKind, Street


@dataclass
class _SeatState:
    """Per-player state during replay. Internal to this module."""

    player_id: str
    seat_index: int
    is_active: bool = True
    is_all_in: bool = False
    contribution_this_street: Decimal = Decimal("0")
    has_acted_this_street: bool = False

    @property
    def can_act(self) -> bool:
        """A player can act iff they're still in the hand and not all-in."""
        return self.is_active and not self.is_all_in


@dataclass
class _ReplayState:
    """Full table state during replay. Internal."""

    seats: list[_SeatState]
    button_seat: int
    street: Street = "preflop"
    current_bet: Decimal = Decimal("0")
    last_raise_size: Decimal = Decimal("0")
    last_aggressor_seat: int | None = None

    def active_can_act(self) -> list[_SeatState]:
        return [s for s in self.seats if s.can_act]

    def by_id(self, pid: str) -> _SeatState:
        for s in self.seats:
            if s.player_id == pid:
                return s
        raise KeyError(f"Unknown player_id: {pid!r}")


@dataclass(frozen=True)
class SyntheticFSMSnapshot:
    """The result of replaying a hand history up to a specific anchor.

    Mirrors the derived-state portion of TableContext so the corpus
    generator can plug it straight into TableContextBuilder."""

    turn_pointer: str | None
    action_order: tuple[str, ...]
    legal_actions_per_player: dict[str, frozenset[ActionKind]]
    street: Street


@dataclass(frozen=True)
class HistoryAction:
    """One action from a ground-truth hand history."""

    player_id: str
    kind: ActionKind
    amount: Decimal = Decimal("0")


# =========================================================================
# Replay engine
# =========================================================================


def compute_synthetic_state(
    *,
    seat_order: list[str],
    button_player_id: str,
    big_blind: Decimal,
    small_blind: Decimal,
    actions_so_far: Iterable[HistoryAction],
    current_street: Street,
) -> SyntheticFSMSnapshot:
    """Replay the action stream and return the synthetic FSM snapshot.

    Parameters
    ----------
    seat_order
        Players in clockwise order starting from the small blind seat.
    button_player_id
        Player on the dealer button (used to anchor blind posts).
    big_blind, small_blind
        Blind sizes for the hand.
    actions_so_far
        All voluntary actions that have occurred *before* the inference
        anchor. Does NOT include blind posts (those are handled internally).
    current_street
        The street the inference anchor is on. The replay engine asserts
        the action stream is consistent with this.

    Returns
    -------
    SyntheticFSMSnapshot
        Captures who is to act next, the order of subsequent actors, and
        each active player's legal action set.
    """
    state = _initialize_with_blinds(
        seat_order=seat_order,
        button_player_id=button_player_id,
        big_blind=big_blind,
        small_blind=small_blind,
    )

    for action in actions_so_far:
        _apply_action(state, action)
        if _street_closed(state):
            _advance_street(state)

    if state.street != current_street:
        raise ValueError(
            f"Replay landed on street={state.street!r} but anchor "
            f"claims street={current_street!r}. Hand history is "
            f"inconsistent with anchor metadata."
        )

    return _snapshot(state)


def _initialize_with_blinds(
    *,
    seat_order: list[str],
    button_player_id: str,
    big_blind: Decimal,
    small_blind: Decimal,
) -> _ReplayState:
    seats = [_SeatState(player_id=pid, seat_index=i) for i, pid in enumerate(seat_order)]
    btn_idx = next(i for i, s in enumerate(seats) if s.player_id == button_player_id)

    n = len(seats)
    if n == 2:
        sb_idx = btn_idx
        bb_idx = (btn_idx + 1) % n
    else:
        sb_idx = (btn_idx + 1) % n
        bb_idx = (btn_idx + 2) % n

    seats[sb_idx].contribution_this_street = small_blind
    seats[bb_idx].contribution_this_street = big_blind

    return _ReplayState(
        seats=seats,
        button_seat=btn_idx,
        street="preflop",
        current_bet=big_blind,
        last_raise_size=big_blind,
        last_aggressor_seat=bb_idx,
    )


def _apply_action(state: _ReplayState, action: HistoryAction) -> None:
    seat = state.by_id(action.player_id)

    if action.kind == "fold":
        seat.is_active = False
    elif action.kind == "check":
        pass
    elif action.kind == "call":
        owed = state.current_bet - seat.contribution_this_street
        seat.contribution_this_street += owed
        if action.amount and action.amount < owed:
            seat.is_all_in = True
            seat.contribution_this_street = seat.contribution_this_street - owed + action.amount
    elif action.kind in ("bet", "raise"):
        new_total = seat.contribution_this_street + action.amount
        raise_increment = new_total - state.current_bet
        state.last_raise_size = max(raise_increment, state.last_raise_size)
        state.current_bet = new_total
        state.last_aggressor_seat = seat.seat_index
        seat.contribution_this_street = new_total
        for other in state.seats:
            if other.seat_index != seat.seat_index and other.can_act:
                other.has_acted_this_street = False
    else:
        raise ValueError(f"Unknown action kind: {action.kind!r}")

    seat.has_acted_this_street = True


def _street_closed(state: _ReplayState) -> bool:
    """A street closes when every player who can act has acted at least
    once this street AND matched the current bet (or is all-in)."""
    actors = state.active_can_act()
    if len(actors) <= 1:
        return True
    for s in actors:
        if not s.has_acted_this_street:
            return False
        if s.contribution_this_street < state.current_bet:
            return False
    return True


def _advance_street(state: _ReplayState) -> None:
    next_street = {
        "preflop": "flop",
        "flop": "turn",
        "turn": "river",
        "river": "river",
    }[state.street]
    state.street = next_street  # type: ignore[assignment]
    state.current_bet = Decimal("0")
    state.last_raise_size = Decimal("0")
    state.last_aggressor_seat = None
    for s in state.seats:
        s.contribution_this_street = Decimal("0")
        s.has_acted_this_street = False


def _snapshot(state: _ReplayState) -> SyntheticFSMSnapshot:
    actors = [
        s
        for s in state.active_can_act()
        if not s.has_acted_this_street or s.contribution_this_street < state.current_bet
    ]

    if not actors:
        return SyntheticFSMSnapshot(
            turn_pointer=None,
            action_order=(),
            legal_actions_per_player={},
            street=state.street,
        )

    start_idx = _first_to_act_seat(state)
    actors_ordered = sorted(
        actors,
        key=lambda s: (s.seat_index - start_idx) % len(state.seats),
    )

    turn_pointer = actors_ordered[0].player_id
    action_order = tuple(s.player_id for s in actors_ordered)

    legal: dict[str, frozenset[ActionKind]] = {}
    for s in state.active_can_act():
        legal[s.player_id] = _legal_actions_for(state, s)

    return SyntheticFSMSnapshot(
        turn_pointer=turn_pointer,
        action_order=action_order,
        legal_actions_per_player=legal,
        street=state.street,
    )


def _first_to_act_seat(state: _ReplayState) -> int:
    n = len(state.seats)
    if state.street == "preflop":
        if n == 2:
            return state.button_seat
        return (state.button_seat + 3) % n
    for offset in range(1, n + 1):
        idx = (state.button_seat + offset) % n
        if state.seats[idx].can_act:
            return idx
    return state.button_seat


def _legal_actions_for(state: _ReplayState, seat: _SeatState) -> frozenset[ActionKind]:
    owed = state.current_bet - seat.contribution_this_street
    if owed == 0:
        if state.current_bet == 0:
            return cast(frozenset[ActionKind], frozenset({"check", "bet"}))
        return cast(frozenset[ActionKind], frozenset({"check", "raise"}))
    return cast(frozenset[ActionKind], frozenset({"fold", "call", "raise"}))
