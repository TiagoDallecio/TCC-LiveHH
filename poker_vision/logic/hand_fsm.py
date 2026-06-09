from __future__ import annotations

import logging
from enum import Enum
from typing import Callable

from poker_vision.inference.opponent_action_inferencer import (
    OpponentActionInferencer,
    Street,
    TableContext,
)
from poker_vision.logic.events import (
    BoardCardsRevealed,
    CardsMucked,
    ChipsAwarded,
    HoleCardsVisible,
    NewHandDetected,
    VisualEvent,
)
from poker_vision.logic.invariants import check_invariants
from poker_vision.logic.models import HandState as HandSnapshot

logger = logging.getLogger(__name__)


class HandPhase(str, Enum):
    IDLE = "idle"
    POSTING_BLINDS = "posting_blinds"
    DEALING_HOLE_CARDS = "dealing_hole_cards"
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"
    SHOWDOWN = "showdown"
    SETTLING = "settling"
    ARCHIVED = "archived"


_BOARD_SIZE_TO_STATE: dict[int, tuple[HandPhase, Street]] = {
    0: (HandPhase.PREFLOP, "preflop"),
    3: (HandPhase.FLOP, "flop"),
    4: (HandPhase.TURN, "turn"),
    5: (HandPhase.RIVER, "river"),
}


class HandFSM:
    def __init__(self, ctx: TableContext, inferencer: OpponentActionInferencer) -> None:
        self.ctx = ctx
        self.inferencer = inferencer
        self.state: HandPhase = HandPhase.IDLE
        self.state_history: list[HandPhase] = [self.state]
        self.hand_state: HandSnapshot = HandSnapshot(street=ctx.current_street)

    def set_hand_state(self, hand_state: HandSnapshot) -> None:
        self.hand_state = hand_state

    def handle(self, event: VisualEvent | str) -> None:
        handlers: dict[HandPhase, Callable[[VisualEvent | str], None]] = {
            HandPhase.IDLE: self._handle_idle,
            HandPhase.POSTING_BLINDS: self._handle_posting_blinds,
            HandPhase.DEALING_HOLE_CARDS: self._handle_dealing_hole_cards,
            HandPhase.PREFLOP: self._handle_preflop,
            HandPhase.FLOP: self._handle_flop,
            HandPhase.TURN: self._handle_turn,
            HandPhase.RIVER: self._handle_river,
            HandPhase.SHOWDOWN: self._handle_showdown,
            HandPhase.SETTLING: self._handle_settling,
            HandPhase.ARCHIVED: self._handle_archived,
        }
        handlers[self.state](event)

    def handle_board_change(self, new_board: list[str]) -> None:
        self.handle(BoardCardsRevealed(frame_idx=0, confidence=1.0, cards=new_board))

    def _handle_idle(self, event: VisualEvent | str) -> None:
        if isinstance(event, NewHandDetected):
            self._transition(HandPhase.POSTING_BLINDS)
            return
        self._ignore(event)

    def _handle_posting_blinds(self, event: VisualEvent | str) -> None:
        if isinstance(event, HoleCardsVisible):
            self._transition(HandPhase.DEALING_HOLE_CARDS)
            self._transition(HandPhase.PREFLOP)
            return
        self._ignore(event)

    def _handle_dealing_hole_cards(self, event: VisualEvent | str) -> None:
        if isinstance(event, BoardCardsRevealed) and len(event.cards) == 0:
            self._apply_board_state(event.cards)
            return
        self._ignore(event)

    def _handle_preflop(self, event: VisualEvent | str) -> None:
        if isinstance(event, BoardCardsRevealed):
            self._apply_board_state(event.cards)
            return
        if self._is_showdown_event(event):
            self._transition(HandPhase.SHOWDOWN)
            return
        self._ignore(event)

    def _handle_flop(self, event: VisualEvent | str) -> None:
        if isinstance(event, BoardCardsRevealed):
            self._apply_board_state(event.cards)
            return
        if self._is_showdown_event(event):
            self._transition(HandPhase.SHOWDOWN)
            return
        self._ignore(event)

    def _handle_turn(self, event: VisualEvent | str) -> None:
        if isinstance(event, BoardCardsRevealed):
            self._apply_board_state(event.cards)
            return
        if self._is_showdown_event(event):
            self._transition(HandPhase.SHOWDOWN)
            return
        self._ignore(event)

    def _handle_river(self, event: VisualEvent | str) -> None:
        if isinstance(event, BoardCardsRevealed):
            self._apply_board_state(event.cards)
            return
        if self._is_showdown_event(event):
            self._transition(HandPhase.SHOWDOWN)
            return
        self._ignore(event)

    def _handle_showdown(self, event: VisualEvent | str) -> None:
        if isinstance(event, ChipsAwarded):
            self._transition(HandPhase.SETTLING)
            return
        self._ignore(event)

    def _handle_settling(self, event: VisualEvent | str) -> None:
        if isinstance(event, NewHandDetected):
            self._transition(HandPhase.ARCHIVED)
            return
        self._ignore(event)

    def _handle_archived(self, event: VisualEvent | str) -> None:
        if isinstance(event, NewHandDetected):
            self._transition(HandPhase.POSTING_BLINDS)
            return
        self._ignore(event)

    def _is_showdown_event(self, event: VisualEvent | str) -> bool:
        return isinstance(event, CardsMucked) or event == "showdown"

    def _apply_board_state(self, board: list[str]) -> None:
        mapping = _BOARD_SIZE_TO_STATE.get(len(board))
        if mapping is None:
            self._ignore(board)
            return
        next_state, next_street = mapping
        if next_state == self.state:
            return
        self.ctx.current_street = next_street
        self._transition(next_state)

    def _transition(self, new_state: HandPhase) -> None:
        if self.state == new_state:
            return
        self.state = new_state
        self.state_history.append(new_state)
        self.hand_state.street = self.ctx.current_street
        self.hand_state.pot = self.ctx.pot
        self.hand_state.current_bet_to_match = self.ctx.current_bet
        check_invariants(self.hand_state)

    def _ignore(self, event: object) -> None:
        logger.warning("Evento inválido para estado atual (%s): %r", self.state.value, event)
