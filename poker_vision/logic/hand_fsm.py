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

logger = logging.getLogger(__name__)


class HandState(str, Enum):
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


_BOARD_SIZE_TO_STATE: dict[int, tuple[HandState, Street]] = {
    0: (HandState.PREFLOP, "preflop"),
    3: (HandState.FLOP, "flop"),
    4: (HandState.TURN, "turn"),
    5: (HandState.RIVER, "river"),
}


class HandFSM:
    def __init__(self, ctx: TableContext, inferencer: OpponentActionInferencer) -> None:
        self.ctx = ctx
        self.inferencer = inferencer
        self.state: HandState = HandState.IDLE
        self.state_history: list[HandState] = [self.state]

    def handle(self, event: VisualEvent | str) -> None:
        handlers: dict[HandState, Callable[[VisualEvent | str], None]] = {
            HandState.IDLE: self._handle_idle,
            HandState.POSTING_BLINDS: self._handle_posting_blinds,
            HandState.DEALING_HOLE_CARDS: self._handle_dealing_hole_cards,
            HandState.PREFLOP: self._handle_preflop,
            HandState.FLOP: self._handle_flop,
            HandState.TURN: self._handle_turn,
            HandState.RIVER: self._handle_river,
            HandState.SHOWDOWN: self._handle_showdown,
            HandState.SETTLING: self._handle_settling,
            HandState.ARCHIVED: self._handle_archived,
        }
        handlers[self.state](event)

    def handle_board_change(self, new_board: list[str]) -> None:
        self.handle(BoardCardsRevealed(frame_idx=0, confidence=1.0, cards=new_board))

    def _handle_idle(self, event: VisualEvent | str) -> None:
        if isinstance(event, NewHandDetected):
            self._transition(HandState.POSTING_BLINDS)
            return
        self._ignore(event)

    def _handle_posting_blinds(self, event: VisualEvent | str) -> None:
        if isinstance(event, HoleCardsVisible):
            self._transition(HandState.DEALING_HOLE_CARDS)
            self._transition(HandState.PREFLOP)
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
            self._transition(HandState.SHOWDOWN)
            return
        self._ignore(event)

    def _handle_flop(self, event: VisualEvent | str) -> None:
        if isinstance(event, BoardCardsRevealed):
            self._apply_board_state(event.cards)
            return
        if self._is_showdown_event(event):
            self._transition(HandState.SHOWDOWN)
            return
        self._ignore(event)

    def _handle_turn(self, event: VisualEvent | str) -> None:
        if isinstance(event, BoardCardsRevealed):
            self._apply_board_state(event.cards)
            return
        if self._is_showdown_event(event):
            self._transition(HandState.SHOWDOWN)
            return
        self._ignore(event)

    def _handle_river(self, event: VisualEvent | str) -> None:
        if isinstance(event, BoardCardsRevealed):
            self._apply_board_state(event.cards)
            return
        if self._is_showdown_event(event):
            self._transition(HandState.SHOWDOWN)
            return
        self._ignore(event)

    def _handle_showdown(self, event: VisualEvent | str) -> None:
        if isinstance(event, ChipsAwarded):
            self._transition(HandState.SETTLING)
            return
        self._ignore(event)

    def _handle_settling(self, event: VisualEvent | str) -> None:
        if isinstance(event, NewHandDetected):
            self._transition(HandState.ARCHIVED)
            return
        self._ignore(event)

    def _handle_archived(self, event: VisualEvent | str) -> None:
        if isinstance(event, NewHandDetected):
            self._transition(HandState.POSTING_BLINDS)
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

    def _transition(self, new_state: HandState) -> None:
        if self.state == new_state:
            return
        self.state = new_state
        self.state_history.append(new_state)

    def _ignore(self, event: object) -> None:
        logger.warning("Evento inválido para estado atual (%s): %r", self.state.value, event)
