from __future__ import annotations

import logging
from decimal import Decimal
from enum import Enum
from typing import Callable

from poker_vision.inference.opponent_action_inferencer import (
    ActionKind,
    OpponentActionInferencer,
    Street,
    TableContext,
)
from poker_vision.logic.events import (
    AnchorEvent,
    AnchorEventDetected,
    BoardCardsRevealed,
    CardsMucked,
    HeroBetDetected,
    HoleCardsVisible,
    NewHandDetected,
    PotChanged,
    VisualEvent,
)
from poker_vision.logic.invariants import check_invariants
from poker_vision.logic.models import ActionLogEntry
from poker_vision.logic.models import HandState as HandSnapshot
from poker_vision.logic.street_fsm import StreetAction, StreetFSM

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
    def __init__(
        self, ctx: TableContext, inferencer: OpponentActionInferencer, num_players: int = 6, hero_position: int = 0
    ) -> None:
        self.ctx = ctx
        self.inferencer = inferencer
        self.num_players = num_players
        self.hero_position = hero_position
        self.state: HandState = HandState.IDLE
        self.state_history: list[HandState] = [self.state]
        self.hand_state: HandSnapshot = HandSnapshot(street=ctx.current_street)
        self._street_fsm: StreetFSM | None = None

    def set_hand_state(self, hand_state: HandSnapshot) -> None:
        self.hand_state = hand_state

    def configure_hand_start(self, num_players: int, hero_position: int) -> None:
        self.num_players = num_players
        self.hero_position = hero_position

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

    def _street_from_state(self, state: HandState) -> Street:
        if state == HandState.PREFLOP:
            return "preflop"
        if state == HandState.FLOP:
            return "flop"
        if state == HandState.TURN:
            return "turn"
        return "river"

    def _is_street_state(self, state: HandState) -> bool:
        return state in (HandState.PREFLOP, HandState.FLOP, HandState.TURN, HandState.RIVER)

    def _street_fsm_enabled(self) -> bool:
        return self._street_fsm is not None

    def _build_street_fsm(self, is_preflop: bool) -> None:
        stacks = {player_id: player.stack for player_id, player in self.hand_state.players.items()}
        folded_players = {player_id for player_id, player in self.hand_state.players.items() if player.has_folded}
        all_in_players = {
            player_id
            for player_id, player in self.hand_state.players.items()
            if player.stack is not None and player.stack <= Decimal("0")
        }
        contributions = {player_id: player.current_bet for player_id, player in self.hand_state.players.items()}
        self._street_fsm = StreetFSM(
            players_in_seat_order=list(self.ctx.seat_order),
            button_seat=self.ctx.button_seat,
            stacks=stacks,
            is_preflop=is_preflop,
            initial_current_bet=self.ctx.current_bet,
            initial_contributions=contributions,
            folded_players=folded_players,
            all_in_players=all_in_players,
        )
        self.ctx.turn_pointer = self._street_fsm.turn_pointer or self.ctx.turn_pointer

    def _set_street_turn_pointer(self, player_id: str) -> None:
        if self._street_fsm is None:
            return
        self._street_fsm.set_turn_pointer(player_id)
        self.ctx.turn_pointer = self._street_fsm.turn_pointer or self.ctx.turn_pointer
        self.hand_state.action_on_seat = self._seat_for_player(self._street_fsm.turn_pointer)

    def _street_action_amount(self, action: ActionKind, amount: Decimal) -> Decimal:
        if action in ("fold", "check"):
            return Decimal("0")
        return Decimal(amount)

    def _apply_street_action(self, player_id: str, action: ActionKind, amount: Decimal) -> None:
        player = self.hand_state.players.get(player_id)
        if player is None:
            return

        action_amount = self._street_action_amount(action, amount)
        if player.stack is not None:
            action_amount = min(action_amount, player.stack)
        if self._street_fsm is not None:
            self._street_fsm.apply_inferred_actions(
                [StreetAction(player_id=player_id, action=action, amount=action_amount)]
            )
            self.ctx.turn_pointer = self._street_fsm.turn_pointer or self.ctx.turn_pointer
            self.ctx.current_bet = self._street_fsm.current_bet_to_match

        player.has_acted_this_street = True
        if action == "fold":
            player.has_folded = True
            return

        player.current_bet += action_amount
        if player.stack is not None:
            player.stack -= action_amount

        if player.current_bet > self.hand_state.current_bet_to_match:
            self.hand_state.current_bet_to_match = player.current_bet
            self.hand_state.last_raiser = player_id

        self.hand_state.action_log.append(
            ActionLogEntry(player_id=player_id, action=action, amount=action_amount, street=self.ctx.current_street)
        )

    def _process_inference_event(self, event: AnchorEvent | AnchorEventDetected) -> None:
        inferred_actions = self.inferencer.on_anchor(event.anchor, self.ctx)
        for act in inferred_actions:
            self._apply_street_action(act.player_id, act.action, act.amount)

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
        if isinstance(event, PotChanged):
            self.ctx.pot = event.new
            self.hand_state.pot = event.new
            return
        if isinstance(event, (AnchorEventDetected, AnchorEvent)):
            self._process_inference_event(event)
            return
        if self._is_showdown_event(event):
            self._transition(HandState.SHOWDOWN)
            return
        if isinstance(event, HeroBetDetected):
            self._apply_street_action(self.hand_state.hero_id, event.action, event.amount)
            return
        self._ignore(event)

    def _handle_flop(self, event: VisualEvent | str) -> None:
        if isinstance(event, BoardCardsRevealed):
            self._apply_board_state(event.cards)
            return
        if isinstance(event, PotChanged):
            self.ctx.pot = event.new
            self.hand_state.pot = event.new
            return
        if isinstance(event, (AnchorEventDetected, AnchorEvent)):
            self._process_inference_event(event)
            return
        if self._is_showdown_event(event):
            self._transition(HandState.SHOWDOWN)
            return
        if isinstance(event, HeroBetDetected):
            self._apply_street_action(self.hand_state.hero_id, event.action, event.amount)
            return
        self._ignore(event)

    def _handle_turn(self, event: VisualEvent | str) -> None:
        if isinstance(event, BoardCardsRevealed):
            self._apply_board_state(event.cards)
            return
        if isinstance(event, PotChanged):
            self.ctx.pot = event.new
            self.hand_state.pot = event.new
            return
        if isinstance(event, (AnchorEventDetected, AnchorEvent)):
            self._process_inference_event(event)
            return
        if self._is_showdown_event(event):
            self._transition(HandState.SHOWDOWN)
            return
        if isinstance(event, HeroBetDetected):
            self._apply_street_action(self.hand_state.hero_id, event.action, event.amount)
            return
        self._ignore(event)

    def _handle_river(self, event: VisualEvent | str) -> None:
        if isinstance(event, BoardCardsRevealed):
            self._apply_board_state(event.cards)
            return
        if isinstance(event, PotChanged):
            self.ctx.pot = event.new
            self.hand_state.pot = event.new
            return
        if isinstance(event, (AnchorEventDetected, AnchorEvent)):
            self._process_inference_event(event)
            return
        if self._is_showdown_event(event):
            self._transition(HandState.SHOWDOWN)
            return
        if isinstance(event, HeroBetDetected):
            self._apply_street_action(self.hand_state.hero_id, event.action, event.amount)
            return
        self._ignore(event)

    def _handle_showdown(self, event: VisualEvent | str) -> None:
        if isinstance(event, PotChanged) and event.new == 0:
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
        self.hand_state.board = list(board)
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
        if self._is_street_state(new_state):
            street = self._street_from_state(new_state)
            self.ctx.current_street = street
            self.hand_state.street = street
            self._build_street_fsm(is_preflop=(new_state == HandState.PREFLOP))
            if self._street_fsm is not None:
                self.hand_state.action_on_seat = self._seat_for_player(self._street_fsm.turn_pointer)
        else:
            self.hand_state.street = self.ctx.current_street
        self.hand_state.pot = self.ctx.pot
        self.hand_state.current_bet_to_match = self.ctx.current_bet
        check_invariants(self.hand_state)

    def _seat_for_player(self, player_id: str | None) -> int | None:
        if player_id is None:
            return None
        player = self.hand_state.players.get(player_id)
        if player is None:
            return None
        return player.seat

    def _ignore(self, event: object) -> None:
        logger.warning("Evento inválido para estado atual (%s): %r", self.state.value, event)
