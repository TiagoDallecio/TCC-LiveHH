from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from poker_vision.inference.opponent_action_inferencer import ActionKind


class StreetState(str, Enum):
    STREET_OPEN = "street_open"
    AWAITING_ACTION = "awaiting_action"
    VALIDATING_ACTION = "validating_action"
    APPLYING_ACTION = "applying_action"
    STREET_CLOSED = "street_closed"


@dataclass(frozen=True)
class StreetAction:
    player_id: str
    action: ActionKind
    amount: Decimal


class StreetFSM:
    def __init__(
        self,
        players_in_seat_order: list[str],
        button_seat: int,
        stacks: dict[str, Decimal],
        is_preflop: bool,
    ) -> None:
        if not players_in_seat_order:
            raise ValueError("players_in_seat_order vazio")
        self.players_in_seat_order = players_in_seat_order
        self.button_seat = button_seat
        self.stacks = {player: Decimal(value) for player, value in stacks.items()}
        self.is_preflop = is_preflop
        self.state: StreetState = StreetState.STREET_OPEN
        self.current_bet_to_match: Decimal = Decimal("0")
        self.contributions: dict[str, Decimal] = {player: Decimal("0") for player in players_in_seat_order}
        self.folded: set[str] = set()
        self.all_in: set[str] = set()
        self.action_log: list[StreetAction] = []
        self.action_on_player: str | None = None
        self._players_to_act: list[str] = []
        self._open_street()

    def _open_street(self) -> None:
        self.state = StreetState.AWAITING_ACTION
        first_to_act = self._first_to_act()
        self._players_to_act = self._order_from(first_to_act)
        self.action_on_player = self._players_to_act[0] if self._players_to_act else None

    def _first_to_act(self) -> str:
        n = len(self.players_in_seat_order)
        if self.is_preflop:
            index = (self.button_seat + 3) % n
        else:
            index = (self.button_seat + 1) % n
        return self.players_in_seat_order[index]

    def _order_from(self, player_id: str) -> list[str]:
        n = len(self.players_in_seat_order)
        start = self.players_in_seat_order.index(player_id)
        out: list[str] = []
        for step in range(n):
            candidate = self.players_in_seat_order[(start + step) % n]
            if candidate in self.folded or candidate in self.all_in:
                continue
            out.append(candidate)
        return out

    def classify_action(self, player_id: str, amount: Decimal) -> ActionKind:
        if player_id in self.folded:
            raise ValueError("Jogador já foldou")
        if player_id in self.all_in:
            raise ValueError("Jogador já está all-in")
        if amount < 0:
            raise ValueError("amount negativo")
        stack = self.stacks[player_id]
        to_call = self.current_bet_to_match - self.contributions[player_id]

        if amount == 0:
            if to_call > 0:
                return "fold"
            return "check"

        is_all_in_amount = amount == stack

        if to_call > 0:
            if amount < to_call:
                if is_all_in_amount:
                    return "all_in"
                raise ValueError("Ação inválida: valor menor que call sem all-in")
            if amount == to_call:
                if is_all_in_amount:
                    return "all_in"
                return "call"
            if is_all_in_amount:
                return "all_in"
            return "raise"

        if is_all_in_amount:
            return "all_in"
        if self.is_preflop:
            return "raise"
        return "bet"

    def apply_action(self, player_id: str, amount: Decimal) -> StreetAction:
        if self.state not in (StreetState.AWAITING_ACTION, StreetState.VALIDATING_ACTION, StreetState.APPLYING_ACTION):
            raise ValueError("Street fechada")
        if self.action_on_player != player_id:
            raise ValueError("Não é a vez do jogador")

        self.state = StreetState.VALIDATING_ACTION
        action_kind = self.classify_action(player_id, amount)

        self.state = StreetState.APPLYING_ACTION
        applied_amount = Decimal("0")

        if action_kind == "fold":
            self.folded.add(player_id)
        else:
            applied_amount = amount
            self.stacks[player_id] = self.stacks[player_id] - applied_amount
            self.contributions[player_id] = self.contributions[player_id] + applied_amount
            if action_kind in ("bet", "raise", "all_in") and self.contributions[player_id] > self.current_bet_to_match:
                self.current_bet_to_match = self.contributions[player_id]
            if action_kind == "all_in":
                self.all_in.add(player_id)

        action = StreetAction(player_id=player_id, action=action_kind, amount=applied_amount)
        self.action_log.append(action)
        self._advance_after_action(player_id, action_kind)
        return action

    def _advance_after_action(self, actor: str, action_kind: ActionKind) -> None:
        if actor in self._players_to_act:
            self._players_to_act.remove(actor)

        is_aggressive = action_kind in ("bet", "raise")
        if is_aggressive:
            reopened = self._order_from(self._next_seat_player(actor))
            self._players_to_act = [p for p in reopened if p != actor]

        if not self._players_to_act:
            self.state = StreetState.STREET_CLOSED
            self.action_on_player = None
            return

        self.state = StreetState.AWAITING_ACTION
        self.action_on_player = self._players_to_act[0]

    def _next_seat_player(self, player_id: str) -> str:
        n = len(self.players_in_seat_order)
        i = self.players_in_seat_order.index(player_id)
        for step in range(1, n + 1):
            candidate = self.players_in_seat_order[(i + step) % n]
            if candidate in self.folded or candidate in self.all_in:
                continue
            return candidate
        return player_id
