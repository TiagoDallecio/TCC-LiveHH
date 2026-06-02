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
        stacks: dict[str, Decimal | None],
        is_preflop: bool,
        initial_current_bet: Decimal = Decimal("0"),
        initial_contributions: dict[str, Decimal] | None = None,
        folded_players: set[str] | None = None,
        all_in_players: set[str] | None = None,
    ) -> None:
        if not players_in_seat_order:
            raise ValueError("players_in_seat_order vazio")
        self.players_in_seat_order = players_in_seat_order
        self.button_seat = button_seat
        self.stacks = {player: (Decimal(value) if value is not None else None) for player, value in stacks.items()}
        self.is_preflop = is_preflop
        self.state: StreetState = StreetState.STREET_OPEN
        self.current_bet_to_match: Decimal = Decimal(initial_current_bet)
        self.contributions: dict[str, Decimal] = {player: Decimal("0") for player in players_in_seat_order}
        if initial_contributions is not None:
            for player_id, amount in initial_contributions.items():
                if player_id in self.contributions:
                    self.contributions[player_id] = Decimal(amount)
        self.folded: set[str] = set(folded_players or set())
        self.all_in: set[str] = set(all_in_players or set())
        self.action_log: list[StreetAction] = []
        self.turn_pointer: str | None = None
        self.action_on_player: str | None = None
        self._players_to_act: list[str] = []
        self._open_street()

    def _open_street(self) -> None:
        self.state = StreetState.AWAITING_ACTION
        first_to_act = self._first_to_act()
        self._players_to_act = self._order_from(first_to_act)
        self.turn_pointer = self._players_to_act[0] if self._players_to_act else None
        self.action_on_player = self.turn_pointer

    def set_turn_pointer(self, player_id: str) -> None:
        if self.state == StreetState.STREET_CLOSED:
            return
        if player_id not in self.players_in_seat_order:
            raise ValueError("player_id desconhecido")

        ordered = self._order_from(player_id)
        if not ordered:
            self._players_to_act = []
            self.turn_pointer = None
            self.action_on_player = None
            self.state = StreetState.STREET_CLOSED
            return

        self._players_to_act = ordered
        self.turn_pointer = ordered[0]
        self.action_on_player = self.turn_pointer
        self.state = StreetState.AWAITING_ACTION

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

    def apply_action(self, player_id: str, amount: Decimal) -> None:
        amount_value = Decimal(amount)
        if amount_value < 0:
            raise ValueError("amount deve ser >= 0")
        if self.turn_pointer is not None and player_id != self.turn_pointer:
            raise ValueError("Ação fora de ordem")

        if player_id not in self.contributions:
            raise ValueError("player_id desconhecido")

        stack = self.stacks.get(player_id)
        contribution_after = self.contributions[player_id] + amount_value

        if amount_value == 0:
            action: ActionKind = "check" if self.current_bet_to_match == self.contributions[player_id] else "fold"
        elif stack is not None and amount_value >= stack:
            action = "all_in"
        elif contribution_after > self.current_bet_to_match:
            if self.is_preflop:
                action = "raise"
            else:
                action = "bet" if self.current_bet_to_match == 0 else "raise"
        elif contribution_after == self.current_bet_to_match:
            action = "call"
        else:
            raise ValueError("amount inválido para estado atual")

        self.apply_inferred_actions([StreetAction(player_id=player_id, action=action, amount=amount_value)])

    def apply_inferred_actions(self, actions: list[StreetAction]) -> None:
        """Avança o estado da street processando em lote as ações deduzidas pelo Inferencer."""
        if self.state not in (StreetState.AWAITING_ACTION, StreetState.VALIDATING_ACTION, StreetState.APPLYING_ACTION):
            raise ValueError("Street fechada")

        self.state = StreetState.APPLYING_ACTION

        for act in actions:
            if self.turn_pointer is not None and act.player_id != self.turn_pointer:
                raise ValueError("Ação inferida fora de ordem")
            if act.player_id not in self.contributions:
                raise ValueError("player_id desconhecido")

            if act.action == "fold":
                self.folded.add(act.player_id)
                act_amount = Decimal("0")
            else:
                act_amount = Decimal(act.amount)
                self.contributions[act.player_id] += act_amount
                if self.contributions[act.player_id] > self.current_bet_to_match:
                    self.current_bet_to_match = self.contributions[act.player_id]
                if act.action == "all_in":
                    self.all_in.add(act.player_id)

            stack = self.stacks.get(act.player_id)
            if stack is not None:
                self.stacks[act.player_id] = stack - act_amount

            self.action_log.append(StreetAction(player_id=act.player_id, action=act.action, amount=act_amount))
            self._advance_after_action(act.player_id, act.action)

    def _advance_after_action(self, actor: str, action_kind: ActionKind) -> None:
        if actor in self._players_to_act:
            self._players_to_act.remove(actor)

        is_aggressive = action_kind in ("bet", "raise")
        if is_aggressive:
            reopened = self._order_from(self._next_seat_player(actor))
            self._players_to_act = [p for p in reopened if p != actor]

        if not self._players_to_act:
            self.state = StreetState.STREET_CLOSED
            self.turn_pointer = None
            self.action_on_player = None
            return

        self.state = StreetState.AWAITING_ACTION
        self.turn_pointer = self._players_to_act[0]
        self.action_on_player = self.turn_pointer

    def _next_seat_player(self, player_id: str) -> str:
        n = len(self.players_in_seat_order)
        i = self.players_in_seat_order.index(player_id)
        for step in range(1, n + 1):
            candidate = self.players_in_seat_order[(i + step) % n]
            if candidate in self.folded or candidate in self.all_in:
                continue
            return candidate
        return player_id
