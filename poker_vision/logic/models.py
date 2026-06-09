"""
Modelos de estado da mão (Issue #18)
======================================

PlayerState  — snapshot do estado de um jogador em determinado momento.
HandPhase    — snapshot completo da mesa em determinado momento.
ActionLogEntry — registro imutável de uma ação realizada durante a mão.

Todos os modelos são validados pelo Pydantic (v2) e suportam round-trip JSON.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, Field

from poker_vision.inference.opponent_action_inferencer import ActionKind, Street

# ---------------------------------------------------------------------------
# Registro de ação (entrada do action_log)
# ---------------------------------------------------------------------------


class ActionLogEntry(BaseModel):
    """Registro imutável de uma única ação realizada durante a mão."""

    player_id: str
    action: ActionKind
    amount: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    street: Street

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Estado de um jogador
# ---------------------------------------------------------------------------


class PlayerState(BaseModel):
    """Snapshot do estado de um jogador em determinado momento da mão."""

    seat: int = Field(ge=0)
    stack: Decimal = Field(ge=Decimal("0"))
    current_bet: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    has_folded: bool = False
    has_acted_this_street: bool = False
    hole_cards: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Estado completo da mão
# ---------------------------------------------------------------------------


class HandQuality(BaseModel):
    needs_review: bool = False


class HandState(BaseModel):
    """Snapshot completo do estado da mesa em determinado momento."""

    street: Street
    board: list[str] = Field(default_factory=list)
    pot: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    current_bet_to_match: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    last_raiser: Optional[str] = None
    action_on_seat: Optional[int] = Field(default=None, ge=0)
    players: dict[str, PlayerState] = Field(default_factory=dict)
    action_log: list[ActionLogEntry] = Field(default_factory=list)
    quality: HandQuality = Field(default_factory=HandQuality)

    @property
    def current_bet(self) -> Decimal:
        return self.current_bet_to_match

    @property
    def big_blind(self) -> Decimal:
        return Decimal("1")

    @property
    def last_raise_size(self) -> Decimal:
        return Decimal("0")

    @property
    def turn_pointer(self) -> Optional[str]:
        if self.action_on_seat is None:
            return None
        for pid, player in self.players.items():
            if getattr(player, "seat", None) == self.action_on_seat:
                return pid
        return None

    @property
    def action_order(self) -> tuple[str, ...]:
        return ()

    def legal_actions_for(self, player_id: str) -> frozenset[str]:
        return frozenset({"fold", "check", "call", "bet", "raise", "all_in"})
