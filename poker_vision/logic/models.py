"""
Modelos de estado da mão (Issue #18)
======================================

PlayerState  — snapshot do estado de um jogador em determinado momento.
HandState    — snapshot completo da mesa em determinado momento.
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
