"""
Eventos visuais da pipeline
========================================

Cada evento representa uma observação de alta confiança emitida pela camada
de visão computacional. São imutáveis (frozen) e serializáveis em JSON.

Campos obrigatórios em todos os eventos:
    frame_idx   : índice do frame de origem (>= 0)
    confidence  : grau de confiança da detecção [0.0, 1.0]
"""

from __future__ import annotations

from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, Field

from poker_vision.inference.opponent_action_inferencer import ActionKind
from poker_vision.inference.opponent_action_inferencer import AnchorEvent as InferenceAnchorEvent


class VisualEvent(BaseModel):
    """Base imutável para todos os eventos visuais."""

    frame_idx: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Eventos de mão
# ---------------------------------------------------------------------------


class NewHandDetected(VisualEvent):
    """Detector identificou o início de uma nova mão."""

    dealer_seat: Optional[int] = None


class HoleCardsVisible(VisualEvent):
    """As duas hole cards do Hero ficaram visíveis."""

    cards: tuple[str, str]


class BoardCardsRevealed(VisualEvent):
    """Novas cartas comunitárias foram reveladas no board."""

    cards: list[str]


# ---------------------------------------------------------------------------
# Eventos de fichas
# ---------------------------------------------------------------------------


class PotChanged(VisualEvent):
    """Emitido pela fase 6 quando o montante de fichas na mesa muda."""

    old: Decimal = Field(ge=Decimal("0"))
    new: Decimal = Field(ge=Decimal("0"))
    delta: Decimal


class HeroBetDetected(VisualEvent):
    """Emitido quando o Hero faz uma jogada explícita."""

    action: ActionKind
    amount: Decimal = Field(ge=Decimal("0"))


class AnchorEvent(VisualEvent):
    """Wrapper visual para carregar âncoras para o motor de inferência."""

    anchor: InferenceAnchorEvent


class AnchorEventDetected(AnchorEvent):
    pass


class ChipsIntoBetZone(VisualEvent):
    seat: int = Field(ge=0)
    amount: Decimal = Field(ge=Decimal("0"))


class ChipsIntoPot(VisualEvent):
    amount: Decimal = Field(ge=Decimal("0"))


class PlayerStackChanged(VisualEvent):
    seat: int = Field(ge=0)
    new_stack: Decimal = Field(ge=Decimal("0"))


class ChipsAwarded(VisualEvent):
    seat: int = Field(ge=0)
    amount: Decimal = Field(ge=Decimal("0"))


# ---------------------------------------------------------------------------
# Eventos de cartas / posição
# ---------------------------------------------------------------------------


class CardsMucked(VisualEvent):
    """Cartas de um jogador foram descartadas (fold ou fim de mão)."""

    seat: int


class DealerButtonMoved(VisualEvent):
    """O botão do dealer foi movido para um novo assento."""

    new_seat: int
