"""
Eventos visuais da pipeline (Issue #17)
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


class ChipsIntoBetZone(VisualEvent):
    """Fichas de um jogador foram movidas para a zona de aposta."""

    seat: int
    amount: Decimal = Field(ge=Decimal("0"))


class ChipsIntoPot(VisualEvent):
    """Fichas da zona de aposta foram recolhidas ao pote central."""

    amount: Decimal = Field(ge=Decimal("0"))


class ChipsAwarded(VisualEvent):
    """Fichas do pote foram entregues ao vencedor."""

    seat: int
    amount: Decimal = Field(ge=Decimal("0"))


class PlayerStackChanged(VisualEvent):
    """Stack de um jogador mudou de valor."""

    seat: int
    new_stack: Decimal = Field(ge=Decimal("0"))


# ---------------------------------------------------------------------------
# Eventos de cartas / posição
# ---------------------------------------------------------------------------


class CardsMucked(VisualEvent):
    """Cartas de um jogador foram descartadas (fold ou fim de mão)."""

    seat: int


class DealerButtonMoved(VisualEvent):
    """O botão do dealer foi movido para um novo assento."""

    new_seat: int
