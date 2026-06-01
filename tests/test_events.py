import json
from decimal import Decimal

import pytest

from poker_vision.logic.events import (
    BoardCardsRevealed,
    CardsMucked,
    ChipsAwarded,
    ChipsIntoBetZone,
    ChipsIntoPot,
    DealerButtonMoved,
    HoleCardsVisible,
    NewHandDetected,
    PlayerStackChanged,
    VisualEvent,
)

ALL_EVENT_INSTANCES = [
    NewHandDetected(frame_idx=0, confidence=0.95, dealer_seat=2),
    HoleCardsVisible(frame_idx=1, confidence=0.99, cards=("Ah", "Kd")),
    BoardCardsRevealed(frame_idx=2, confidence=0.98, cards=["Ah", "Kd", "2s"]),
    ChipsIntoBetZone(frame_idx=3, confidence=0.90, seat=1, amount=Decimal("50")),
    ChipsIntoPot(frame_idx=4, confidence=0.88, amount=Decimal("150")),
    CardsMucked(frame_idx=5, confidence=0.97, seat=3),
    DealerButtonMoved(frame_idx=6, confidence=0.99, new_seat=4),
    PlayerStackChanged(frame_idx=7, confidence=0.85, seat=2, new_stack=Decimal("900")),
    ChipsAwarded(frame_idx=8, confidence=0.92, seat=1, amount=Decimal("200")),
]


@pytest.mark.parametrize("event", ALL_EVENT_INSTANCES)
def test_every_event_has_required_fields(event: VisualEvent) -> None:
    assert hasattr(event, "frame_idx")
    assert hasattr(event, "confidence")
    assert isinstance(event.frame_idx, int)
    assert isinstance(event.confidence, float)


@pytest.mark.parametrize("event", ALL_EVENT_INSTANCES)
def test_frame_idx_is_non_negative(event: VisualEvent) -> None:
    assert event.frame_idx >= 0


@pytest.mark.parametrize("event", ALL_EVENT_INSTANCES)
def test_confidence_bounds(event: VisualEvent) -> None:
    assert 0.0 <= event.confidence <= 1.0


@pytest.mark.parametrize("event", ALL_EVENT_INSTANCES)
def test_json_serialization(event: VisualEvent) -> None:
    raw = event.model_dump_json()
    data = json.loads(raw)
    assert "frame_idx" in data
    assert "confidence" in data


def test_negative_frame_idx_raises() -> None:
    with pytest.raises(Exception):
        NewHandDetected(frame_idx=-1, confidence=0.9)


def test_confidence_above_one_raises() -> None:
    with pytest.raises(Exception):
        NewHandDetected(frame_idx=0, confidence=1.1)


def test_confidence_below_zero_raises() -> None:
    with pytest.raises(Exception):
        NewHandDetected(frame_idx=0, confidence=-0.1)


def test_negative_amount_chips_raises() -> None:
    with pytest.raises(Exception):
        ChipsIntoBetZone(frame_idx=0, confidence=0.9, seat=1, amount=Decimal("-10"))


def test_negative_stack_raises() -> None:
    with pytest.raises(Exception):
        PlayerStackChanged(frame_idx=0, confidence=0.9, seat=0, new_stack=Decimal("-1"))


def test_events_are_immutable() -> None:
    event = NewHandDetected(frame_idx=0, confidence=0.9)
    with pytest.raises(Exception):
        event.frame_idx = 99
