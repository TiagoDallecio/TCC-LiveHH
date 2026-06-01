import json
from decimal import Decimal

import pytest

from poker_vision.logic.models import ActionLogEntry, HandState, PlayerState


def make_player(seat: int = 0, stack: str = "1000") -> PlayerState:
    return PlayerState(seat=seat, stack=Decimal(stack))


def make_hand_state() -> HandState:
    return HandState(
        street="flop",
        board=["Ah", "Kd", "2s"],
        pot=Decimal("150"),
        current_bet_to_match=Decimal("50"),
        last_raiser="Villain_1",
        action_on_seat=2,
        players={
            "Hero": PlayerState(seat=0, stack=Decimal("950"), current_bet=Decimal("50")),
            "Villain_1": PlayerState(
                seat=1,
                stack=Decimal("800"),
                current_bet=Decimal("100"),
                has_acted_this_street=True,
            ),
        },
        action_log=[
            ActionLogEntry(player_id="Villain_1", action="bet", amount=Decimal("100"), street="flop"),
        ],
    )


def test_player_state_valid() -> None:
    p = make_player(seat=3, stack="500")
    assert p.seat == 3
    assert p.stack == Decimal("500")
    assert p.has_folded is False
    assert p.hole_cards == []


def test_player_negative_stack_raises() -> None:
    with pytest.raises(Exception):
        PlayerState(seat=0, stack=Decimal("-1"))


def test_player_negative_current_bet_raises() -> None:
    with pytest.raises(Exception):
        PlayerState(seat=0, stack=Decimal("500"), current_bet=Decimal("-5"))


def test_player_negative_seat_raises() -> None:
    with pytest.raises(Exception):
        PlayerState(seat=-1, stack=Decimal("500"))


def test_player_zero_stack_is_valid() -> None:
    p = PlayerState(seat=0, stack=Decimal("0"))
    assert p.stack == Decimal("0")


def test_hand_state_valid() -> None:
    hs = make_hand_state()
    assert hs.street == "flop"
    assert len(hs.board) == 3
    assert hs.pot == Decimal("150")
    assert "Hero" in hs.players


def test_hand_state_negative_pot_raises() -> None:
    with pytest.raises(Exception):
        HandState(street="preflop", pot=Decimal("-1"))


def test_hand_state_negative_bet_to_match_raises() -> None:
    with pytest.raises(Exception):
        HandState(street="preflop", current_bet_to_match=Decimal("-10"))


def test_hand_state_negative_action_on_seat_raises() -> None:
    with pytest.raises(Exception):
        HandState(street="preflop", action_on_seat=-1)


def test_hand_state_defaults() -> None:
    hs = HandState(street="preflop")
    assert hs.board == []
    assert hs.pot == Decimal("0")
    assert hs.players == {}
    assert hs.action_log == []
    assert hs.last_raiser is None
    assert hs.action_on_seat is None


def test_player_state_json_round_trip() -> None:
    original = PlayerState(
        seat=1,
        stack=Decimal("750"),
        current_bet=Decimal("25"),
        has_folded=False,
        has_acted_this_street=True,
        hole_cards=["As", "Kh"],
    )
    raw = original.model_dump_json()
    restored = PlayerState.model_validate_json(raw)
    assert restored.seat == original.seat
    assert restored.stack == original.stack
    assert restored.current_bet == original.current_bet
    assert restored.has_acted_this_street == original.has_acted_this_street
    assert restored.hole_cards == original.hole_cards


def test_hand_state_json_round_trip() -> None:
    original = make_hand_state()
    raw = original.model_dump_json()
    restored = HandState.model_validate_json(raw)
    assert restored.street == original.street
    assert restored.board == original.board
    assert restored.pot == original.pot
    assert restored.current_bet_to_match == original.current_bet_to_match
    assert restored.last_raiser == original.last_raiser
    assert restored.action_on_seat == original.action_on_seat
    assert set(restored.players.keys()) == set(original.players.keys())
    assert len(restored.action_log) == len(original.action_log)


def test_hand_state_json_is_valid_json() -> None:
    hs = make_hand_state()
    raw = hs.model_dump_json()
    data = json.loads(raw)
    assert "street" in data and "board" in data
    assert "pot" in data and "players" in data and "action_log" in data
