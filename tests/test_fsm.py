from decimal import Decimal

from poker_vision.inference.opponent_action_inferencer import OpponentActionInferencer, TableContext
from poker_vision.logic.events import BoardCardsRevealed, HoleCardsVisible, NewHandDetected
from poker_vision.logic.hand_fsm import HandFSM, HandState
from poker_vision.logic.street_fsm import StreetFSM, StreetState


def _make_ctx() -> TableContext:
    return TableContext(
        num_players=6,
        button_seat=0,
        hero_seat=0,
        seat_order=["Hero", "Villain_1", "Villain_2", "Villain_3", "Villain_4", "Villain_5"],
        active_players=["Hero", "Villain_1", "Villain_2", "Villain_3", "Villain_4", "Villain_5"],
        current_street="preflop",
        current_bet=Decimal("0"),
        last_raise_size=Decimal("0"),
        turn_pointer="Villain_3",
        pot=Decimal("0"),
        contributions_this_street={},
        hero_id="Hero",
    )


def test_hand_fsm_transitions_street_order() -> None:
    fsm = HandFSM(_make_ctx(), OpponentActionInferencer())

    fsm.handle(NewHandDetected(frame_idx=1, confidence=0.99, dealer_seat=0))
    fsm.handle(HoleCardsVisible(frame_idx=2, confidence=0.99, cards=("Ah", "Kd")))
    fsm.handle(BoardCardsRevealed(frame_idx=3, confidence=0.99, cards=["2c", "7d", "Jh"]))
    fsm.handle(BoardCardsRevealed(frame_idx=4, confidence=0.99, cards=["2c", "7d", "Jh", "Qs"]))
    fsm.handle(BoardCardsRevealed(frame_idx=5, confidence=0.99, cards=["2c", "7d", "Jh", "Qs", "9h"]))
    fsm.handle("showdown")

    assert fsm.state_history == [
        HandState.IDLE,
        HandState.POSTING_BLINDS,
        HandState.DEALING_HOLE_CARDS,
        HandState.PREFLOP,
        HandState.FLOP,
        HandState.TURN,
        HandState.RIVER,
        HandState.SHOWDOWN,
    ]


def test_hand_fsm_invalid_event_is_ignored(caplog) -> None:
    fsm = HandFSM(_make_ctx(), OpponentActionInferencer())

    with caplog.at_level("WARNING"):
        fsm.handle(BoardCardsRevealed(frame_idx=1, confidence=0.8, cards=["As", "Kd", "Qc"]))

    assert fsm.state == HandState.IDLE
    assert "Evento inválido para estado atual" in caplog.text


def test_street_fsm_preflop_raise_folds_bb_call() -> None:
    players = ["BTN", "SB", "BB", "UTG", "HJ", "CO"]
    fsm = StreetFSM(
        players_in_seat_order=players,
        button_seat=0,
        stacks={p: Decimal("100") for p in players},
        is_preflop=True,
    )

    assert fsm.state == StreetState.AWAITING_ACTION
    assert fsm.action_on_player == "UTG"

    fsm.apply_action("UTG", Decimal("6"))
    fsm.apply_action("HJ", Decimal("0"))
    fsm.apply_action("CO", Decimal("0"))
    fsm.apply_action("BTN", Decimal("0"))
    fsm.apply_action("SB", Decimal("0"))
    fsm.apply_action("BB", Decimal("6"))

    assert [a.player_id for a in fsm.action_log] == ["UTG", "HJ", "CO", "BTN", "SB", "BB"]
    assert [a.action for a in fsm.action_log] == ["raise", "fold", "fold", "fold", "fold", "call"]
    assert [a.amount for a in fsm.action_log] == [
        Decimal("6"),
        Decimal("0"),
        Decimal("0"),
        Decimal("0"),
        Decimal("0"),
        Decimal("6"),
    ]
    assert fsm.state == StreetState.STREET_CLOSED


def test_street_fsm_identifies_all_in_with_different_amounts() -> None:
    players = ["P0", "P1", "P2"]
    fsm = StreetFSM(
        players_in_seat_order=players,
        button_seat=0,
        stacks={"P0": Decimal("100"), "P1": Decimal("60"), "P2": Decimal("200")},
        is_preflop=True,
    )

    fsm.apply_action("P0", Decimal("100"))
    fsm.apply_action("P1", Decimal("60"))
    fsm.apply_action("P2", Decimal("0"))

    assert fsm.action_log[0].action == "all_in"
    assert fsm.action_log[0].amount == Decimal("100")
    assert fsm.action_log[1].action == "all_in"
    assert fsm.action_log[1].amount == Decimal("60")
