from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import yaml

from poker_vision.inference.opponent_action_inferencer import (
    AnchorEvent as InferenceAnchorEvent,
)
from poker_vision.inference.opponent_action_inferencer import (
    OpponentActionInferencer,
    Street,
    TableContext,
)
from poker_vision.logic.events import (
    AnchorEvent,
    BoardCardsRevealed,
    CardsMucked,
    ChipsAwarded,
    DealerButtonMoved,
    HeroBetDetected,
    HoleCardsVisible,
    NewHandDetected,
    PotChanged,
    VisualEvent,
)
from poker_vision.logic.hand_fsm import HandFSM
from poker_vision.logic.invariants import InvariantViolationError, check_invariants
from poker_vision.logic.models import HandState as HandSnapshot
from poker_vision.logic.models import PlayerState

logger = logging.getLogger(__name__)


@dataclass
class ReplayResult:
    scenario_name: str
    transition_logs: list[str]
    final_state: str
    needs_review: bool


def run_replay_scenario(scenario_path: str | Path) -> ReplayResult:
    path = Path(scenario_path)
    scenario = _load_scenario(path)
    hand_state = _build_hand_state(scenario)
    ctx = _build_context(scenario, hand_state)
    fsm = HandFSM(ctx, OpponentActionInferencer())
    fsm.configure_hand_start(num_players=ctx.num_players, hero_position=ctx.hero_seat)
    fsm.set_hand_state(hand_state)
    check_invariants(fsm.hand_state)

    transition_logs: list[str] = []
    raw_events = scenario.get("events", [])
    if not isinstance(raw_events, list):
        raise ValueError("Scenario field 'events' must be a list")

    for index, raw_event in enumerate(raw_events, start=1):
        history_before = list(fsm.state_history)
        event_name = _apply_replay_event(fsm, ctx, raw_event, index)
        history_after = fsm.state_history
        if len(history_after) > len(history_before):
            previous_state = history_before[-1]
            for transitioned_state in history_after[len(history_before) :]:
                log_line = f"{index:03d} {event_name}: {previous_state.value} -> {transitioned_state.value}"
                transition_logs.append(log_line)
                logger.info(log_line)
                previous_state = transitioned_state
        check_invariants(fsm.hand_state)

    check_invariants(fsm.hand_state)
    if fsm.hand_state.quality.needs_review:
        raise InvariantViolationError("Scenario finished with quality.needs_review=True")

    return ReplayResult(
        scenario_name=str(scenario.get("name", path.stem)),
        transition_logs=transition_logs,
        final_state=fsm.state.value,
        needs_review=fsm.hand_state.quality.needs_review,
    )


def _load_scenario(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Scenario root must be a mapping")
    return loaded


def _build_hand_state(scenario: dict[str, Any]) -> HandSnapshot:
    initial_state = scenario.get("initial_state", {})
    if not isinstance(initial_state, dict):
        raise ValueError("Scenario field 'initial_state' must be a mapping")

    players_raw = initial_state.get("players", {})
    if not isinstance(players_raw, dict):
        raise ValueError("Scenario field 'initial_state.players' must be a mapping")

    players: dict[str, PlayerState] = {}
    for player_id, player_payload in players_raw.items():
        if not isinstance(player_payload, dict):
            raise ValueError(f"Player payload for '{player_id}' must be a mapping")
        players[player_id] = PlayerState(
            seat=int(player_payload["seat"]),
            stack=_to_decimal(player_payload["stack"]) if player_payload.get("stack") is not None else None,
            current_bet=_to_decimal(player_payload.get("current_bet", 0)),
            has_folded=bool(player_payload.get("has_folded", False)),
            has_acted_this_street=bool(player_payload.get("has_acted_this_street", False)),
            hole_cards=list(player_payload.get("hole_cards", [])),
        )

    street_value = str(initial_state.get("street", "preflop"))
    if street_value not in {"preflop", "flop", "turn", "river"}:
        raise ValueError(f"Unsupported street in initial_state: {street_value}")
    street = cast(Street, street_value)

    return HandSnapshot(
        street=street,
        board=list(initial_state.get("board", [])),
        pot=_to_decimal(initial_state.get("pot", 0)),
        current_bet_to_match=_to_decimal(initial_state.get("current_bet_to_match", 0)),
        last_raiser=initial_state.get("last_raiser"),
        action_on_seat=initial_state.get("action_on_seat"),
        players=players,
        hero_id=str(initial_state.get("hero_id", "Hero")),
    )


def _build_context(scenario: dict[str, Any], hand_state: HandSnapshot) -> TableContext:
    table = scenario.get("table", {})
    if not isinstance(table, dict):
        raise ValueError("Scenario field 'table' must be a mapping")

    hero_id = str(table.get("hero_id", hand_state.hero_id))
    hand_state.hero_id = hero_id

    seat_order_raw = table.get("seat_order", list(hand_state.players.keys()))
    if not isinstance(seat_order_raw, list):
        raise ValueError("Scenario field 'table.seat_order' must be a list")
    seat_order = [str(player_id) for player_id in seat_order_raw]

    if seat_order:
        default_turn_pointer = seat_order[min(3, len(seat_order) - 1)]
    else:
        default_turn_pointer = hero_id

    active_players_raw = table.get("active_players")
    if active_players_raw is None:
        active_players = [player_id for player_id, player in hand_state.players.items() if not player.has_folded]
    else:
        if not isinstance(active_players_raw, list):
            raise ValueError("Scenario field 'table.active_players' must be a list")
        active_players = [str(player_id) for player_id in active_players_raw]

    return TableContext(
        num_players=int(table.get("num_players", len(seat_order))),
        button_seat=int(table.get("button_seat", 0)),
        hero_seat=int(table.get("hero_seat", 0)),
        seat_order=seat_order,
        active_players=active_players,
        current_street=hand_state.street,
        current_bet=hand_state.current_bet_to_match,
        last_raise_size=_to_decimal(table.get("last_raise_size", 0)),
        turn_pointer=str(table.get("turn_pointer", default_turn_pointer)),
        pot=hand_state.pot,
        contributions_this_street={
            player_id: player.current_bet for player_id, player in hand_state.players.items() if player.current_bet > 0
        },
        hero_id=hero_id,
    )


def _apply_replay_event(fsm: HandFSM, ctx: TableContext, raw_event: Any, index: int) -> str:
    if isinstance(raw_event, str):
        event_type = raw_event
        payload: dict[str, Any] = {"type": raw_event}
    elif isinstance(raw_event, dict):
        event_type = str(raw_event.get("type", ""))
        payload = raw_event
    else:
        raise ValueError(f"Event entry at index {index} must be string or mapping")

    if event_type == "SyntheticStreetReset":
        _apply_street_reset(fsm, ctx, payload)
        return event_type

    if event_type == "SyntheticAction":
        _apply_synthetic_action(fsm, ctx, payload)
        return event_type

    if event_type == "showdown":
        fsm.handle("showdown")
        return event_type

    if event_type in ("AnchorEvent", "AnchorEventDetected") and "turn_pointer" in payload:
        ctx.turn_pointer = str(payload["turn_pointer"])
        fsm._set_street_turn_pointer(ctx.turn_pointer)
        fsm.hand_state.action_on_seat = _seat_for_player_id(fsm.hand_state, ctx.turn_pointer)

    visual_event = _build_visual_event(event_type, payload, index)
    _apply_visual_side_effects(fsm, ctx, visual_event)
    fsm.handle(visual_event)
    _sync_current_bet(fsm.hand_state, ctx)
    return event_type


def _build_visual_event(event_type: str, payload: dict[str, Any], index: int) -> VisualEvent:
    frame_idx = int(payload.get("frame_idx", index))
    confidence = float(payload.get("confidence", 1.0))

    if event_type == "NewHandDetected":
        return NewHandDetected(frame_idx=frame_idx, confidence=confidence, dealer_seat=payload.get("dealer_seat"))
    if event_type == "HoleCardsVisible":
        cards = payload.get("cards")
        if not isinstance(cards, list) or len(cards) != 2:
            raise ValueError("HoleCardsVisible requires a two-card list in 'cards'")
        return HoleCardsVisible(frame_idx=frame_idx, confidence=confidence, cards=(str(cards[0]), str(cards[1])))
    if event_type == "BoardCardsRevealed":
        cards = payload.get("cards", [])
        if not isinstance(cards, list):
            raise ValueError("BoardCardsRevealed requires list field 'cards'")
        return BoardCardsRevealed(frame_idx=frame_idx, confidence=confidence, cards=[str(card) for card in cards])
    if event_type == "CardsMucked":
        return CardsMucked(frame_idx=frame_idx, confidence=confidence, seat=int(payload["seat"]))
    if event_type == "PotChanged":
        return PotChanged(
            frame_idx=frame_idx,
            confidence=confidence,
            old=_to_decimal(payload["old"]),
            new=_to_decimal(payload["new"]),
            delta=_to_decimal(payload["delta"]),
        )
    if event_type == "HeroBetDetected":
        return HeroBetDetected(
            frame_idx=frame_idx, confidence=confidence, action=payload["action"], amount=_to_decimal(payload["amount"])
        )
    if event_type in ("AnchorEvent", "AnchorEventDetected"):
        anchor_data_raw = payload.get("anchor_data", payload.get("anchor"))
        if not isinstance(anchor_data_raw, dict):
            raise ValueError("AnchorEvent requires mapping field 'anchor_data' or 'anchor'")
        anchor_data = dict(anchor_data_raw)
        anchor_data["pot_before"] = _to_decimal(anchor_data["pot_before"])
        anchor_data["pot_after"] = _to_decimal(anchor_data["pot_after"])
        if "timestamp" in anchor_data:
            anchor_data["timestamp"] = float(anchor_data["timestamp"])
        if "board" in anchor_data:
            board = anchor_data["board"]
            if not isinstance(board, (list, tuple)):
                raise ValueError("AnchorEvent field 'board' must be a list or tuple")
            anchor_data["board"] = tuple(str(card) for card in board)
        hero_action = anchor_data.get("hero_action")
        if isinstance(hero_action, dict) and "amount" in hero_action:
            anchor_data["hero_action"] = {**hero_action, "amount": _to_decimal(hero_action["amount"])}
        anchor = InferenceAnchorEvent(**anchor_data)
        return AnchorEvent(frame_idx=frame_idx, confidence=confidence, anchor=anchor)
    if event_type == "DealerButtonMoved":
        return DealerButtonMoved(frame_idx=frame_idx, confidence=confidence, new_seat=int(payload["new_seat"]))
    if event_type == "ChipsAwarded":
        return ChipsAwarded(
            frame_idx=frame_idx,
            confidence=confidence,
            seat=int(payload["seat"]),
            amount=_to_decimal(payload["amount"]),
        )

    raise ValueError(f"Unsupported visual event type: {event_type}")


def _apply_synthetic_action(fsm: HandFSM, ctx: TableContext, payload: dict[str, Any]) -> None:
    player_id = str(payload["player_id"])
    action = cast(str, payload["action"])
    amount = _to_decimal(payload.get("amount", 0))

    player = fsm.hand_state.players.get(player_id)
    if player is not None:
        player.has_acted_this_street = True
        if action == "fold":
            player.has_folded = True
        elif action != "check":
            previous_bet_to_match = fsm.hand_state.current_bet_to_match
            player.current_bet += amount
            if player.stack is not None:
                player.stack -= amount
            if player.current_bet > fsm.hand_state.current_bet_to_match:
                fsm.hand_state.current_bet_to_match = player.current_bet
                fsm.hand_state.last_raiser = player_id
                ctx.last_raise_size = player.current_bet - previous_bet_to_match
            fsm.hand_state.pot += amount
            ctx.pot += amount

    if "next_player_id" in payload:
        ctx.turn_pointer = str(payload["next_player_id"])

    if "next_action_on_seat" in payload:
        fsm.hand_state.action_on_seat = int(payload["next_action_on_seat"])
    else:
        fsm.hand_state.action_on_seat = _seat_for_player_id(fsm.hand_state, ctx.turn_pointer)

    _sync_active_players(fsm.hand_state, ctx)
    _sync_current_bet(fsm.hand_state, ctx)


def _apply_street_reset(fsm: HandFSM, ctx: TableContext, payload: dict[str, Any]) -> None:
    for player in fsm.hand_state.players.values():
        player.current_bet = Decimal("0")
        player.has_acted_this_street = False

    fsm.hand_state.current_bet_to_match = Decimal("0")
    fsm.hand_state.last_raiser = None
    ctx.current_bet = Decimal("0")
    ctx.last_raise_size = Decimal("0")
    ctx.contributions_this_street = {}

    if "next_player_id" in payload:
        ctx.turn_pointer = str(payload["next_player_id"])

    if "action_on_seat" in payload:
        fsm.hand_state.action_on_seat = int(payload["action_on_seat"])
    elif "next_action_on_seat" in payload:
        fsm.hand_state.action_on_seat = int(payload["next_action_on_seat"])
    else:
        fsm.hand_state.action_on_seat = _seat_for_player_id(fsm.hand_state, ctx.turn_pointer)

    _sync_active_players(fsm.hand_state, ctx)


def _apply_visual_side_effects(fsm: HandFSM, ctx: TableContext, event: VisualEvent) -> None:
    if isinstance(event, BoardCardsRevealed):
        fsm.hand_state.board = list(event.cards)
        if len(event.cards) in (3, 4, 5):
            _apply_street_reset(fsm, ctx, {})
        return

    if isinstance(event, CardsMucked):
        player_id = _player_id_by_seat(fsm.hand_state, event.seat)
        if player_id is not None:
            fsm.hand_state.players[player_id].has_folded = True
        _sync_active_players(fsm.hand_state, ctx)
        return

    if isinstance(event, DealerButtonMoved):
        ctx.button_seat = event.new_seat
        return

    if isinstance(event, HeroBetDetected):
        hero_id = ctx.hero_id
        if hero_id in fsm.hand_state.players:
            player = fsm.hand_state.players[hero_id]
            if player.stack is not None:
                player.stack -= event.amount

            player.current_bet += event.amount
            fsm.hand_state.pot += event.amount
            ctx.pot += event.amount
            _sync_current_bet(fsm.hand_state, ctx)
        return


def _seat_for_player_id(hand_state: HandSnapshot, player_id: str | None) -> int | None:
    if player_id is None:
        return None
    player = hand_state.players.get(player_id)
    if player is None:
        return None
    return player.seat


def _player_id_by_seat(hand_state: HandSnapshot, seat: int) -> str | None:
    for player_id, player in hand_state.players.items():
        if player.seat == seat:
            return player_id
    return None


def _sync_active_players(hand_state: HandSnapshot, ctx: TableContext) -> None:
    ctx.active_players = [player_id for player_id, player in hand_state.players.items() if not player.has_folded]


def _sync_current_bet(hand_state: HandSnapshot, ctx: TableContext) -> None:
    current_bet = max(
        (player.current_bet for player in hand_state.players.values() if not player.has_folded), default=Decimal("0")
    )
    hand_state.current_bet_to_match = current_bet
    ctx.current_bet = current_bet
    ctx.contributions_this_street = {
        player_id: player.current_bet for player_id, player in hand_state.players.items() if player.current_bet > 0
    }


def _to_decimal(value: Any) -> Decimal:
    return Decimal(str(value))
