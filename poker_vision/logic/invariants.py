from __future__ import annotations

import logging
from decimal import Decimal

from poker_vision.logic.models import HandState

logger = logging.getLogger(__name__)


class InvariantViolationError(ValueError):
    pass


def check_invariants(hand_state: HandState) -> None:
    active_players_bet_sum = Decimal("0")
    non_folded_player_count = 0
    seat_to_players: dict[int, list[str]] = {}

    for player_id, player in hand_state.players.items():
        seat_to_players.setdefault(player.seat, []).append(player_id)

        if player.stack is not None and player.stack < 0:
            _raise_violation(
                hand_state,
                f"Invariant failed: player '{player_id}' has negative stack={player.stack}",
            )

        if player.current_bet < 0:
            _raise_violation(
                hand_state,
                f"Invariant failed: player '{player_id}' has negative current_bet={player.current_bet}",
            )

        if not player.has_folded:
            non_folded_player_count += 1
            active_players_bet_sum += player.current_bet

            if player.current_bet > hand_state.current_bet_to_match:
                _raise_violation(
                    hand_state,
                    "Invariant failed: active player bet exceeds current_bet_to_match "
                    f"(player='{player_id}', current_bet={player.current_bet}, "
                    f"current_bet_to_match={hand_state.current_bet_to_match})",
                )

    duplicated_seats = {seat: ids for seat, ids in seat_to_players.items() if len(ids) > 1}
    if duplicated_seats:
        _raise_violation(
            hand_state,
            f"Invariant failed: duplicate seats assigned to multiple players ({duplicated_seats})",
        )

    if active_players_bet_sum > hand_state.pot:
        _raise_violation(
            hand_state,
            "Invariant failed: sum of active player bets exceeds pot "
            f"(active_bets_sum={active_players_bet_sum}, pot={hand_state.pot})",
        )

    if hand_state.action_on_seat is not None:
        acting_players = seat_to_players.get(hand_state.action_on_seat, [])
        if len(acting_players) == 0:
            _raise_violation(
                hand_state,
                "Invariant failed: action_on_seat does not match any player seat "
                f"(action_on_seat={hand_state.action_on_seat})",
            )
        if len(acting_players) > 1:
            _raise_violation(
                hand_state,
                "Invariant failed: more than one player is acting on the same seat "
                f"(action_on_seat={hand_state.action_on_seat}, players={acting_players})",
            )

    if hand_state.last_raiser is not None and hand_state.last_raiser not in hand_state.players:
        _raise_violation(
            hand_state,
            "Invariant failed: last_raiser is not a known player " f"(last_raiser='{hand_state.last_raiser}')",
        )

    if hand_state.current_bet_to_match > 0 and non_folded_player_count == 0:
        _raise_violation(
            hand_state,
            "Invariant failed: positive current_bet_to_match with no active players "
            f"(current_bet_to_match={hand_state.current_bet_to_match})",
        )


def _raise_violation(hand_state: HandState, message: str) -> None:
    hand_state.quality.needs_review = True
    logger.error(
        "%s | street=%s pot=%s current_bet_to_match=%s action_on_seat=%s last_raiser=%s players=%s",
        message,
        hand_state.street,
        hand_state.pot,
        hand_state.current_bet_to_match,
        hand_state.action_on_seat,
        hand_state.last_raiser,
        {
            player_id: {
                "seat": player.seat,
                "stack": str(player.stack),
                "current_bet": str(player.current_bet),
                "has_folded": player.has_folded,
                "has_acted_this_street": player.has_acted_this_street,
            }
            for player_id, player in hand_state.players.items()
        },
    )
    raise InvariantViolationError(message)
