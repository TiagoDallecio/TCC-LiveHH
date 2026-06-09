from decimal import Decimal
from unittest.mock import patch

import pytest

from poker_vision.inference import opponent_action_inferencer
from poker_vision.inference.opponent_action_inferencer import (
    FSMCandidateDisagreement,
    InferencerConfig,
    _resolve_actor_for_event,
    enumerate_action_sequences,
    get_inference_metrics,
    reset_inference_metrics,
)
from poker_vision.inference.table_context import TableContext


def test_resolve_raises_in_strict_mode():
    ctx = make_ctx(action_order=("v2", "hero", "v1"))
    with patch.object(opponent_action_inferencer, "ENABLE_FSM_HARD_PRUNING", True):
        with pytest.raises(FSMCandidateDisagreement):
            _resolve_actor_for_event(0, ctx, fallback_candidates=["hero"], strict=True)


def test_resolve_falls_back_in_lenient_mode_and_increments_metric():
    ctx = make_ctx(action_order=("v2", "hero", "v1"))
    reset_inference_metrics()
    with patch.object(opponent_action_inferencer, "ENABLE_FSM_HARD_PRUNING", True):
        candidates = _resolve_actor_for_event(0, ctx, fallback_candidates=["hero"], strict=False)
    assert candidates == ["hero"]
    assert get_inference_metrics()["fsm_candidate_disagreement"] == 1


def make_ctx(**overrides) -> TableContext:
    """Minimal context with FSM state populated."""
    defaults = dict(
        pot=Decimal("100"),
        current_bet=Decimal("10"),
        last_raise_size=Decimal("10"),
        big_blind=Decimal("10"),
        contributions_this_street={"hero": Decimal("0"), "v1": Decimal("0"), "v2": Decimal("0")},
        street="preflop",
        active_players=["v2", "hero", "v1"],
        turn_pointer="v2",
        action_order=("v2", "hero", "v1"),
    )
    defaults.update(overrides)
    return TableContext(**defaults)


def test_flag_on_prunes_invalid_player_order():
    """If the player sequence differs from FSM action_order, it must be pruned."""
    ctx = make_ctx(action_order=("v2", "hero", "v1"))
    cfg = InferencerConfig()

    players_wrong_order = ["hero", "v2", "v1"]
    delta_pot = Decimal("30")

    with patch.object(opponent_action_inferencer, "ENABLE_FSM_HARD_PRUNING", True):
        sequences = enumerate_action_sequences(players_wrong_order, delta_pot, ctx, cfg)

    assert len(sequences) == 0


def test_flag_on_respects_legal_actions():
    """If FSM says hero can only check/bet, fold/call must be pruned."""
    ctx = make_ctx(
        legal_actions_per_player={
            "v2": frozenset({"fold", "call", "raise"}),
            "hero": frozenset({"check", "bet"}),  # restrito!
            "v1": frozenset({"fold", "call", "raise"}),
        }
    )
    cfg = InferencerConfig()
    players = ["v2", "hero", "v1"]
    delta_pot = Decimal("30")

    with patch.object(opponent_action_inferencer, "ENABLE_FSM_HARD_PRUNING", True):
        sequences = enumerate_action_sequences(players, delta_pot, ctx, cfg)

    for seq in sequences:
        hero_actions = [a for a in seq if a.player_id == "hero"]
        for a in hero_actions:
            assert a.action in {"check", "bet"}


def test_flag_off_ignores_invalid_player_order():
    """With pruning disabled, enumeration must behave exactly as before."""
    ctx = make_ctx(action_order=("v2", "hero", "v1"))
    cfg = InferencerConfig()

    players_wrong_order = ["hero", "v2", "v1"]
    delta_pot = Decimal("30")

    with patch.object(opponent_action_inferencer, "ENABLE_FSM_HARD_PRUNING", False):
        sequences = enumerate_action_sequences(players_wrong_order, delta_pot, ctx, cfg)

    assert len(sequences) > 0
