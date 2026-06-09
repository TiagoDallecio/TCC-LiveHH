from __future__ import annotations

from decimal import Decimal

import pytest

from poker_vision.inference.table_context import TableContext

# =========================================================================
# Fixtures
# =========================================================================


def make_minimal_ctx(**overrides) -> TableContext:
    """A minimally-valid TableContext for tests. Override fields as needed."""
    defaults = dict(
        pot=Decimal("100"),
        current_bet=Decimal("10"),
        last_raise_size=Decimal("10"),
        big_blind=Decimal("10"),
        contributions_this_street={"hero": Decimal("10"), "v1": Decimal("10")},
        street="preflop",
        active_players=["hero", "v1", "v2"],
    )
    defaults.update(overrides)
    return TableContext(**defaults)


# =========================================================================
# Happy paths
# =========================================================================


def test_minimal_context_is_valid():
    ctx = make_minimal_ctx()
    assert ctx.turn_pointer is None
    assert ctx.has_fsm_state is False
    assert ctx.action_order == ()


def test_fsm_state_population():
    ctx = make_minimal_ctx(
        turn_pointer="hero",
        action_order=("hero", "v1", "v2"),
    )
    assert ctx.has_fsm_state is True
    assert ctx.action_order == ("hero", "v1", "v2")


def test_legal_actions_lenient_default():
    """Players with no legal_actions entry should be unconstrained."""
    ctx = make_minimal_ctx()
    assert ctx.is_action_legal("hero", "raise") is True
    assert ctx.is_action_legal("v1", "fold") is True


def test_legal_actions_constrained():
    ctx = make_minimal_ctx(
        legal_actions_per_player={
            "hero": frozenset({"check", "bet"}),
        }
    )
    assert ctx.is_action_legal("hero", "check") is True
    assert ctx.is_action_legal("hero", "fold") is False
    assert ctx.is_action_legal("v1", "fold") is True  # no constraint


# =========================================================================
# Invariant violations (I1–I7)
# =========================================================================


def test_i1_turn_pointer_must_be_active():
    with pytest.raises(ValueError, match="turn_pointer.*not in active_players"):
        make_minimal_ctx(turn_pointer="ghost")


def test_i2_action_order_requires_turn_pointer():
    with pytest.raises(ValueError, match="action_order is non-empty"):
        make_minimal_ctx(action_order=("hero", "v1"))


def test_i2_action_order_starts_with_turn_pointer():
    with pytest.raises(ValueError, match="action_order\\[0\\].*does not match"):
        make_minimal_ctx(
            turn_pointer="hero",
            action_order=("v1", "hero"),
        )


def test_i3_action_order_only_active_players():
    with pytest.raises(ValueError, match="non-active players"):
        make_minimal_ctx(
            turn_pointer="hero",
            action_order=("hero", "ghost"),
        )


def test_i4_non_folders_must_be_active():
    with pytest.raises(ValueError, match="non_folders.*non-active"):
        make_minimal_ctx(non_folders=frozenset({"ghost"}))


def test_i5_legal_actions_only_active():
    with pytest.raises(ValueError, match="legal_actions_per_player.*non-active"):
        make_minimal_ctx(
            legal_actions_per_player={"ghost": frozenset({"fold"})},
        )


def test_i6_negative_pot_rejected():
    with pytest.raises(ValueError, match="pot must be non-negative"):
        make_minimal_ctx(pot=Decimal("-1"))


def test_i7_zero_big_blind_rejected():
    with pytest.raises(ValueError, match="big_blind must be > 0"):
        make_minimal_ctx(big_blind=Decimal("0"))


# =========================================================================
# Soft-prior decay
# =========================================================================


def test_stack_prior_weight_no_measurement():
    ctx = make_minimal_ctx()
    assert ctx.stack_prior_weight("hero") == 0.0


def test_stack_prior_weight_fresh():
    ctx = make_minimal_ctx(
        player_stacks={"hero": Decimal("1000")},
        stack_measurement_age_frames={"hero": 0},
    )
    assert ctx.stack_prior_weight("hero") == 1.0


def test_stack_prior_weight_decays_linearly():
    ctx = make_minimal_ctx(
        player_stacks={"hero": Decimal("1000")},
        stack_measurement_age_frames={"hero": 45},
    )
    assert ctx.stack_prior_weight("hero", max_age_frames=90) == pytest.approx(0.5)


def test_stack_prior_weight_stale():
    ctx = make_minimal_ctx(
        player_stacks={"hero": Decimal("1000")},
        stack_measurement_age_frames={"hero": 200},
    )
    assert ctx.stack_prior_weight("hero", max_age_frames=90) == 0.0
