"""Tests for TableContextBuilder.

We use minimal mock FSM states rather than the real HandPhase, because the
builder is defined against Protocols and we want these tests to remain
valid even if the FSM implementation changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

import pytest

from poker_vision.inference.table_context import ActionKind
from poker_vision.inference.table_context_builder import (
    BuilderConfig,
    TableContextBuilder,
)

# =========================================================================
# Mock FSM state
# =========================================================================


@dataclass
class MockPlayer:
    player_id: str
    is_active: bool = True
    stack: Decimal = Decimal("1000")
    stack_measurement_age_frames: int = 0
    contribution_this_street: Decimal = Decimal("0")


@dataclass
class MockHand:
    pot: Decimal = Decimal("30")
    current_bet: Decimal = Decimal("10")
    last_raise_size: Decimal = Decimal("10")
    big_blind: Decimal = Decimal("10")
    street: str = "preflop"
    turn_pointer: str | None = None
    action_order: tuple[str, ...] = ()
    players: dict[str, MockPlayer] = field(default_factory=dict)
    _legal: dict[str, frozenset[ActionKind]] = field(default_factory=dict)

    def legal_actions_for(self, player_id: str) -> frozenset[ActionKind]:
        return self._legal.get(player_id, frozenset({"fold", "call", "raise"}))


def make_mock_hand(**overrides) -> MockHand:
    players = {
        "hero": MockPlayer("hero", contribution_this_street=Decimal("10")),
        "v1": MockPlayer("v1", contribution_this_street=Decimal("10")),
        "v2": MockPlayer("v2"),
    }
    defaults = dict(players=players)
    defaults.update(overrides)
    return MockHand(**defaults)


# =========================================================================
# Happy paths
# =========================================================================


def test_build_measured_only_when_fsm_uninitialized():
    """Before the FSM establishes turn order, the builder still produces
    a valid (measured-only) context."""
    builder = TableContextBuilder()
    hand = make_mock_hand()  # turn_pointer=None

    ctx = builder.build(hand)

    assert ctx.has_fsm_state is False
    assert ctx.turn_pointer is None
    assert ctx.action_order == ()
    assert ctx.pot == Decimal("30")
    assert set(ctx.active_players) == {"hero", "v1", "v2"}


def test_build_with_fsm_state_populated():
    """When the FSM has turn order, the builder populates derived state."""
    builder = TableContextBuilder()
    hand = make_mock_hand(
        turn_pointer="v2",
        action_order=("v2", "hero", "v1"),
    )

    ctx = builder.build(hand)

    assert ctx.has_fsm_state is True
    assert ctx.turn_pointer == "v2"
    assert ctx.action_order == ("v2", "hero", "v1")
    assert ctx.constraint_sources["turn_pointer"] == "fsm_derived"


def test_inactive_players_excluded_from_active_set():
    """Folded players should not appear in active_players or
    legal_actions_per_player."""
    builder = TableContextBuilder()
    hand = make_mock_hand()
    hand.players["v1"].is_active = False  # v1 folded

    ctx = builder.build(hand)

    assert "v1" not in ctx.active_players
    assert "v1" not in ctx.legal_actions_per_player
    assert "v1" not in ctx.player_stacks


def test_non_folders_threaded_through():
    """Reconciliation's non_folders set should propagate to the context."""
    builder = TableContextBuilder()
    hand = make_mock_hand()

    ctx = builder.build(hand, non_folders=frozenset({"hero"}))

    assert "hero" in ctx.non_folders
    assert ctx.constraint_sources.get("non_folders") == "showdown_reveals"


def test_provenance_custom_source():
    builder = TableContextBuilder()
    hand = make_mock_hand()

    ctx = builder.build(
        hand,
        non_folders=frozenset({"hero"}),
        non_folders_source="synthetic_corpus",
    )

    assert ctx.constraint_sources["non_folders"] == "synthetic_corpus"


# =========================================================================
# Strict mode failures
# =========================================================================


def test_strict_non_folders_inconsistent_with_fsm():
    """If non_folders names a player the FSM has folded, strict mode raises."""
    builder = TableContextBuilder()
    hand = make_mock_hand()
    hand.players["v1"].is_active = False

    with pytest.raises(ValueError, match="players the FSM has as folded"):
        builder.build(hand, non_folders=frozenset({"v1"}))


def test_lenient_drops_inconsistent_non_folders(caplog):
    builder = TableContextBuilder(BuilderConfig(strict=False))
    hand = make_mock_hand()
    hand.players["v1"].is_active = False

    ctx = builder.build(hand, non_folders=frozenset({"v1", "hero"}))

    assert ctx.non_folders == frozenset({"hero"})
    assert "Dropping non_folders" in caplog.text


def test_require_fsm_state_raises_when_uninitialized():
    builder = TableContextBuilder(BuilderConfig(require_fsm_state=True, strict=True))
    hand = make_mock_hand()  # turn_pointer=None

    with pytest.raises(ValueError, match="no turn_pointer"):
        builder.build(hand)


def test_strict_empty_legal_actions_caught():
    """An FSM bug where a player has no legal actions should fail loudly."""
    builder = TableContextBuilder()
    hand = make_mock_hand(turn_pointer="hero", action_order=("hero",))
    hand._legal = {"hero": frozenset(), "v1": frozenset({"fold"}), "v2": frozenset({"fold"})}

    with pytest.raises(ValueError, match="empty legal_actions"):
        builder.build(hand)


# =========================================================================
# Determinism
# =========================================================================


def test_active_players_deterministic_ordering():
    """active_players should be sorted, so identical FSM states produce
    identical contexts (important for golden-fixture tests)."""
    builder = TableContextBuilder()

    # Build the same hand twice with different dict insertion order.
    h1 = MockHand(
        players={
            "v2": MockPlayer("v2"),
            "hero": MockPlayer("hero"),
            "v1": MockPlayer("v1"),
        }
    )
    h2 = MockHand(
        players={
            "hero": MockPlayer("hero"),
            "v1": MockPlayer("v1"),
            "v2": MockPlayer("v2"),
        }
    )

    ctx1 = builder.build(h1)
    ctx2 = builder.build(h2)

    assert ctx1.active_players == ctx2.active_players
    assert ctx1.active_players == ["hero", "v1", "v2"]


def test_action_order_is_tuple_not_list():
    """action_order must be hashable for snapshot_constraints() to work."""
    builder = TableContextBuilder()
    hand = make_mock_hand(turn_pointer="hero", action_order=("hero", "v1", "v2"))

    ctx = builder.build(hand)

    assert isinstance(ctx.action_order, tuple)
    # Sanity: should be hashable
    hash(ctx.action_order)


# =========================================================================
# Constraint snapshot integration
# =========================================================================


def test_built_context_snapshot_is_stable():
    """A built context's snapshot should be stable across calls."""
    builder = TableContextBuilder()
    hand = make_mock_hand(turn_pointer="hero", action_order=("hero", "v1", "v2"))

    ctx = builder.build(hand, non_folders=frozenset({"hero"}))
    snap1 = ctx.snapshot_constraints()
    snap2 = ctx.snapshot_constraints()

    assert snap1 == snap2
