"""Tests for HandHistoryExporter. Mirrors test_window_attribution.py style."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional

import pytest

from poker_vision.export.json_exporter import (
    ActionInferenceMetadata,
    AmbiguityBuilder,
    AmbiguousWindowRef,
    ExporterConfig,
    ExportInputs,
    HandHistoryExporter,
    HandMetadata,
    PlayersBuilder,
    SeatMapper,
    StreetsBuilder,
    TableBuilder,
    WinnerInfo,
    encode_action_kind,
    encode_confidence,
    encode_money,
    make_action_id,
)
from poker_vision.inference.opponent_action_inferencer import (
    InferredAction,
    WindowAttribution,
)

# ============================================================
# Fake domain models (match real shapes structurally)
# ============================================================


@dataclass
class FakePlayer:
    player_id: str
    seat: int
    stack_initial: Decimal
    stack_final: Decimal
    is_hero: bool = False
    hole_cards: Optional[tuple[str, ...]] = None


@dataclass
class FakeAction:
    """Pure domain — exactly 4 fields, like the real ActionLogEntry."""

    street: str
    player_id: str
    action: str
    amount: Decimal


@dataclass
class FakeHand:
    hero_id: str
    button_seat: int
    blinds: tuple[Decimal, Decimal]
    players: dict[str, FakePlayer]
    action_log: list[FakeAction]
    pot_final: Decimal
    board_final: tuple[str, ...]


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def simple_hand() -> FakeHand:
    """3-handed hand: hero (BTN) raises, v1 folds, v2 calls; flop c-bet folds out."""
    return FakeHand(
        hero_id="hero",
        button_seat=6,
        blinds=(Decimal("1"), Decimal("2")),
        players={
            "v1": FakePlayer("v1", 1, Decimal("200"), Decimal("199")),
            "v2": FakePlayer("v2", 2, Decimal("200"), Decimal("194")),
            "hero": FakePlayer("hero", 6, Decimal("200"), Decimal("207"), is_hero=True, hole_cards=("Ah", "Kh")),
        },
        action_log=[
            FakeAction("preflop", "v1", "post", Decimal("1")),  # SB
            FakeAction("preflop", "v2", "post", Decimal("2")),  # BB
            FakeAction("preflop", "hero", "raise", Decimal("6")),
            FakeAction("preflop", "v1", "fold", Decimal("0")),
            FakeAction("preflop", "v2", "call", Decimal("4")),
            FakeAction("flop", "v2", "check", Decimal("0")),
            FakeAction("flop", "hero", "bet", Decimal("8")),
            FakeAction("flop", "v2", "fold", Decimal("0")),
        ],
        pot_final=Decimal("18"),
        board_final=("Qs", "Jd", "2c"),
    )


@pytest.fixture
def metadata() -> HandMetadata:
    return HandMetadata(
        hand_id="hand_001",
        table_id="table_42",
        timestamp_start=datetime(2026, 6, 9, 12, 0, 0),
        timestamp_end=datetime(2026, 6, 9, 12, 2, 30),
        stakes="1/2",
    )


# ============================================================
# Layer 1: primitive encoders
# ============================================================


class TestEncodeMoney:
    def test_basic_dollar_to_cents(self):
        assert encode_money(Decimal("10.50")) == 1050
        assert encode_money(Decimal("0.01")) == 1
        assert encode_money(Decimal("0")) == 0

    def test_no_float_drift_on_dime(self):
        # The classic float trap: 0.1 + 0.2 != 0.3 in float.
        # Decimal arithmetic must give exact 10 cents.
        assert encode_money(Decimal("0.10")) == 10

    def test_bankers_rounding(self):
        # 10.005 → 10.00 (rounds to even) at 2 decimals → 1000 cents.
        # 10.015 → 10.02 (rounds to even) at 2 decimals → 1002 cents.
        assert encode_money(Decimal("10.005")) == 1000
        assert encode_money(Decimal("10.015")) == 1002

    def test_large_amounts(self):
        assert encode_money(Decimal("1000000.00")) == 100_000_000

    def test_rejects_float(self):
        with pytest.raises(TypeError, match="Decimal"):
            encode_money(10.50)  # type: ignore[arg-type]

    def test_custom_scale_jpy(self):
        # JPY has 0 minor units — scale = 1.
        assert encode_money(Decimal("1000"), minor_unit_scale=1) == 1000

    def test_custom_scale_btc(self):
        # 1 BTC = 100_000_000 satoshi.
        assert encode_money(Decimal("0.00000001"), minor_unit_scale=100_000_000) == 1


class TestEncodeConfidence:
    def test_four_decimal_places(self):
        assert encode_confidence(Decimal("0.8")) == "0.8000"
        assert encode_confidence(Decimal("0.87423")) == "0.8742"

    def test_extremes(self):
        assert encode_confidence(Decimal("1.0")) == "1.0000"
        assert encode_confidence(Decimal("0")) == "0.0000"


class TestEncodeActionKind:
    @pytest.mark.parametrize(
        "python_kind,java_kind",
        [
            ("fold", "FOLD"),
            ("check", "CHECK"),
            ("call", "CALL"),
            ("bet", "BET"),
            ("raise", "RAISE"),
            ("post", "POST"),
        ],
    )
    def test_known_kinds(self, python_kind, java_kind):
        assert encode_action_kind(python_kind) == java_kind

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="Unknown ActionKind"):
            encode_action_kind("straddle")


class TestMakeActionId:
    def test_deterministic_format(self):
        assert make_action_id("preflop", 0) == "preflop-0"
        assert make_action_id("river", 7) == "river-7"

    def test_same_inputs_same_id(self):
        assert make_action_id("flop", 2) == make_action_id("flop", 2)


# ============================================================
# Layer 2a: SeatMapper
# ============================================================


class TestSeatMapper:
    def test_round_trip(self, simple_hand):
        m = SeatMapper.from_hand(simple_hand)
        assert m.seat_of("hero") == 6
        assert m.id_of(6) == "hero"

    def test_unknown_player_raises(self, simple_hand):
        m = SeatMapper.from_hand(simple_hand)
        with pytest.raises(KeyError, match="ghost"):
            m.seat_of("ghost")

    def test_duplicate_seat_rejected(self):
        bad = FakeHand(
            "x",
            1,
            (Decimal("1"), Decimal("2")),
            players={
                "a": FakePlayer("a", 1, Decimal("100"), Decimal("100")),
                "b": FakePlayer("b", 1, Decimal("100"), Decimal("100")),
            },
            action_log=[],
            pot_final=Decimal("0"),
            board_final=(),
        )
        with pytest.raises(ValueError, match="Duplicate seat"):
            SeatMapper.from_hand(bad)


# ============================================================
# Layer 2b: Builders
# ============================================================


class TestTableBuilder:
    def test_blinds_as_long_cents(self, simple_hand):
        block = TableBuilder.build(simple_hand, ExporterConfig())
        assert block["small_blind"] == 100
        assert block["big_blind"] == 200
        assert block["button_seat"] == 6


class TestPlayersBuilder:
    def test_sorted_by_seat(self, simple_hand):
        out = PlayersBuilder.build(simple_hand, ExporterConfig())
        assert [p["seat"] for p in out] == [1, 2, 6]

    def test_stacks_as_long_cents(self, simple_hand):
        out = PlayersBuilder.build(simple_hand, ExporterConfig())
        hero = next(p for p in out if p["is_hero"])
        assert hero["stack_initial"] == 20000
        assert hero["stack_final"] == 20700
        assert hero["hole_cards"] == ["Ah", "Kh"]


class TestStreetsBuilder:
    def test_blinds_emitted_as_post_kind(self, simple_hand):
        m = SeatMapper.from_hand(simple_hand)
        streets, _ = StreetsBuilder.build(simple_hand, m, {}, ExporterConfig())
        preflop = next(s for s in streets if s["street"] == "preflop")

        # Blinds present in output, kind=POST, ordered first.
        assert preflop["actions"][0]["kind"] == "POST"
        assert preflop["actions"][1]["kind"] == "POST"
        assert preflop["actions"][2]["kind"] == "RAISE"

    def test_action_ids_are_deterministic(self, simple_hand):
        m = SeatMapper.from_hand(simple_hand)
        streets, _ = StreetsBuilder.build(simple_hand, m, {}, ExporterConfig())
        preflop_ids = [a["action_id"] for a in streets[0]["actions"]]
        assert preflop_ids == ["preflop-0", "preflop-1", "preflop-2", "preflop-3", "preflop-4"]

    def test_log_position_map_covers_every_action(self, simple_hand):
        m = SeatMapper.from_hand(simple_hand)
        _, log_map = StreetsBuilder.build(simple_hand, m, {}, ExporterConfig())
        assert len(log_map) == len(simple_hand.action_log)
        assert log_map[0] == "preflop-0"
        assert log_map[5] == "flop-0"

    def test_inference_metadata_applied(self, simple_hand):
        m = SeatMapper.from_hand(simple_hand)
        inference = {
            3: ActionInferenceMetadata(confidence=Decimal("0.72"), window_id="w1"),
            4: ActionInferenceMetadata(confidence=Decimal("0.85"), window_id="w1"),
        }
        streets, _ = StreetsBuilder.build(simple_hand, m, inference, ExporterConfig())
        preflop = streets[0]["actions"]
        assert preflop[3]["confidence"] == "0.7200"
        assert preflop[3]["window_id"] == "w1"
        assert preflop[4]["window_id"] == "w1"
        # Hero raise (no inference metadata) defaults to confidence 1.0, no window.
        assert preflop[2]["confidence"] == "1.0000"
        assert preflop[2]["window_id"] is None

    def test_empty_streets_omitted(self, simple_hand):
        simple_hand.action_log = [a for a in simple_hand.action_log if a.street == "preflop"]
        m = SeatMapper.from_hand(simple_hand)
        streets, _ = StreetsBuilder.build(simple_hand, m, {}, ExporterConfig())
        assert {s["street"] for s in streets} == {"preflop"}

    def test_amount_is_long(self, simple_hand):
        m = SeatMapper.from_hand(simple_hand)
        streets, _ = StreetsBuilder.build(simple_hand, m, {}, ExporterConfig())
        hero_raise = streets[0]["actions"][2]
        assert hero_raise["amount"] == 600
        assert isinstance(hero_raise["amount"], int)


class TestAmbiguityBuilder:
    def test_unambiguous_window_skipped(self, simple_hand):
        m = SeatMapper.from_hand(simple_hand)
        _, log_map = StreetsBuilder.build(simple_hand, m, {}, ExporterConfig())
        attr = WindowAttribution(
            primary=[InferredAction("v1", "fold", Decimal("0"))],
            alternatives=[],
        )
        ref = AmbiguousWindowRef("w1", "preflop", [3], attr)
        result = AmbiguityBuilder.build([ref], m, log_map, ExporterConfig())
        assert result == []

    def test_ambiguous_window_full_shape(self, simple_hand):
        m = SeatMapper.from_hand(simple_hand)
        _, log_map = StreetsBuilder.build(simple_hand, m, {}, ExporterConfig())

        primary = [InferredAction("v1", "fold", Decimal("0")), InferredAction("v2", "call", Decimal("4"))]
        alt = [InferredAction("v1", "call", Decimal("4")), InferredAction("v2", "fold", Decimal("0"))]
        attr = WindowAttribution(primary=primary, alternatives=[alt])
        ref = AmbiguousWindowRef("w1", "preflop", [3, 4], attr)

        result = AmbiguityBuilder.build([ref], m, log_map, ExporterConfig())
        assert len(result) == 1
        w = result[0]
        assert w["window_id"] == "w1"
        assert w["street"] == "preflop"
        assert w["action_ids"] == ["preflop-3", "preflop-4"]
        assert w["primary_selection_method"] == "lexicographic_seat"
        assert len(w["alternatives"]) == 2  # primary + 1 alt
        # No consistent_set_size, no start_index/end_index in V1 schema.
        assert "consistent_set_size" not in w
        assert "start_index" not in w

    def test_assignments_use_seats_not_player_ids(self, simple_hand):
        m = SeatMapper.from_hand(simple_hand)
        _, log_map = StreetsBuilder.build(simple_hand, m, {}, ExporterConfig())
        attr = WindowAttribution(
            primary=[InferredAction("v1", "call", Decimal("2"))],
            alternatives=[[InferredAction("v1", "raise", Decimal("4"))]],
        )
        ref = AmbiguousWindowRef("w1", "preflop", [3], attr)
        result = AmbiguityBuilder.build([ref], m, log_map, ExporterConfig())
        first_assignment = result[0]["alternatives"][0]["assignments"][0]
        assert first_assignment["actor_seat"] == 1  # v1 → seat 1
        assert "player_id" not in first_assignment


# ============================================================
# Layer 3: end-to-end golden hand
# ============================================================


class TestGoldenHand:
    def test_top_level_shape(self, simple_hand, metadata):
        result = HandHistoryExporter().export(
            ExportInputs(
                hand=simple_hand,
                metadata=metadata,
                winners=[WinnerInfo("hero", Decimal("18"), "High Card Ace")],
            )
        )
        assert set(result.keys()) == {
            "schema_version",
            "metadata",
            "table",
            "players",
            "streets",
            "ambiguous_windows",
            "result",
        }

    def test_metadata_block_nested(self, simple_hand, metadata):
        result = HandHistoryExporter().export(
            ExportInputs(
                hand=simple_hand,
                metadata=metadata,
                winners=[WinnerInfo("hero", Decimal("18"))],
            )
        )
        assert result["metadata"]["hand_id"] == "hand_001"
        assert result["metadata"]["table_id"] == "table_42"
        # hand_id is NOT at the root.
        assert "hand_id" not in result

    def test_deterministic(self, simple_hand, metadata):
        exporter = HandHistoryExporter()
        inputs = ExportInputs(
            hand=simple_hand,
            metadata=metadata,
            winners=[WinnerInfo("hero", Decimal("18"))],
        )
        assert exporter.export(inputs) == exporter.export(inputs)

    def test_result_block_money_as_long(self, simple_hand, metadata):
        result = HandHistoryExporter().export(
            ExportInputs(
                hand=simple_hand,
                metadata=metadata,
                winners=[WinnerInfo("hero", Decimal("18"))],
            )
        )
        assert result["result"]["pot_final"] == 1800
        assert result["result"]["winners"][0]["amount_won"] == 1800
        assert result["result"]["winners"][0]["actor_seat"] == 6

    def test_export_to_json_is_valid_json(self, simple_hand, metadata):
        import json

        s = HandHistoryExporter().export_to_json(
            ExportInputs(
                hand=simple_hand,
                metadata=metadata,
                winners=[WinnerInfo("hero", Decimal("18"))],
            )
        )
        parsed = json.loads(s)
        assert parsed["metadata"]["hand_id"] == "hand_001"
