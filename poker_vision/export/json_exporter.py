"""
json_exporter.py
================

Serializes a finalized HandState (+ external metadata + inference
metadata + WindowAttributions) into the V1 hand-history dict consumed by
the Java Spring Boot CalibratedEvCalculator service.

Architecture
------------
Layer 3 : HandHistoryExporter           — public orchestrator
Layer 2 : SeatMapper + Builders         — one builder per top-level block
Layer 1 : Primitive encoders            — Decimal → minor-unit int, etc.

Design contract
---------------
* Pure: same inputs → same dict, no I/O, no input mutation.
* Money serialized as `Long` (integer minor units / cents) using
  Decimal.quantize() → int(). NEVER goes through float.
* Confidence serialized as a String (e.g. "0.8742") to match the Java DTO.
* Action IDs are deterministic: f"{street}-{action_index_within_street}".
* Ambiguous windows are top-level and reference actions by action_id list
  (not by positional indices), matching AmbiguousWindowDTO on the Java side.
* Blinds are identified by ActionKind == POST (first-class enum value),
  NOT by a heuristic or an is_blind flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import ROUND_HALF_EVEN, Decimal
from typing import Literal, Optional, Protocol

from poker_vision.inference.opponent_action_inferencer import (
    WindowAttribution,
)
from poker_vision.inference.table_context import ActionKind, Street

# ============================================================
# Type aliases & literals
# ============================================================

PrimarySelectionMethod = Literal[
    "lexicographic_seat",
    "highest_confidence",
    "manual_override",
]

# Maps Python ActionKind (lowercase) → Java ActionType enum value (uppercase).
# Keeping this explicit prevents a "rename in Python silently breaks Java" bug.
_ACTION_KIND_TO_JAVA: dict[str, str] = {
    "fold": "FOLD",
    "check": "CHECK",
    "call": "CALL",
    "bet": "BET",
    "raise": "RAISE",
    "post": "POST",
    "all_in": "ALL_IN",  # if your enum has it; harmless if unused
}


# ============================================================
# Domain-model shapes (structural Protocols)
# ============================================================
# The exporter reads from these shapes via duck typing. Your real
# poker_vision.logic.models classes already satisfy them — these
# Protocols exist purely for type-checker clarity and documentation.


class PlayerStateLike(Protocol):
    player_id: str
    seat: int
    stack_initial: Decimal
    stack_final: Decimal
    is_hero: bool
    hole_cards: Optional[tuple[str, ...]]


class ActionLogEntryLike(Protocol):
    """PURE domain entry — exactly four fields, nothing else."""

    street: Street
    player_id: str
    action: ActionKind  # includes ActionKind.POST for blinds
    amount: Decimal


class HandStateLike(Protocol):
    hero_id: str
    button_seat: int
    blinds: tuple[Decimal, Decimal]
    players: dict[str, PlayerStateLike]
    action_log: list[ActionLogEntryLike]
    pot_final: Decimal
    board_final: tuple[str, ...]


# ============================================================
# Inference metadata (parallel to action_log — Option A)
# ============================================================


@dataclass(frozen=True)
class ActionInferenceMetadata:
    """Per-action inference metadata, kept OUT of the pure domain model.

    Lives in a parallel map keyed by action_log position. The FSM is
    responsible for populating this map as it records actions, since
    only the FSM knows which actions came from the inferencer.
    """

    confidence: Decimal = Decimal("1.0")  # 1.0 = observed directly, not inferred
    window_id: Optional[str] = None  # set iff this action came from an ambiguous window


# ============================================================
# Exporter configuration
# ============================================================


@dataclass(frozen=True)
class ExporterConfig:
    schema_version: str = "1.0.0"
    # Currency unit scale. 100 for USD/EUR (cents), 1 for JPY,
    # 100_000_000 for BTC (satoshi). Drives Decimal → Long conversion.
    minor_unit_scale: int = 100
    include_inference_metadata: bool = True
    default_primary_selection_method: PrimarySelectionMethod = "lexicographic_seat"


# ============================================================
# External metadata (per the MetadataDTO contract)
# ============================================================


@dataclass(frozen=True)
class HandMetadata:
    hand_id: str
    table_id: str
    timestamp_start: datetime
    timestamp_end: datetime
    currency: str = "USD"
    game_type: str = "NLHE"
    stakes: str = ""


@dataclass(frozen=True)
class WinnerInfo:
    player_id: str
    amount_won: Decimal
    hand_description: str = ""


# ============================================================
# Layer 1: Primitive encoders
# ============================================================


def encode_money(amount: Decimal, minor_unit_scale: int = 100) -> int:
    """Serialize a Decimal as Long minor units (e.g. cents).

    Uses banker's rounding via Decimal.quantize — NEVER touches float.

        encode_money(Decimal("10.50"))   == 1050
        encode_money(Decimal("10.005"))  == 1000   (rounds to even)
        encode_money(Decimal("0.10"))    == 10     (exact; float would drift)
    """
    if not isinstance(amount, Decimal):
        raise TypeError(
            f"encode_money requires Decimal, got {type(amount).__name__}. "
            "Convert at the FSM boundary, not here, to preserve precision."
        )
    scaled = (amount * minor_unit_scale).quantize(Decimal("1"), rounding=ROUND_HALF_EVEN)
    return int(scaled)


def encode_confidence(confidence: Decimal) -> str:
    """Serialize confidence as a 4-decimal string (Java DTO expects String)."""
    quantized = Decimal(confidence).quantize(Decimal("0.0001"), rounding=ROUND_HALF_EVEN)
    return f"{quantized:f}"


def encode_cards(cards: Optional[tuple[str, ...]]) -> list[str]:
    return list(cards) if cards else []


def encode_action_kind(kind: ActionKind) -> str:
    """Translate Python ActionKind → Java ActionType enum value."""
    java_value = _ACTION_KIND_TO_JAVA.get(str(kind))
    if java_value is None:
        raise ValueError(
            f"Unknown ActionKind {kind!r}; add it to _ACTION_KIND_TO_JAVA "
            "and to the Java ActionType enum before exporting."
        )
    return java_value


def make_action_id(street: Street, index_within_street: int) -> str:
    """Deterministic action ID — same inputs ALWAYS produce same ID."""
    return f"{street}-{index_within_street}"


# ============================================================
# Layer 2a: SeatMapper
# ============================================================


@dataclass(frozen=True)
class SeatMapper:
    """Immutable, hand-scoped player_id ↔ actor_seat translator."""

    _id_to_seat: dict[str, int]
    _seat_to_id: dict[int, str]

    @classmethod
    def from_hand(cls, hand: HandStateLike) -> SeatMapper:
        id_to_seat = {pid: p.seat for pid, p in hand.players.items()}
        seat_to_id = {seat: pid for pid, seat in id_to_seat.items()}
        if len(id_to_seat) != len(seat_to_id):
            raise ValueError("Duplicate seat assignment in HandState.players — " "every seat must be unique.")
        return cls(_id_to_seat=id_to_seat, _seat_to_id=seat_to_id)

    def seat_of(self, player_id: str) -> int:
        try:
            return self._id_to_seat[player_id]
        except KeyError:
            raise KeyError(f"Unknown player_id {player_id!r}. " f"Known: {sorted(self._id_to_seat)}") from None

    def id_of(self, seat: int) -> str:
        return self._seat_to_id[seat]


# ============================================================
# Layer 2b: Builders
# ============================================================


class MetadataBuilder:
    """Builds the top-level `metadata` block (MetadataDTO on Java side)."""

    @staticmethod
    def build(meta: HandMetadata) -> dict:
        return {
            "hand_id": meta.hand_id,
            "table_id": meta.table_id,
            "timestamp_start": meta.timestamp_start.isoformat(),
            "timestamp_end": meta.timestamp_end.isoformat(),
            "currency": meta.currency,
            "game_type": meta.game_type,
            "stakes": meta.stakes,
        }


class TableBuilder:
    """Builds the `table` block."""

    @staticmethod
    def build(hand: HandStateLike, cfg: ExporterConfig) -> dict:
        sb, bb = hand.blinds
        return {
            "button_seat": hand.button_seat,
            "small_blind": encode_money(sb, cfg.minor_unit_scale),
            "big_blind": encode_money(bb, cfg.minor_unit_scale),
        }


class PlayersBuilder:
    """Builds the `players` array, sorted by seat."""

    @staticmethod
    def build(hand: HandStateLike, cfg: ExporterConfig) -> list[dict]:
        sorted_players = sorted(hand.players.values(), key=lambda p: p.seat)
        return [
            {
                "seat": p.seat,
                "player_id": p.player_id,
                "is_hero": p.is_hero,
                "stack_initial": encode_money(p.stack_initial, cfg.minor_unit_scale),
                "stack_final": encode_money(p.stack_final, cfg.minor_unit_scale),
                "hole_cards": encode_cards(p.hole_cards),
            }
            for p in sorted_players
        ]


class StreetsBuilder:
    """Builds the `streets` array + index map for the AmbiguityBuilder.

    The index map translates `action_log` positions → synthetic action_ids,
    so the AmbiguityBuilder can resolve WindowAttribution windows into
    List<String> action_ids without ever knowing about list positions.

    Blinds (ActionKind.POST) are recorded as actions with kind=POST in the
    output — they are NOT filtered out, because the Java side wants the
    full action stream. They simply carry their own kind.
    """

    STREET_ORDER: tuple[Street, ...] = ("preflop", "flop", "turn", "river")

    @staticmethod
    def build(
        hand: HandStateLike,
        seat_mapper: SeatMapper,
        inference_meta: dict[int, ActionInferenceMetadata],
        cfg: ExporterConfig,
    ) -> tuple[list[dict], dict[int, str]]:
        """Returns (streets_array, log_position_to_action_id_map)."""
        streets_out: list[dict] = []
        log_pos_to_action_id: dict[int, str] = {}

        # Bucket the flat log by street, preserving order.
        per_street: dict[Street, list[tuple[int, ActionLogEntryLike]]] = {s: [] for s in StreetsBuilder.STREET_ORDER}
        for log_pos, entry in enumerate(hand.action_log):
            if entry.street in per_street:
                per_street[entry.street].append((log_pos, entry))

        for street in StreetsBuilder.STREET_ORDER:
            entries = per_street[street]
            if not entries:
                continue

            actions_out: list[dict] = []
            for idx_within_street, (log_pos, entry) in enumerate(entries):
                action_id = make_action_id(street, idx_within_street)
                log_pos_to_action_id[log_pos] = action_id

                meta = inference_meta.get(log_pos, ActionInferenceMetadata())
                actions_out.append(
                    {
                        "action_id": action_id,
                        "actor_seat": seat_mapper.seat_of(entry.player_id),
                        "kind": encode_action_kind(entry.action),
                        "amount": encode_money(entry.amount, cfg.minor_unit_scale),
                        "confidence": encode_confidence(meta.confidence),
                        "window_id": meta.window_id,
                    }
                )

            streets_out.append(
                {
                    "street": street,
                    "board": encode_cards(StreetsBuilder._board_for_street(hand, street)),
                    "actions": actions_out,
                }
            )

        return streets_out, log_pos_to_action_id

    @staticmethod
    def _board_for_street(hand: HandStateLike, street: Street) -> tuple[str, ...]:
        board = hand.board_final
        return {
            "preflop": (),
            "flop": board[:3],
            "turn": board[:4],
            "river": board[:5],
        }.get(street, ())


@dataclass(frozen=True)
class AmbiguousWindowRef:
    """How the FSM points at a window: which log positions it covers,
    plus the attribution itself and the selection method used.

    The FSM owns the (window_id, log_positions[], method) triple because
    only the FSM knows which actions a window produced.
    """

    window_id: str
    street: Street
    log_positions: list[int]
    attribution: WindowAttribution
    primary_selection_method: PrimarySelectionMethod = "lexicographic_seat"


class AmbiguityBuilder:
    """Builds the top-level `ambiguous_windows` array.

    Schema per AmbiguousWindowDTO:
        window_id, street, action_ids (List<String>),
        primary_selection_method, alternatives
    """

    @staticmethod
    def build(
        refs: list[AmbiguousWindowRef],
        seat_mapper: SeatMapper,
        log_pos_to_action_id: dict[int, str],
        cfg: ExporterConfig,
    ) -> list[dict]:
        out: list[dict] = []
        for ref in refs:
            attr = ref.attribution
            if not attr.is_ambiguous:
                continue

            # Resolve log positions → action_ids. Skip silently if a
            # position never made it into the streets block (e.g. dropped
            # by the FSM); the window simply references fewer actions.
            action_ids = [log_pos_to_action_id[pos] for pos in ref.log_positions if pos in log_pos_to_action_id]
            if not action_ids:
                continue

            # Primary first, then alternatives (rank order from inferencer).
            all_sequences = [attr.primary] + list(attr.alternatives)

            weights = getattr(attr, "weights", [])
            if len(weights) != len(all_sequences):
                weights = [1.0 / len(all_sequences)] * len(all_sequences)
            alternatives_out = [
                {
                    "weight": float(weights[i]),
                    "assignments": [
                        {
                            "actor_seat": seat_mapper.seat_of(a.player_id),
                            "kind": encode_action_kind(a.action),
                            "amount": encode_money(a.amount, cfg.minor_unit_scale),
                        }
                        for a in seq
                    ],
                }
                for i, seq in enumerate(all_sequences)
            ]

            out.append(
                {
                    "window_id": ref.window_id,
                    "street": ref.street,
                    "action_ids": action_ids,
                    "primary_selection_method": ref.primary_selection_method,
                    "alternatives": alternatives_out,
                }
            )

        return out


class ResultBuilder:
    """Builds the `result` block."""

    @staticmethod
    def build(
        hand: HandStateLike,
        winners: list[WinnerInfo],
        seat_mapper: SeatMapper,
        cfg: ExporterConfig,
    ) -> dict:
        return {
            "pot_final": encode_money(hand.pot_final, cfg.minor_unit_scale),
            "board": encode_cards(hand.board_final),
            "winners": [
                {
                    "actor_seat": seat_mapper.seat_of(w.player_id),
                    "amount_won": encode_money(w.amount_won, cfg.minor_unit_scale),
                    "hand_description": w.hand_description,
                }
                for w in winners
            ],
        }


# ============================================================
# Layer 3: Orchestrator
# ============================================================


@dataclass(frozen=True)
class ExportInputs:
    """Bundles every input the exporter needs."""

    hand: HandStateLike
    metadata: HandMetadata
    winners: list[WinnerInfo]
    inference_metadata: dict[int, ActionInferenceMetadata] = field(default_factory=dict)
    ambiguous_windows: list[AmbiguousWindowRef] = field(default_factory=list)


class HandHistoryExporter:
    """Pure function from ExportInputs → HandHistoryRequestDTO-shaped dict."""

    def __init__(self, config: Optional[ExporterConfig] = None) -> None:
        self.cfg = config or ExporterConfig()

    def export(self, inputs: ExportInputs) -> dict:
        hand = inputs.hand
        seat_mapper = SeatMapper.from_hand(hand)

        metadata_block = MetadataBuilder.build(inputs.metadata)
        table_block = TableBuilder.build(hand, self.cfg)
        players_block = PlayersBuilder.build(hand, self.cfg)
        streets_block, log_pos_to_action_id = StreetsBuilder.build(
            hand, seat_mapper, inputs.inference_metadata, self.cfg
        )
        ambiguous_block = AmbiguityBuilder.build(inputs.ambiguous_windows, seat_mapper, log_pos_to_action_id, self.cfg)
        result_block = ResultBuilder.build(hand, inputs.winners, seat_mapper, self.cfg)

        return {
            "schema_version": self.cfg.schema_version,
            "metadata": metadata_block,
            "table": table_block,
            "players": players_block,
            "streets": streets_block,
            "ambiguous_windows": ambiguous_block,
            "result": result_block,
        }

    def export_to_json(self, inputs: ExportInputs) -> str:
        import json

        return json.dumps(self.export(inputs), ensure_ascii=False, indent=2)


__all__ = [
    "HandHistoryExporter",
    "ExportInputs",
    "ExporterConfig",
    "HandMetadata",
    "WinnerInfo",
    "ActionInferenceMetadata",
    "AmbiguousWindowRef",
    "SeatMapper",
    "MetadataBuilder",
    "TableBuilder",
    "PlayersBuilder",
    "StreetsBuilder",
    "AmbiguityBuilder",
    "ResultBuilder",
    "encode_money",
    "encode_confidence",
    "encode_action_kind",
    "encode_cards",
    "make_action_id",
    "PrimarySelectionMethod",
]
