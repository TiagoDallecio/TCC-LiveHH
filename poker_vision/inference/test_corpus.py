"""
Test Corpus Generator for OpponentActionInferencer
===================================================

Builds (anchor_start, anchor_end, ctx, expected_actions) tuples from two
sources:

  1. PokerStars .txt hand histories  -> statistical volume
  2. Custom YAML fixtures            -> surgical edge cases

Supports Hero re-rooting (data augmentation: N seats -> N test cases per hand)
and showdown-reveal metadata for post-hoc reconciliation testing (Task 4.5.9).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Iterable, Literal, Optional, cast

import yaml

from poker_vision.inference.opponent_action_inferencer import (
    AnchorEvent,
    AnchorType,
    HeroAction,
    InferredAction,
    Street,
    TableContext,
)
from poker_vision.inference.table_context import ActionKind

# ---------------------------------------------------------------------------
# 1. Public data model
# ---------------------------------------------------------------------------

Complexity = Literal["trivial", "simple", "moderate", "complex"]


@dataclass
class WindowTestCase:
    case_id: str
    ctx_before: TableContext
    anchor_start: AnchorEvent
    anchor_end: AnchorEvent
    expected_actions: list[InferredAction]
    metadata: dict = field(default_factory=dict)
    non_folders: list[str] = field(default_factory=list)

    # Convenience accessors
    @property
    def complexity(self) -> Complexity:
        return self.metadata.get("complexity", "moderate")

    @property
    def num_villains_in_window(self) -> int:
        return len(self.expected_actions)


# ---------------------------------------------------------------------------
# 2. Internal: parsed hand representation
# ---------------------------------------------------------------------------


@dataclass
class ParsedAction:
    player_id: str
    action: str  # fold/check/call/bet/raise/all_in
    amount: Decimal
    street: Street


@dataclass
class ParsedHand:
    hand_id: str
    source: str  # file path or "yaml:<name>"
    num_players: int
    button_seat: int
    seat_order: list[str]  # clockwise from seat 0
    hero_id: str  # the explicitly tagged "Hero"
    small_blind: Decimal
    big_blind: Decimal
    blinds_posted: list[tuple[str, Decimal]]  # (player, amount)
    actions_by_street: dict[Street, list[ParsedAction]]
    board_by_street: dict[Street, tuple[str, ...]]
    showdown_reveals: dict[str, list[str]]  # player_id -> hole cards
    winners: list[str]  # for post-hoc reconciliation


# ---------------------------------------------------------------------------
# 3. PokerStars .txt parser
# ---------------------------------------------------------------------------


class PokerStarsParser:
    """
    Minimal PokerStars NL Hold'em parser. Handles the formats used by the
    desktop client circa 2018-2024. Robust enough for thesis-scale corpora.
    """

    _RE_HAND_HEADER = re.compile(r"PokerStars Hand #(?P<hand_id>\d+):\s+.*?\(\$?(?P<sb>[\d.]+)/\$?(?P<bb>[\d.]+)")
    _RE_BUTTON = re.compile(r"Seat #(?P<seat>\d+) is the button")
    _RE_SEAT = re.compile(r"Seat (?P<seat>\d+):\s+(?P<name>.+?)\s+\(\$?(?P<stack>[\d.]+) in chips\)")
    _RE_POST_BLIND = re.compile(r"(?P<name>.+?): posts\s+(?:small|big)\s+blind\s+\$?(?P<amount>[\d.]+)")
    _RE_ACTION = re.compile(
        r"(?P<name>.+?):\s+(?P<verb>folds|checks|calls|bets|raises)"
        r"(?:\s+\$?(?P<amount>[\d.]+))?"
        r"(?:\s+to\s+\$?(?P<raise_to>[\d.]+))?"
        r"(?P<all_in>\s+and is all-in)?"
    )
    _RE_BOARD = re.compile(r"Board\s+\[(?P<cards>[^\]]+)\]")
    _RE_FLOP = re.compile(r"\*\*\* FLOP \*\*\* \[(?P<cards>[^\]]+)\]")
    _RE_TURN = re.compile(r"\*\*\* TURN \*\*\* \[[^\]]+\]\s+\[(?P<card>[^\]]+)\]")
    _RE_RIVER = re.compile(r"\*\*\* RIVER \*\*\* \[[^\]]+\]\s+\[(?P<card>[^\]]+)\]")
    _RE_SHOWS = re.compile(r"(?P<name>.+?): shows\s+\[(?P<cards>[^\]]+)\]")
    _RE_MUCKS = re.compile(r"(?P<name>.+?): mucks hand")
    _RE_COLLECTED = re.compile(r"(?P<name>.+?) collected \$?(?P<amount>[\d.]+)")

    def parse_file(self, path: Path) -> list[ParsedHand]:
        text = path.read_text(encoding="utf-8", errors="replace")
        chunks = re.split(r"\n\n\n+", text.strip())
        out: list[ParsedHand] = []
        for chunk in chunks:
            try:
                hand = self._parse_chunk(chunk, source=str(path))
                if hand is not None:
                    out.append(hand)
            except Exception:
                # Skip malformed hands silently; log in production code.
                continue
        return out

    def _parse_chunk(self, chunk: str, source: str) -> Optional[ParsedHand]:
        header_match = self._RE_HAND_HEADER.search(chunk)
        if not header_match:
            return None

        hand_id = header_match.group("hand_id")
        sb = Decimal(header_match.group("sb"))
        bb = Decimal(header_match.group("bb"))

        # --- Seats and button ---
        seats: dict[int, str] = {}
        for m in self._RE_SEAT.finditer(chunk):
            seats[int(m.group("seat"))] = m.group("name").strip()

        button_match = self._RE_BUTTON.search(chunk)
        if not button_match or not seats:
            return None
        button_seat_num = int(button_match.group("seat"))

        # Order seats clockwise starting from seat 1 (PokerStars convention).
        sorted_seat_nums = sorted(seats.keys())
        seat_order = [seats[s] for s in sorted_seat_nums]
        button_idx = sorted_seat_nums.index(button_seat_num)

        # Hero detection: PokerStars writes "Dealt to Hero [...]".
        hero_id = "Hero" if "Hero" in seat_order else seat_order[0]

        # --- Blinds ---
        blinds: list[tuple[str, Decimal]] = []
        # Split chunk into preflop section before parsing blinds vs actions.
        preflop_section, _, post_preflop = chunk.partition("*** HOLE CARDS ***")
        for m in self._RE_POST_BLIND.finditer(preflop_section):
            blinds.append((m.group("name").strip(), Decimal(m.group("amount"))))

        # --- Streets ---
        sections = self._split_streets(chunk)
        actions_by_street: dict[Street, list[ParsedAction]] = {}
        board_by_street: dict[Street, tuple[str, ...]] = {}

        flop_cards: tuple[str, ...] = ()
        turn_card: tuple[str, ...] = ()
        river_card: tuple[str, ...] = ()

        flop_match = self._RE_FLOP.search(chunk)
        if flop_match:
            flop_cards = tuple(c.strip() for c in flop_match.group("cards").split())
        turn_match = self._RE_TURN.search(chunk)
        if turn_match:
            turn_card = (turn_match.group("card").strip(),)
        river_match = self._RE_RIVER.search(chunk)
        if river_match:
            river_card = (river_match.group("card").strip(),)

        board_by_street["preflop"] = ()
        board_by_street["flop"] = flop_cards
        board_by_street["turn"] = flop_cards + turn_card
        board_by_street["river"] = flop_cards + turn_card + river_card

        for street, section in sections.items():
            if street == "showdown":
                continue
            street_typed = cast(Street, street)
            actions_by_street[street_typed] = self._parse_actions(section, street_typed)

        # --- Showdown reveals ---
        showdown_section = sections.get("showdown", "")
        reveals: dict[str, list[str]] = {}
        for m in self._RE_SHOWS.finditer(showdown_section or chunk):
            cards = [c.strip() for c in m.group("cards").split()]
            reveals[m.group("name").strip()] = cards

        winners: list[str] = []
        for m in self._RE_COLLECTED.finditer(chunk):
            winners.append(m.group("name").strip())

        return ParsedHand(
            hand_id=hand_id,
            source=source,
            num_players=len(seats),
            button_seat=button_idx,
            seat_order=seat_order,
            hero_id=hero_id,
            small_blind=sb,
            big_blind=bb,
            blinds_posted=blinds,
            actions_by_street=actions_by_street,
            board_by_street=board_by_street,
            showdown_reveals=reveals,
            winners=winners,
        )

    @staticmethod
    def _split_streets(chunk: str) -> dict[str, str]:
        markers = [
            ("preflop", "*** HOLE CARDS ***", "*** FLOP ***"),
            ("flop", "*** FLOP ***", "*** TURN ***"),
            ("turn", "*** TURN ***", "*** RIVER ***"),
            ("river", "*** RIVER ***", "*** SHOW DOWN ***"),
            ("showdown", "*** SHOW DOWN ***", "*** SUMMARY ***"),
        ]
        out: dict[str, str] = {}
        for name, start, end in markers:
            i = chunk.find(start)
            if i == -1:
                continue
            j = chunk.find(end, i)
            out[name] = chunk[i:j] if j != -1 else chunk[i:]
        return out

    def _parse_actions(self, section: str, street: Street) -> list[ParsedAction]:
        out: list[ParsedAction] = []
        for line in section.splitlines():
            m = self._RE_ACTION.search(line)
            if not m:
                continue
            verb = m.group("verb")
            name = m.group("name").strip()
            amount_str = m.group("raise_to") or m.group("amount") or "0"
            amount = Decimal(amount_str)
            is_all_in = bool(m.group("all_in"))
            action = {
                "folds": "fold",
                "checks": "check",
                "calls": "call",
                "bets": "bet",
                "raises": "raise",
            }[verb]
            if is_all_in:
                action = "all_in"
            out.append(ParsedAction(player_id=name, action=action, amount=amount, street=street))
        return out


# ---------------------------------------------------------------------------
# 4. Custom YAML fixture loader
# ---------------------------------------------------------------------------


class YamlFixtureLoader:
    """
    Loads hand-written YAML fixtures for surgical edge-case tests.

    Schema example:

        hand_id: edge_3bet_pot
        num_players: 4
        button_seat: 0
        seat_order: [V1, V2, V3, Hero]
        hero_id: Hero
        small_blind: 0.5
        big_blind: 1.0
        blinds: [[V2, 0.5], [V3, 1.0]]
        boards:
          flop: [As, Kh, 7d]
        actions:
          preflop:
            - {player: Hero, action: raise, amount: 3.0}
            - {player: V1, action: fold}
            - {player: V2, action: fold}
            - {player: V3, action: raise, amount: 10.0}
            - {player: Hero, action: call, amount: 7.0}
          flop:
            - {player: V3, action: bet, amount: 12.0}
            - {player: Hero, action: fold}
        showdown_reveals: {}
        winners: [V3]
    """

    def load_file(self, path: Path) -> list[ParsedHand]:
        docs = list(yaml.safe_load_all(path.read_text(encoding="utf-8")))
        return [self._to_parsed_hand(d, source=f"yaml:{path.name}") for d in docs if d is not None]

    def _to_parsed_hand(self, doc: dict, source: str) -> ParsedHand:
        actions_by_street: dict[Street, list[ParsedAction]] = {}
        for street, items in (doc.get("actions") or {}).items():
            actions_by_street[street] = [
                ParsedAction(
                    player_id=a["player"],
                    action=a["action"],
                    amount=Decimal(str(a.get("amount", 0))),
                    street=street,
                )
                for a in items
            ]

        boards_raw = doc.get("boards") or {}
        flop = tuple(boards_raw.get("flop", []))
        turn = tuple(
            boards_raw.get("turn", flop + (boards_raw.get("turn_card"),) if boards_raw.get("turn_card") else flop)
        )
        river = tuple(boards_raw.get("river", turn))
        board_by_street: dict[Street, tuple[str, ...]] = {
            "preflop": (),
            "flop": flop,
            "turn": turn,
            "river": river,
        }

        return ParsedHand(
            hand_id=doc["hand_id"],
            source=source,
            num_players=doc["num_players"],
            button_seat=doc["button_seat"],
            seat_order=list(doc["seat_order"]),
            hero_id=doc.get("hero_id", "Hero"),
            small_blind=Decimal(str(doc["small_blind"])),
            big_blind=Decimal(str(doc["big_blind"])),
            blinds_posted=[(p, Decimal(str(a))) for p, a in doc.get("blinds", [])],
            actions_by_street=actions_by_street,
            board_by_street=board_by_street,
            showdown_reveals=doc.get("showdown_reveals") or {},
            winners=list(doc.get("winners") or []),
        )


# ---------------------------------------------------------------------------
# 5. Window extraction from a parsed hand (given a chosen Hero)
# ---------------------------------------------------------------------------


@dataclass
class _SimState:
    """Mutable state used while walking through a parsed hand."""

    pot: Decimal
    current_bet: Decimal
    last_raise_size: Decimal
    contributions_this_street: dict[str, Decimal]
    active_players: list[str]
    street: Street
    turn_pointer: str


def _first_to_act_preflop(seat_order: list[str], button_idx: int, n: int) -> int:
    # Heads-up: button acts first preflop. Otherwise UTG = button + 3.
    if n == 2:
        return button_idx
    return (button_idx + 3) % n


def _first_to_act_postflop(seat_order: list[str], button_idx: int, active: list[str]) -> str:
    n = len(seat_order)
    for k in range(1, n + 1):
        candidate = seat_order[(button_idx + k) % n]
        if candidate in active:
            return candidate
    return active[0]


def _next_active(current: str, seat_order: list[str], active: list[str]) -> str:
    if not active:
        return current
    n = len(seat_order)
    i = seat_order.index(current)
    for k in range(1, n + 1):
        cand = seat_order[(i + k) % n]
        if cand in active:
            return cand
    return current


def _complexity_tag(num_villain_actions: int, has_raise: bool) -> Complexity:
    if num_villain_actions == 0:
        return "trivial"
    if num_villain_actions <= 2 and not has_raise:
        return "simple"
    if num_villain_actions <= 4 or has_raise:
        return "moderate"
    return "complex"


def _derive_showdown_inferences(
    hand: ParsedHand,
    hero_id: str,
) -> dict[str, dict]:
    """
    Build a per-player showdown summary for post-hoc reconciliation:
      - reached_showdown: bool
      - revealed_cards:   list[str] | None
      - folded_street:    Street | None  (back-deducible from action log)
    """
    out: dict[str, dict] = {}
    folded_at: dict[str, Street] = {}
    for street, actions in hand.actions_by_street.items():
        for a in actions:
            if a.action == "fold":
                folded_at.setdefault(a.player_id, street)
    for player in hand.seat_order:
        if player == hero_id:
            continue
        reveal = hand.showdown_reveals.get(player)
        out[player] = {
            "reached_showdown": player not in folded_at,
            "revealed_cards": reveal,
            "folded_street": folded_at.get(player),
        }
    return out


def extract_windows_for_hero(
    hand: ParsedHand,
    hero_id: str,
) -> list[WindowTestCase]:
    """
    Walk through `hand` treating `hero_id` as Hero. Emit one test case per
    Hero-bounded window (plus a final window from last Hero action to hand end).
    """
    cases: list[WindowTestCase] = []
    n = hand.num_players
    seat_order = hand.seat_order
    button_idx = hand.button_seat
    hero_seat = seat_order.index(hero_id)

    # --- Initialize sim state with blinds posted ---
    state = _SimState(
        pot=Decimal(0),
        current_bet=Decimal(0),
        last_raise_size=hand.big_blind,
        contributions_this_street={},
        active_players=list(seat_order),
        street="preflop",
        turn_pointer=seat_order[_first_to_act_preflop(seat_order, button_idx, n)],
    )
    for player, amount in hand.blinds_posted:
        state.pot += amount
        state.contributions_this_street[player] = state.contributions_this_street.get(player, Decimal(0)) + amount
        if amount > state.current_bet:
            state.current_bet = amount

    showdown_meta = _derive_showdown_inferences(hand, hero_id)

    # --- Window accumulator ---
    pending_anchor: Optional[AnchorEvent] = None
    pending_ctx: Optional[TableContext] = None
    window_buffer: list[InferredAction] = []
    window_has_raise = False

    def snapshot_ctx() -> TableContext:
        return TableContext(
            num_players=n,
            button_seat=button_idx,
            hero_seat=hero_seat,
            seat_order=list(seat_order),
            active_players=list(state.active_players),
            current_street=state.street,
            current_bet=state.current_bet,
            last_raise_size=state.last_raise_size,
            turn_pointer=state.turn_pointer,
            pot=state.pot,
            contributions_this_street=dict(state.contributions_this_street),
            hero_id=hero_id,
        )

    def emit_case(anchor_end: AnchorEvent, end_reason: str) -> None:
        nonlocal pending_anchor, pending_ctx, window_buffer, window_has_raise
        if pending_anchor is None or pending_ctx is None:
            return
        non_folders_list = []
        for p, meta in showdown_meta.items():
            if meta["reached_showdown"]:
                non_folders_list.append(p)
            elif meta["folded_street"] is not None:
                streets_order = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
                if streets_order.get(meta["folded_street"], 0) > streets_order.get(state.street, 0):
                    non_folders_list.append(p)

        complexity = _complexity_tag(len(window_buffer), window_has_raise)
        case = WindowTestCase(
            case_id=f"{hand.hand_id}::hero={hero_id}::end={end_reason}@{state.street}",
            ctx_before=pending_ctx,
            anchor_start=pending_anchor,
            anchor_end=anchor_end,
            expected_actions=list(window_buffer),
            non_folders=non_folders_list,
            metadata={
                "source": hand.source,
                "hand_id": hand.hand_id,
                "hero_id": hero_id,
                "street_at_window_end": state.street,
                "complexity": complexity,
                "num_villains_in_window": len(window_buffer),
                "showdown_reveals": showdown_meta,
                "winners": list(hand.winners),
                "non_folders_count": len(non_folders_list),
            },
        )
        cases.append(case)
        window_buffer = []
        window_has_raise = False

    # --- Open the first window at hand start ---
    pending_ctx = snapshot_ctx()
    pending_anchor = AnchorEvent(
        anchor_type=AnchorType.HAND_START,
        timestamp=0.0,
        street="preflop",
        pot_before=Decimal(0),
        pot_after=state.pot,
        board=(),
    )

    timestamp = 1.0
    streets_in_order: list[Street] = ["preflop", "flop", "turn", "river"]

    for street in streets_in_order:
        if street != "preflop":
            # Street transition: emit pending window, then open a new one.
            pot_before_transition = state.pot
            state.street = street
            state.current_bet = Decimal(0)
            state.last_raise_size = hand.big_blind
            state.contributions_this_street = {}
            state.turn_pointer = _first_to_act_postflop(seat_order, button_idx, state.active_players)
            anchor = AnchorEvent(
                anchor_type=AnchorType.STREET_START,
                timestamp=timestamp,
                street=street,
                pot_before=pot_before_transition,
                pot_after=state.pot,
                board=hand.board_by_street.get(street, ()),
            )
            timestamp += 1.0
            emit_case(anchor, end_reason="street_start")
            pending_anchor = anchor
            pending_ctx = snapshot_ctx()

        for pa in hand.actions_by_street.get(street, []):
            # Apply action to sim state.
            pc = state.contributions_this_street.get(pa.player_id, Decimal(0))
            if pa.action in ("fold",):
                if pa.player_id in state.active_players:
                    state.active_players.remove(pa.player_id)
            elif pa.action == "check":
                pass
            elif pa.action == "call":
                to_call = state.current_bet - pc
                state.pot += to_call
                state.contributions_this_street[pa.player_id] = pc + to_call
            elif pa.action in ("bet", "raise", "all_in"):
                # `amount` in PokerStars "raise X to Y" form is Y (raise_to).
                raise_to = pa.amount if pa.action != "bet" else pa.amount
                contribution = raise_to - pc
                if contribution < 0:
                    contribution = pa.amount  # fallback for bets
                    raise_to = pc + contribution
                state.pot += contribution
                state.contributions_this_street[pa.player_id] = raise_to
                if raise_to - state.current_bet > 0:
                    state.last_raise_size = raise_to - state.current_bet
                state.current_bet = max(state.current_bet, raise_to)

            if pa.player_id == hero_id:
                hero_kind = cast(ActionKind, pa.action)
                if hero_kind == "all_in":
                    hero_kind = cast(ActionKind, "raise" if state.current_bet > 0 else "bet")
                hero_amount = pa.amount
                anchor = AnchorEvent(
                    anchor_type=AnchorType.HERO_ACTION,
                    timestamp=timestamp,
                    street=street,
                    pot_before=state.pot,
                    pot_after=state.pot,
                    board=hand.board_by_street.get(street, ()),
                    hero_action=HeroAction(action=hero_kind, amount=hero_amount),
                )
                timestamp += 1.0
                emit_case(anchor, end_reason="hero_action")
                pending_anchor = anchor
                pending_ctx = snapshot_ctx()
            else:
                # Buffer this villain action as ground truth.
                if pa.action in ("bet", "raise", "all_in"):
                    window_has_raise = True
                window_buffer.append(
                    InferredAction(
                        player_id=pa.player_id,
                        action=cast(ActionKind, pa.action),
                        amount=(pa.amount - pc) if pa.action in ("call", "bet", "raise", "all_in") else Decimal(0),
                        confidence=1.0,
                        rationale="ground truth",
                        is_inferred=False,
                    )
                )

            state.turn_pointer = _next_active(pa.player_id, seat_order, state.active_players)

    # --- Final window: hand_end anchor ---
    anchor = AnchorEvent(
        anchor_type=AnchorType.HAND_END,
        timestamp=timestamp,
        street=state.street,
        pot_before=state.pot,
        pot_after=state.pot,
        board=hand.board_by_street.get(state.street, ()),
    )
    emit_case(anchor, end_reason="hand_end")
    return cases


# ---------------------------------------------------------------------------
# 6. Corpus builder (orchestrator with re-rooting)
# ---------------------------------------------------------------------------


@dataclass
class CorpusBuildOptions:
    reroot: bool = True  # treat every seat as Hero
    include_trivial: bool = True  # keep zero-action windows
    skip_uncontested: bool = False  # drop hands where all but one fold preflop
    max_hands: Optional[int] = None  # cap for fast dev cycles


class CorpusBuilder:
    """
    Top-level entrypoint. Combines parsed sources into a single test corpus.
    """

    def __init__(self, options: Optional[CorpusBuildOptions] = None) -> None:
        self.options = options or CorpusBuildOptions()
        self.pokerstars = PokerStarsParser()
        self.yaml_loader = YamlFixtureLoader()

    def build(
        self,
        pokerstars_files: Iterable[Path] = (),
        yaml_files: Iterable[Path] = (),
    ) -> list[WindowTestCase]:
        hands: list[ParsedHand] = []
        for p in pokerstars_files:
            hands.extend(self.pokerstars.parse_file(p))
        for p in yaml_files:
            hands.extend(self.yaml_loader.load_file(p))

        if self.options.max_hands is not None:
            hands = hands[: self.options.max_hands]

        cases: list[WindowTestCase] = []
        for hand in hands:
            if self.options.skip_uncontested and self._is_uncontested(hand):
                continue
            hero_candidates = hand.seat_order if self.options.reroot else [hand.hero_id]
            for hero in hero_candidates:
                for case in extract_windows_for_hero(hand, hero):
                    if not self.options.include_trivial and case.complexity == "trivial":
                        continue
                    cases.append(case)
        return cases

    @staticmethod
    def _is_uncontested(hand: ParsedHand) -> bool:
        preflop = hand.actions_by_street.get("preflop", [])
        folds = sum(1 for a in preflop if a.action == "fold")
        return folds >= hand.num_players - 1 and not hand.actions_by_street.get("flop")


# ---------------------------------------------------------------------------
# 7. Serialization (JSON, for caching expensive parses)
# ---------------------------------------------------------------------------


def _serialize_decimal(obj):
    if isinstance(obj, Decimal):
        return str(obj)
    raise TypeError(f"Not serializable: {type(obj)}")


def save_corpus(cases: list[WindowTestCase], path: Path) -> None:
    payload = [_case_to_dict(c) for c in cases]
    path.write_text(
        json.dumps(payload, indent=2, default=_serialize_decimal),
        encoding="utf-8",
    )


def load_corpus(path: Path) -> list[WindowTestCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [_case_from_dict(d) for d in raw]


def _case_to_dict(c: WindowTestCase) -> dict:
    return {
        "case_id": c.case_id,
        "ctx_before": _ctx_to_dict(c.ctx_before),
        "anchor_start": _anchor_to_dict(c.anchor_start),
        "anchor_end": _anchor_to_dict(c.anchor_end),
        "expected_actions": [_action_to_dict(a) for a in c.expected_actions],
        "metadata": c.metadata,
        "non_folders": getattr(c, "non_folders", []),
    }


def _case_from_dict(d: dict) -> WindowTestCase:
    return WindowTestCase(
        case_id=d["case_id"],
        ctx_before=_ctx_from_dict(d["ctx_before"]),
        anchor_start=_anchor_from_dict(d["anchor_start"]),
        anchor_end=_anchor_from_dict(d["anchor_end"]),
        expected_actions=[_action_from_dict(a) for a in d["expected_actions"]],
        metadata=d.get("metadata", {}),
        non_folders=d.get("non_folders", []),
    )


def _ctx_to_dict(ctx: TableContext) -> dict:
    out = asdict(ctx)
    out["current_bet"] = str(ctx.current_bet)
    out["last_raise_size"] = str(ctx.last_raise_size)
    out["pot"] = str(ctx.pot)
    out["contributions_this_street"] = {k: str(v) for k, v in ctx.contributions_this_street.items()}
    return out


def _ctx_from_dict(d: dict) -> TableContext:
    return TableContext(
        num_players=d["num_players"],
        button_seat=d["button_seat"],
        hero_seat=d["hero_seat"],
        seat_order=list(d["seat_order"]),
        active_players=list(d["active_players"]),
        current_street=d["current_street"],
        current_bet=Decimal(d["current_bet"]),
        last_raise_size=Decimal(d["last_raise_size"]),
        turn_pointer=d["turn_pointer"],
        pot=Decimal(d["pot"]),
        contributions_this_street={k: Decimal(v) for k, v in d["contributions_this_street"].items()},
        hero_id=d["hero_id"],
    )


def _anchor_to_dict(a: AnchorEvent) -> dict:
    return {
        "anchor_type": a.anchor_type.value,
        "timestamp": a.timestamp,
        "street": a.street,
        "pot_before": str(a.pot_before),
        "pot_after": str(a.pot_after),
        "board": list(a.board),
        "hero_action": (
            {"action": a.hero_action.action, "amount": str(a.hero_action.amount)} if a.hero_action else None
        ),
        "pot_estimator_quality": a.pot_estimator_quality,
    }


def _anchor_from_dict(d: dict) -> AnchorEvent:
    hero = d.get("hero_action")
    return AnchorEvent(
        anchor_type=AnchorType(d["anchor_type"]),
        timestamp=d["timestamp"],
        street=d["street"],
        pot_before=Decimal(d["pot_before"]),
        pot_after=Decimal(d["pot_after"]),
        board=tuple(d.get("board", [])),
        hero_action=HeroAction(action=hero["action"], amount=Decimal(hero["amount"])) if hero else None,
        pot_estimator_quality=d.get("pot_estimator_quality", 1.0),
    )


def _action_to_dict(a: InferredAction) -> dict:
    return {
        "player_id": a.player_id,
        "action": a.action,
        "amount": str(a.amount),
        "confidence": a.confidence,
        "rationale": a.rationale,
        "is_inferred": a.is_inferred,
    }


def _action_from_dict(d: dict) -> InferredAction:
    return InferredAction(
        player_id=d["player_id"],
        action=d["action"],
        amount=Decimal(d["amount"]),
        confidence=d.get("confidence", 1.0),
        rationale=d.get("rationale", ""),
        is_inferred=d.get("is_inferred", False),
    )


__all__ = [
    "Complexity",
    "CorpusBuildOptions",
    "CorpusBuilder",
    "ParsedAction",
    "ParsedHand",
    "PokerStarsParser",
    "WindowTestCase",
    "YamlFixtureLoader",
    "extract_windows_for_hero",
    "load_corpus",
    "save_corpus",
]
