"""
OpponentActionInferencer
========================

Reconstructs opponent actions between observable anchor events (street
transitions, Hero actions, hand end) using pot-delta attribution.

The inferencer is *stateless across hands* — all per-hand state lives in the
TableContext supplied by the FSM. The inferencer's job is purely:

    (anchor_start, anchor_end, ctx) -> [InferredAction, ...]

Design reference: see thesis chapter "Hero-Centric Inference Engine".
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Optional

from poker_vision.inference.table_context import ActionKind, Street, TableContext

ENABLE_FSM_HARD_PRUNING = False


class FSMCandidateDisagreement(RuntimeError):
    def __init__(self, *, event_index: int, expected_actor: str, candidates: list[str]) -> None:
        self.event_index = event_index
        self.expected_actor = expected_actor
        self.candidates = candidates
        super().__init__(
            f"FSM expects actor {expected_actor!r} at event {event_index} " f"but candidates are {candidates!r}."
        )


_INFERENCE_METRICS: dict[str, int] = {
    "horizon_overflow": 0,
    "fsm_candidate_disagreement": 0,
}


def reset_inference_metrics() -> None:
    for k in _INFERENCE_METRICS:
        _INFERENCE_METRICS[k] = 0


def get_inference_metrics() -> dict[str, int]:
    return dict(_INFERENCE_METRICS)


class AnchorType(str, Enum):
    HAND_START = "hand_start"
    STREET_START = "street_start"
    HERO_ACTION = "hero_action"
    HAND_END = "hand_end"


@dataclass(frozen=True)
class HeroAction:
    action: ActionKind
    amount: Decimal


@dataclass(frozen=True)
class AnchorEvent:
    """An observable event the CV/FSM layer is highly confident about."""

    anchor_type: AnchorType
    timestamp: float
    street: Street
    pot_before: Decimal
    pot_after: Decimal
    board: tuple[str, ...] = ()
    hero_action: Optional[HeroAction] = None
    pot_estimator_quality: float = 1.0


@dataclass
class InferredAction:
    player_id: str
    action: ActionKind
    amount: Decimal
    confidence: float = 0.0
    rationale: str = ""
    is_inferred: bool = True
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class EnumerationResult:
    """Output of enumerate_action_sequences with truncation metadata."""

    sequences: list[list[InferredAction]]
    enumeration_capped: bool = False
    raw_count_before_cap: int = 0


@dataclass(frozen=True)
class WindowAttribution:
    """Full attribution result for one anchor-bounded window."""

    primary: list[InferredAction]
    alternatives: list[list[InferredAction]]
    weights: list[float] = field(default_factory=list)
    enumeration_capped: bool = False
    deduplication_high_collision: bool = False
    was_reconciled: bool = False
    constraint_relaxed: bool = False

    @property
    def is_ambiguous(self) -> bool:
        return len(self.alternatives) > 0

    @property
    def consistent_set_size(self) -> int:
        """Total number of consistent sequences (primary + alternatives)."""
        return 1 + len(self.alternatives) if self.primary else 0


# ---------------------------------------------------------------------------
# 2. Configuration (all tunable hyperparameters)
# ---------------------------------------------------------------------------


@dataclass
class InferencerConfig:
    pot_noise_abs: Decimal = Decimal("2")
    pot_noise_rel: float = 0.05

    max_sequence_depth: int = 8
    raise_size_grid: tuple[float, ...] = (
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        4.0,
        6.0,
    )
    pot_fraction_grid: tuple[float, ...] = (
        0.33,
        0.5,
        0.66,
        0.75,
        1.0,
        1.5,
        2.0,
        3.0,
    )

    action_priors: dict[Street, dict[ActionKind, float]] = field(
        default_factory=lambda: {
            "preflop": {"fold": 0.55, "call": 0.20, "raise": 0.18, "check": 0.07},
            "flop": {"check": 0.55, "fold": 0.15, "call": 0.20, "raise": 0.10},
            "turn": {"check": 0.50, "fold": 0.18, "call": 0.20, "raise": 0.12},
            "river": {"check": 0.45, "fold": 0.20, "call": 0.22, "raise": 0.13},
        }
    )

    sizing_sharpness: float = 4.0

    occam_quadratic_weight: float = 0.05

    occam_linear_weight: float = 0.15

    rebrace_penalty: float = 0.25

    constraint_pruning_margin_weight: float = 1.5
    reconciliation_dampening: float = 0.15
    confidence_margin_gain: float = 2.0
    min_window_confidence: float = 0.05
    min_action_confidence: float = 0.05
    confidence_forced_action: float = 0.97
    confidence_clean_check: float = 0.92
    confidence_inferred_fold: float = 0.65
    confidence_dfs_base: float = 0.55
    confidence_dfs_margin_weight: float = 0.35
    confidence_dfs_max: float = 0.90
    confidence_fallback_action: float = 0.20
    confidence_fallback_fold: float = 0.25

    zero_delta_epsilon: Decimal = Decimal("2.0")

    action_confidence_modifier: dict[ActionKind, float] = field(
        default_factory=lambda: {
            "fold": +0.05,
            "check": +0.02,
            "call": 0.00,
            "raise": -0.10,
            "bet": -0.10,
            "all_in": -0.15,
        }
    )

    # All-in heuristic: contribution > this multiple of current_bet -> all_in.
    all_in_contribution_multiplier: Decimal = Decimal("5")


# ---------------------------------------------------------------------------
# 3. Helpers
# ---------------------------------------------------------------------------


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _safe_log(p: float) -> float:
    return math.log(max(p, 1e-6))


def _players_between(
    start_after: str,
    end_before: str,
    seat_order: list[str],
    active: list[str],
) -> list[str]:
    """
    Return active players whose turn falls strictly *after* start_after and
    strictly *before* end_before (clockwise), wrapping around.
    """
    if not active:
        return []
    active_set = set(active)
    n = len(seat_order)
    try:
        i = seat_order.index(start_after)
    except ValueError:
        return []
    out: list[str] = []
    for k in range(1, n + 1):
        p = seat_order[(i + k) % n]
        if p == end_before:
            break
        if p in active_set:
            out.append(p)
    return out


# ---------------------------------------------------------------------------
# 4. Action-sequence enumeration (DFS with poker legality pruning)
# ---------------------------------------------------------------------------


def _legal_raise_contributions(
    current_bet: Decimal,
    last_raise: Decimal,
    player_contribution: Decimal,
    remaining_delta: Decimal,
    pot: Decimal,
    cfg: InferencerConfig,
) -> list[Decimal]:
    """
    Generate a discrete set of legal raise *contributions* (what this player
    adds to the pot) that fit within remaining_delta.
    """
    min_total = current_bet + max(last_raise, Decimal("0.01"))
    candidates: set[Decimal] = set()

    # Grid 1: multiples of min-raise total.
    for k in cfg.raise_size_grid:
        raise_to = (min_total * Decimal(str(k))).quantize(Decimal("0.01"))
        contribution = raise_to - player_contribution
        if 0 < contribution <= remaining_delta:
            candidates.add(contribution)

    # Grid 2: canonical pot-fraction bets (informative postflop).
    for frac in cfg.pot_fraction_grid:
        raise_to = (pot * Decimal(str(frac)) + current_bet).quantize(Decimal("0.01"))
        if raise_to < min_total:
            continue
        contribution = raise_to - player_contribution
        if 0 < contribution <= remaining_delta:
            candidates.add(contribution)

    # Grid 3: exact remaining_delta (single-player-takes-all-of-Δ).
    if remaining_delta >= (min_total - player_contribution):
        candidates.add(remaining_delta)

    return sorted(candidates)


def _resolve_actor_for_event(
    event_index: int,
    ctx: TableContext,
    *,
    fallback_candidates: list[str],
    strict: bool = False,
) -> list[str]:
    if not ENABLE_FSM_HARD_PRUNING:
        return fallback_candidates
    if not ctx.has_fsm_state:
        return fallback_candidates
    if event_index >= len(ctx.action_order):
        _INFERENCE_METRICS["horizon_overflow"] += 1
        return fallback_candidates

    expected_actor = ctx.action_order[event_index]
    if expected_actor in fallback_candidates:
        return [expected_actor]

    _INFERENCE_METRICS["fsm_candidate_disagreement"] += 1
    if strict:
        raise FSMCandidateDisagreement(
            event_index=event_index,
            expected_actor=expected_actor,
            candidates=fallback_candidates,
        )
    return fallback_candidates


def _action_is_fsm_legal(
    player: str,
    action: ActionKind,
    ctx: TableContext,
) -> bool:
    """Check whether an action kind is FSM-legal for a player.

    Uses TableContext.is_action_legal which returns True for players
    with no constraint entry (lenient default), so this is safe to call
    even when the FSM hasn't populated legal_actions_per_player."""
    if not ENABLE_FSM_HARD_PRUNING:
        return True
    return ctx.is_action_legal(player, action)


def enumerate_action_sequences(
    players: list[str],
    delta_pot: Decimal,
    ctx: TableContext,
    cfg: InferencerConfig,
    non_folders: frozenset[str] = frozenset(),
    *,
    max_alternatives: int = 16,
) -> EnumerationResult:
    """
    DFS through legal action sequences whose contributions sum to delta_pot.
    Caps enumeration at `max_alternatives` consistent sequences.
    """
    results: list[list[InferredAction]] = []
    capped = False

    if len(players) > cfg.max_sequence_depth:
        players = players[: cfg.max_sequence_depth]

    contribs0 = dict(ctx.contributions_this_street)

    def dfs(
        idx: int,
        remaining: Decimal,
        current_bet: Decimal,
        last_raise: Decimal,
        contribs: dict[str, Decimal],
        seq: list[InferredAction],
    ) -> None:
        nonlocal capped
        if capped:
            return  # short-circuit all branches once cap is hit

        if idx == len(players):
            if remaining == 0:
                results.append([InferredAction(a.player_id, a.action, a.amount) for a in seq])
                if len(results) >= max_alternatives:
                    capped = True
            return

        player = players[idx]

        candidate_players = _resolve_actor_for_event(
            event_index=idx, ctx=ctx, fallback_candidates=players, strict=False
        )
        if player not in candidate_players:
            return

        pc = contribs.get(player, Decimal(0))
        to_call = current_bet - pc
        fold_allowed = player not in non_folders

        if to_call == 0:
            hypothesis_order = ["check", "call", "raise"]
        else:
            hypothesis_order = ["call", "raise", "check"]

        if fold_allowed:
            hypothesis_order.append("fold")

        for action in hypothesis_order:
            if capped:
                return
            if not _action_is_fsm_legal(player, action, ctx):
                continue

            if action == "fold":
                seq.append(InferredAction(player, "fold", Decimal(0)))
                dfs(idx + 1, remaining, current_bet, last_raise, contribs, seq)
                seq.pop()

            elif action == "check":
                if to_call == 0:
                    seq.append(InferredAction(player, "check", Decimal(0)))
                    dfs(idx + 1, remaining, current_bet, last_raise, contribs, seq)
                    seq.pop()

            elif action == "call":
                if 0 < to_call <= remaining:
                    new_contribs = dict(contribs)
                    new_contribs[player] = pc + to_call
                    seq.append(InferredAction(player, "call", to_call))
                    dfs(idx + 1, remaining - to_call, current_bet, last_raise, new_contribs, seq)
                    seq.pop()

            elif action == "raise":
                raise_options = _legal_raise_contributions(
                    current_bet=current_bet,
                    last_raise=last_raise,
                    player_contribution=pc,
                    remaining_delta=remaining,
                    pot=ctx.pot,
                    cfg=cfg,
                )
                for contribution in raise_options:
                    if capped:
                        return
                    new_raise_to = pc + contribution
                    new_last_raise = new_raise_to - current_bet
                    new_contribs = dict(contribs)
                    new_contribs[player] = new_raise_to

                    action_kind: ActionKind = "raise"
                    if (
                        current_bet > 0
                        and contribution >= cfg.all_in_contribution_multiplier * current_bet
                        and contribution == remaining
                    ):
                        action_kind = "all_in"
                    elif current_bet == 0:
                        action_kind = "bet"

                    seq.append(InferredAction(player, action_kind, contribution))
                    dfs(idx + 1, remaining - contribution, new_raise_to, new_last_raise, new_contribs, seq)
                    seq.pop()

    dfs(0, delta_pot, ctx.current_bet, ctx.last_raise_size, contribs0, [])
    return EnumerationResult(
        sequences=results,
        enumeration_capped=capped,
        raw_count_before_cap=len(results),
    )


def _canonical_key(sequence: list[InferredAction]) -> tuple:
    """Canonical equivalence key for a sequence (per-player outcomes match)."""
    # Usamos .normalize() para garantir que Decimal("2") e Decimal("2.00") têm a mesma chave
    return tuple(sorted((a.player_id, a.action, str(a.amount.normalize())) for a in sequence))


def _deduplicate_sequences(
    sequences: list[list[InferredAction]],
) -> tuple[list[list[InferredAction]], int]:
    """Deduplicate sequences by canonical per-player outcome."""
    seen: dict[tuple, int] = {}
    unique: list[list[InferredAction]] = []
    collisions = 0
    for seq in sequences:
        key = _canonical_key(seq)
        if key in seen:
            collisions += 1
            continue
        seen[key] = len(unique)
        unique.append(seq)
    return unique, collisions


def _lexicographic_seat_key(sequence: list[InferredAction]) -> tuple:
    """Deterministic tiebreaker key for primary selection."""
    return tuple((a.player_id, a.action, str(a.amount.normalize())) for a in sequence)


# ---------------------------------------------------------------------------
# 5. Scoring
# ---------------------------------------------------------------------------


def _action_prior(action: InferredAction, ctx: TableContext, cfg: InferencerConfig) -> float:
    table = cfg.action_priors.get(ctx.current_street, {})
    # Treat bet/all_in as raise-family for priors.
    key: ActionKind = action.action
    if key in ("bet", "all_in"):
        key = "raise"
    return _safe_log(table.get(key, 0.05))


def _sizing_prior(action: InferredAction, ctx: TableContext, cfg: InferencerConfig) -> float:
    if action.action not in ("bet", "raise", "all_in"):
        return 0.0
    if ctx.pot <= 0:
        return 0.0
    pot_frac = float(action.amount) / float(ctx.pot)
    distance = min(abs(pot_frac - c) for c in cfg.pot_fraction_grid)
    return _safe_log(math.exp(-distance * cfg.sizing_sharpness))


def _occams_penalty(seq: list[InferredAction], cfg: InferencerConfig) -> float:
    non_fold = sum(1 for a in seq if a.action != "fold")
    raises = sum(1 for a in seq if a.action in ("bet", "raise", "all_in"))

    penalty = cfg.occam_linear_weight * max(0, non_fold, -1)
    penalty += cfg.occam_quadratic_weight * max(0, non_fold - 1) ** 2
    penalty += cfg.rebrace_penalty * max(0, raises - 1)
    return -penalty


def _sequence_coherence(seq: list[InferredAction]) -> float:
    bonus = 0.0
    raised = False
    for a in seq:
        if a.action in ("bet", "raise", "all_in"):
            if raised:
                bonus -= 0.2  # 3-bets/4-bets are less common
            raised = True
    return bonus


def score_sequence(seq: list[InferredAction], ctx: TableContext, cfg: InferencerConfig) -> float:
    score = 0.0
    for a in seq:
        score += _action_prior(a, ctx, cfg)
        score += _sizing_prior(a, ctx, cfg)
    score += _occams_penalty(seq, cfg)
    score += _sequence_coherence(seq)
    return score


# ---------------------------------------------------------------------------
# 6. Confidence
# ---------------------------------------------------------------------------


def compute_window_confidence(
    chosen_score: float,
    all_scores: list[float],
    pot_estimator_quality: float,
    chosen_seq: list[InferredAction],
    cfg: InferencerConfig,
) -> float:
    if len(all_scores) <= 1:
        margin_factor = 1.0
    else:
        runner_up = sorted(all_scores, reverse=True)[1]
        margin_factor = _sigmoid((chosen_score - runner_up) * cfg.confidence_margin_gain)
    cv_factor = _clip(pot_estimator_quality, 0.0, 1.0)
    non_fold = sum(1 for a in chosen_seq if a.action != "fold")
    complexity_factor = max(0.4, 1.0 - 0.1 * (non_fold - 1))
    conf = margin_factor * cv_factor * complexity_factor
    return _clip(conf, cfg.min_window_confidence, 1.0)


def per_action_confidence(action: InferredAction, window_conf: float, cfg: InferencerConfig) -> float:
    modifier = cfg.action_confidence_modifier.get(action.action, 0.0)
    return _clip(window_conf + modifier, cfg.min_action_confidence, 1.0)


# ---------------------------------------------------------------------------
# 7. The Inferencer
# ---------------------------------------------------------------------------


class OpponentActionInferencer:
    """
    Stateless-per-hand inferencer. Maintains a single pending anchor between
    calls; the FSM owns all poker state.
    """

    def __init__(self, config: Optional[InferencerConfig] = None) -> None:
        self.cfg = config or InferencerConfig()
        self._pending: Optional[AnchorEvent] = None
        self._pot_history: deque[tuple[float, Decimal]] = deque(maxlen=600)

    # ---- FSM-facing API ----

    def reset(self) -> None:
        """Call at the start of every new hand."""
        self._pending = None
        self._pot_history.clear()

    def observe_pot(self, timestamp: float, pot: Decimal) -> None:
        """Optional: feed continuous pot readings for variance estimation."""
        self._pot_history.append((timestamp, pot))

    def on_anchor(
        self, anchor: AnchorEvent, ctx: TableContext, non_folders: frozenset[str] = frozenset()
    ) -> list[InferredAction]:
        """
        Primary entrypoint. Returns the actions attributed to the window
        ending at `anchor`. The first anchor in a hand returns an empty list
        and is stored as the open window.
        """

        snapshot = ctx.snapshot_constraints()
        try:
            if self._pending is None:
                self._pending = anchor
                return []

            try:
                inferred = self._attribute_window(self._pending, anchor, ctx, non_folders=non_folders)
            finally:
                self._pending = anchor
            return inferred

        finally:
            ctx.assert_constraints_match(snapshot, where="on_anchor")

    # ---- Internals ----

    def _sanitized_delta(self, start: AnchorEvent, end: AnchorEvent, ctx: TableContext) -> Decimal:
        delta = end.pot_before - start.pot_after
        threshold = max(
            self.cfg.pot_noise_abs,
            Decimal(str(self.cfg.pot_noise_rel)) * ctx.pot,
        )
        if abs(delta) < threshold:
            return Decimal(0)
        # Negative deltas should never happen except via CV noise.
        return max(delta, Decimal(0))

    def _window_players(self, start: AnchorEvent, end: AnchorEvent, ctx: TableContext) -> list[str]:
        """Players (excluding Hero) whose turn fell inside the window."""
        # The window starts strictly *after* the player who triggered `start`.
        # By FSM convention, ctx.turn_pointer is whose turn it is at the
        # *start* of the window (i.e. immediately after `start` happened).
        if end.anchor_type == AnchorType.HERO_ACTION:
            end_marker = ctx.hero_id
        else:
            # End at the next-to-act after the last villain; using hero as a
            # sentinel works whenever the FSM wraps the window cleanly.
            end_marker = ctx.hero_id

        # We want players from turn_pointer (inclusive) up to (but not
        # including) end_marker, all from active opponents.
        if ctx.turn_pointer is None:
            return []
        start_after = self._prev_in_order(ctx.turn_pointer, ctx.seat_order)
        players = _players_between(
            start_after=start_after,
            end_before=end_marker,
            seat_order=ctx.seat_order,
            active=ctx.active_players,
        )
        return [p for p in players if p != ctx.hero_id]

    @staticmethod
    def _prev_in_order(player: str, seat_order: list[str]) -> str:
        i = seat_order.index(player)
        return seat_order[(i - 1) % len(seat_order)]

    def _estimator_quality(self, start: AnchorEvent, end: AnchorEvent) -> float:
        return min(start.pot_estimator_quality, end.pot_estimator_quality)

    def _all_zero_delta(self, players: list[str], ctx: TableContext) -> list[InferredAction]:
        out: list[InferredAction] = []
        for p in players:
            pc = ctx.contributions_this_street.get(p, Decimal(0))
            is_matched = pc == ctx.current_bet
            if is_matched:
                out.append(
                    InferredAction(
                        player_id=p,
                        action="check",
                        amount=Decimal(0),
                        confidence=self.cfg.confidence_clean_check,
                        rationale="zero delta, player matched to current bet",
                    )
                )
            else:
                out.append(
                    InferredAction(
                        player_id=p,
                        action="fold",
                        amount=Decimal(0),
                        confidence=self.cfg.confidence_inferred_fold,
                        rationale="zero delta, player owes but did not contribute",
                    )
                )
        return out

    def _fallback(self, players: list[str], delta: Decimal, ctx: TableContext) -> list[InferredAction]:
        """No legal sequence found — emit a single 'unknown bet' attributed
        to the first player and flag low confidence. This should be rare."""
        if not players:
            return []

        if abs(delta) <= self.cfg.zero_delta_epsilon:
            return self._all_zero_delta(players, ctx)

        is_opening = ctx.current_bet == 0
        head, *rest = players

        out = [
            InferredAction(
                player_id=head,
                action="bet" if is_opening else "raise",
                amount=delta,
                confidence=self.cfg.confidence_fallback_action,
                rationale="fallback: unnattributed delta, attributed to first-to-act",
                metadata={"needs_review": True, "fallback_reason": "no_legal_sequence"},
            )
        ]
        for p in rest:
            out.append(
                InferredAction(
                    player_id=p,
                    action="fold",
                    amount=Decimal(0),
                    confidence=self.cfg.confidence_fallback_fold,
                    rationale="fallback: assumed fold after unattributed action",
                    metadata={"needs_review": True},
                )
            )
        return out

    def _explain(
        self, chosen: list[InferredAction], scores: list[tuple[list[InferredAction], float]], delta: Decimal
    ) -> str:
        scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
        top_score = scores_sorted[0][1]
        if len(scores_sorted) > 1:
            margin = top_score - scores_sorted[1][1]
            return (
                f"Δpot=${delta}; chosen score={top_score:.3f}, "
                f"margin over runner-up={margin:.3f}; "
                f"candidates={len(scores_sorted)}"
            )
        return f"Δpot=${delta}; unique legal sequence; score={top_score:.3f}"

    def _rank_and_select_primary(
        self,
        scored_sequences: list[tuple[list[InferredAction], float]],
    ) -> tuple[list[InferredAction], list[list[InferredAction]], float, float]:
        if not scored_sequences:
            return [], [], 0.0, 0.0

        sorted_pairs = sorted(
            scored_sequences,
            key=lambda pair: (-pair[1], _lexicographic_seat_key(pair[0])),
        )

        primary, primary_score = sorted_pairs[0]
        alternatives = [seq for seq, _ in sorted_pairs[1:]]
        runner_up_score = sorted_pairs[1][1] if len(sorted_pairs) > 1 else primary_score - 1.0

        return primary, alternatives, primary_score, runner_up_score

    def _compute_primary_confidence(
        self,
        primary_score: float,
        runner_up_score: float,
        cfg: InferencerConfig,
        pre_constraint_count: int,
        post_constraint_count: int,
    ) -> tuple[float, bool]:
        margin = max(0.0, primary_score - runner_up_score)
        was_reconciled = pre_constraint_count > post_constraint_count

        if was_reconciled:
            pruning_ratio = 1.0 - (post_constraint_count / pre_constraint_count)
            margin += getattr(cfg, "constraint_pruning_margin_weight", 1.5) * pruning_ratio

        confidence = min(
            getattr(cfg, "confidence_dfs_max", 0.90),
            getattr(cfg, "confidence_dfs_base", 0.55) + getattr(cfg, "confidence_dfs_margin_weight", 0.35) * margin,
        )

        if was_reconciled:
            confidence -= getattr(cfg, "reconciliation_dampening", 0.15)
            confidence = max(0.1, confidence)

        return confidence, was_reconciled

    def _apply_confidence_to_primary(
        self,
        primary: list[InferredAction],
        confidence: float,
        cfg: InferencerConfig,
    ) -> None:
        forced_threshold = getattr(cfg, "confidence_forced_action", 0.97)
        for action in primary:
            if getattr(action, "confidence", 0.0) < forced_threshold:
                action.confidence = confidence

    def _attribute_window_full(
        self,
        start: AnchorEvent,
        end: AnchorEvent,
        ctx: TableContext,
        non_folders: frozenset[str] = frozenset(),
    ) -> WindowAttribution:
        delta = self._sanitized_delta(start, end, ctx)
        players = self._window_players(start, end, ctx)

        if not players:
            return WindowAttribution(primary=[], alternatives=[])

        if delta == 0 or abs(delta) <= getattr(self.cfg, "zero_delta_epsilon", Decimal("0.5")):
            zero_delta_primary = self._all_zero_delta(players, ctx)
            return WindowAttribution(primary=zero_delta_primary, alternatives=[])

        unconstrained = enumerate_action_sequences(
            players,
            delta,
            ctx,
            self.cfg,
            frozenset(),
            max_alternatives=10_000,
        )
        unconstrained_count = len(unconstrained.sequences)

        constrained = enumerate_action_sequences(players, delta, ctx, self.cfg, non_folders=non_folders)

        constraint_relaxed = False
        if not constrained.sequences and non_folders:
            constrained = enumerate_action_sequences(players, delta, ctx, self.cfg, non_folders=frozenset())
            constraint_relaxed = True

        if not constrained.sequences:
            fallback = self._fallback(players, delta, ctx)
            return WindowAttribution(primary=fallback, alternatives=[])

        deduped, collision_count = _deduplicate_sequences(constrained.sequences)
        high_collision = len(constrained.sequences) > 0 and (collision_count / len(constrained.sequences)) > 0.5

        scored = [(seq, score_sequence(seq, ctx, self.cfg)) for seq in deduped]

        sorted_pairs = sorted(scored, key=lambda pair: (-pair[1], _lexicographic_seat_key(pair[0])))

        raw_scores = [pair[1] for pair in sorted_pairs]
        max_score = raw_scores[0] if raw_scores else 0.0
        exp_scores = [math.exp(s - max_score) for s in raw_scores]
        sum_exp = sum(exp_scores)
        weights = [es / sum_exp for es in exp_scores] if sum_exp > 0 else []

        primary, alternatives, primary_score, runner_up_score = self._rank_and_select_primary(scored)

        confidence, was_reconciled = self._compute_primary_confidence(
            primary_score=primary_score,
            runner_up_score=runner_up_score,
            cfg=self.cfg,
            pre_constraint_count=unconstrained_count,
            post_constraint_count=len(constrained.sequences),
        )
        self._apply_confidence_to_primary(primary, confidence, self.cfg)

        rationale = self._explain(primary, scored, delta)
        if constraint_relaxed:
            rationale += " [CONSTRAINT RELAXED: Over-pruned]"
        elif was_reconciled:
            rationale += " [RECONCILED: DFS pruned via Future Showdown]"

        for action in primary:
            action.rationale = rationale
            if not hasattr(action, "metadata"):
                action.metadata = {}
            if was_reconciled:
                action.metadata["reconciled_via_showdown"] = True
            if constraint_relaxed:
                action.metadata["constraint_relaxed"] = True

        return WindowAttribution(
            primary=primary,
            alternatives=alternatives,
            weights=weights,
            enumeration_capped=constrained.enumeration_capped,
            deduplication_high_collision=high_collision,
            was_reconciled=was_reconciled,
            constraint_relaxed=constraint_relaxed,
        )

    def _attribute_window(
        self,
        start: AnchorEvent,
        end: AnchorEvent,
        ctx: TableContext,
        non_folders: frozenset[str] = frozenset(),
    ) -> list[InferredAction]:
        """Legacy entry point. Returns primary only for FSM compatibility."""
        return self._attribute_window_full(start, end, ctx, non_folders=non_folders).primary

    def on_anchor_with_alternatives(
        self,
        anchor: AnchorEvent,
        ctx: TableContext,
        non_folders: frozenset[str] = frozenset(),
    ) -> Optional[WindowAttribution]:
        """Like on_anchor, but returns the full WindowAttribution including alternatives."""
        snapshot = ctx.snapshot_constraints()
        try:
            if self._pending is None:
                self._pending = anchor
                return None

            try:
                attribution = self._attribute_window_full(self._pending, anchor, ctx, non_folders=non_folders)
            finally:
                self._pending = anchor
            return attribution
        finally:
            ctx.assert_constraints_match(snapshot, where="on_anchor_with_alternatives")


# ---------------------------------------------------------------------------
# 8. Convenience: a one-shot helper for tests / replay
# ---------------------------------------------------------------------------


def infer_window(
    start: AnchorEvent,
    end: AnchorEvent,
    ctx: TableContext,
    config: Optional[InferencerConfig] = None,
) -> list[InferredAction]:
    """Stateless one-call helper, primarily for unit tests."""
    inf = OpponentActionInferencer(config)
    inf.on_anchor(start, ctx)
    return inf.on_anchor(end, ctx)


__all__ = [
    "ActionKind",
    "AnchorEvent",
    "AnchorType",
    "HeroAction",
    "InferencerConfig",
    "InferredAction",
    "OpponentActionInferencer",
    "Street",
    "TableContext",
    "enumerate_action_sequences",
    "infer_window",
    "score_sequence",
    "compute_window_confidence",
    "per_action_confidence",
    "EnumerationResult",
    "WindowAttribution",
]
