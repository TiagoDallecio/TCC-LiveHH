from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum


class HandStructure(Enum):
    PREFLOP_FOLD_TO_OPEN = "preflop_fold_to_open"
    PREFLOP_MULTIWAY = "preflop_multiway"
    FLOP_CHECK_AROUND = "flop_check_around"
    POSTFLOP_HEADS_UP = "postflop_heads_up"
    POSTFLOP_MULTIWAY = "postflop_multiway"
    LATE_STREET_FOLD = "late_street_fold"
    OTHER = "other"


class ConfusedActionPair(Enum):
    FOLD_VS_CHECK = "fold_vs_check"
    FOLD_VS_CALL = "fold_vs_call"
    FOLD_VS_BET = "fold_vs_bet"
    FOLD_VS_RAISE = "fold_vs_raise"
    CHECK_VS_BET = "check_vs_bet"
    CALL_VS_RAISE = "call_vs_raise"
    BET_VS_RAISE = "bet_vs_raise"
    CHECK_VS_CALL = "check_vs_call"
    OTHER = "other"


class ErrorMode(Enum):
    SAME_PLAYER_MAGNITUDE = "same_player_magnitude"
    CROSS_PLAYER_ATTRIBUTION = "cross_player_attribution"
    AMBIGUOUS_NO_MOTION = "ambiguous_no_motion"


@dataclass(frozen=True)
class SubclassifiedError:
    hand_id: str
    window_id: str
    structure: HandStructure
    action_pair: ConfusedActionPair
    error_mode: ErrorMode
    predicted_kind: str
    actual_kind: str
    predicted_player: str
    actual_player: str
    num_active_players: int
    street: str


def classify_structure(
    *,
    street: str,
    num_active_players: int,
    current_bet_before_window: float,
    is_terminal_fold: bool,
    is_first_fold_of_street: bool,
) -> HandStructure:
    if street == "preflop":
        if num_active_players == 2 and current_bet_before_window > 0:
            return HandStructure.PREFLOP_FOLD_TO_OPEN
        if num_active_players >= 3:
            return HandStructure.PREFLOP_MULTIWAY
        return HandStructure.PREFLOP_FOLD_TO_OPEN
    if street == "flop":
        if current_bet_before_window == 0:
            return HandStructure.FLOP_CHECK_AROUND
        if num_active_players == 2:
            return HandStructure.POSTFLOP_HEADS_UP
        return HandStructure.POSTFLOP_MULTIWAY
    if street in ("turn", "river"):
        if is_terminal_fold:
            return HandStructure.LATE_STREET_FOLD
        if num_active_players == 2:
            return HandStructure.POSTFLOP_HEADS_UP
        return HandStructure.POSTFLOP_MULTIWAY
    return HandStructure.OTHER


def classify_action_pair(*, predicted_kind: str, actual_kind: str) -> ConfusedActionPair:
    pair = frozenset({predicted_kind, actual_kind})
    mapping = {
        frozenset({"fold", "check"}): ConfusedActionPair.FOLD_VS_CHECK,
        frozenset({"fold", "call"}): ConfusedActionPair.FOLD_VS_CALL,
        frozenset({"fold", "bet"}): ConfusedActionPair.FOLD_VS_BET,
        frozenset({"fold", "raise"}): ConfusedActionPair.FOLD_VS_RAISE,
        frozenset({"check", "bet"}): ConfusedActionPair.CHECK_VS_BET,
        frozenset({"call", "raise"}): ConfusedActionPair.CALL_VS_RAISE,
        frozenset({"bet", "raise"}): ConfusedActionPair.BET_VS_RAISE,
        frozenset({"check", "call"}): ConfusedActionPair.CHECK_VS_CALL,
    }
    return mapping.get(pair, ConfusedActionPair.OTHER)


def classify_error_mode(
    *, predicted_kind: str, actual_kind: str, predicted_player: str, actual_player: str
) -> ErrorMode:
    no_motion = frozenset({"fold", "check"})
    if predicted_kind in no_motion and actual_kind in no_motion:
        return ErrorMode.AMBIGUOUS_NO_MOTION
    if predicted_player != actual_player:
        return ErrorMode.CROSS_PLAYER_ATTRIBUTION
    return ErrorMode.SAME_PLAYER_MAGNITUDE


def build_3d_crosstab(
    errors: list[SubclassifiedError],
) -> dict[ErrorMode, dict[tuple[HandStructure, ConfusedActionPair], int]]:
    result = {mode: Counter() for mode in ErrorMode}
    for err in errors:
        result[err.error_mode][(err.structure, err.action_pair)] += 1
    return {k: dict(v) for k, v in result.items()}


def render_crosstab_markdown(crosstab: dict[tuple[HandStructure, ConfusedActionPair], int], *, total: int) -> str:
    structures = list(HandStructure)
    pairs = list(ConfusedActionPair)
    header = "| Structure ↓ / Pair → | " + " | ".join(p.value for p in pairs) + " | Row total |"
    sep = "|" + "---|" * (len(pairs) + 2)
    rows = [header, sep]
    for s in structures:
        row_total = sum(crosstab.get((s, p), 0) for p in pairs)
        if row_total == 0:
            continue
        cells = [str(crosstab.get((s, p), 0)) if crosstab.get((s, p), 0) > 0 else "" for p in pairs]
        rows.append(f"| {s.value} | " + " | ".join(cells) + f" | {row_total} |")
    col_totals = [sum(crosstab.get((s, p), 0) for s in structures) for p in pairs]
    rows.append("| **Column total** | " + " | ".join(str(t) if t > 0 else "" for t in col_totals) + f" | **{total}** |")
    return "\n".join(rows)


def classify_error_mode_sequence(
    *,
    predicted_actions: list[dict],  # [{player: str, action: str}]
    actual_actions: list[dict],  # [{player: str, action: str}]
) -> ErrorMode:
    """
    Nova lógica de classificação livre da máscara de ordem do DFS.
    """
    # 1. Extrair quais ações com movimento de fichas aconteceram
    motion_actions = frozenset({"bet", "raise", "call", "all_in"})

    pred_motion = {pa["player"]: pa["action"] for pa in predicted_actions if pa["action"] in motion_actions}
    actual_motion = {aa["player"]: aa["action"] for aa in actual_actions if aa["action"] in motion_actions}

    # 2. Verificação de No-Motion (Todos deram fold/check em ambos os lados)
    if not pred_motion and not actual_motion:
        return ErrorMode.AMBIGUOUS_NO_MOTION

    # 3. Verificação de Atribuição Cruzada Verdadeira
    # Se uma pessoa que fez movimento na realidade não tem movimento previsto (ou vice-versa)
    # ou se os atores de movimento são diferentes, o sistema atribuiu o evento à pessoa errada.
    pred_actors = set(pred_motion.keys())
    actual_actors = set(actual_motion.keys())

    if pred_actors != actual_actors:
        return ErrorMode.CROSS_PLAYER_ATTRIBUTION

    # 4. Verificação de Magnitude
    # Os mesmos atores agiram (ex: Vilão 1), mas as ações divergiram (ex: Call vs Raise)
    return ErrorMode.SAME_PLAYER_MAGNITUDE
