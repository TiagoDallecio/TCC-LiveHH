import csv
from collections import Counter
from pathlib import Path
from enum import Enum

# Importações do nosso classificador anterior
from poker_vision.corpus.ambiguity_subclassifier import (
    ErrorMode, classify_structure, classify_error_mode_sequence
)
from poker_vision.inference.test_corpus import load_corpus
from poker_vision.inference.opponent_action_inferencer import OpponentActionInferencer

class BindingFailureProxy(Enum):
    ZONE_BOUNDARY_PROXIMITY = "zone_boundary_proximity"
    OCCLUSION_FLAG_SET = "occlusion_flag_set"
    TEMPORAL_EDGE = "temporal_edge"
    MULTI_MOTION_WINDOW = "multi_motion_window"
    HOMOGRAPHY_DEGRADED = "homography_degraded"
    NO_PROXY_FIRED = "no_proxy_fired"

def get_proxy_for_window(predicted_actions, actual_actions) -> BindingFailureProxy:
    """
    Tenta deduzir a proxy de falha.
    Nota: Proxies de CV não dispararão porque nosso corpus é gerado a partir de .txt
    """
    motion_actions = frozenset({"bet", "raise", "call", "all_in"})

    # 1. Multi-Motion Window: Duas ou mais ações de movimento na mesma janela
    actual_motions = [a for a in actual_actions if a["action"] in motion_actions]
    if len(actual_motions) > 1:
        return BindingFailureProxy.MULTI_MOTION_WINDOW

    # 2. Temporal Edge: Ações muito próximas no índice da sequência
    # Como não temos timestamps de frames, usamos a sequência lógica
    if actual_actions and predicted_actions:
        # Se a sequência prevista for muito diferente em tamanho, pode indicar corte temporal
        if len(predicted_actions) != len(actual_actions):
            return BindingFailureProxy.TEMPORAL_EDGE

    # 3. Proxies de CV (Sempre falso no dataset de texto atual)
    # return BindingFailureProxy.ZONE_BOUNDARY_PROXIMITY
    # return BindingFailureProxy.OCCLUSION_FLAG_SET
    # return BindingFailureProxy.HOMOGRAPHY_DEGRADED

    # 4. Fallback
    return BindingFailureProxy.NO_PROXY_FIRED

def main():
    print("Iniciando Calibração de Proxies de Falha de Atribuição...")

    corpus_path = Path("data/corpus/test_corpus.json")
    csv_path = Path("output_ab_test/run1_baseline/per_case.csv")
    corpus = {c.case_id: c for c in load_corpus(corpus_path)}

    error_cases = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["sequence_exact_match"].strip().lower() in ("false", "0"):
                error_cases.append(row["case_id"])

    proxy_distribution = Counter()
    cell_distribution = Counter()

    print(f"Buscando erros de Atribuição Cruzada...")

    for case_id in error_cases:
        if case_id not in corpus: continue
        case = corpus[case_id]
        if not case.expected_actions: continue

        inf = OpponentActionInferencer()
        inf.on_anchor(case.anchor_start, case.ctx_before)
        raw_pred = inf.on_anchor(case.anchor_end, case.ctx_before, non_folders=frozenset(getattr(case, "non_folders", [])))

        actual_actions = [{"player": a.player_id, "action": a.action} for a in case.expected_actions]
        predicted_actions = [{"player": pa.player_id, "action": pa.action} for pa in raw_pred]

        error_mode = classify_error_mode_sequence(predicted_actions=predicted_actions, actual_actions=actual_actions)

        # SÓ QUEREMOS O BUCKET DE ATRIBUIÇÃO CRUZADA (Os 1.525 casos)
        if error_mode != ErrorMode.CROSS_PLAYER_ATTRIBUTION:
            continue

        street = case.metadata["street_at_window_end"]
        num_active = len(case.metadata.get("active_players", [])) if case.metadata.get("active_players") else 2
        current_bet = float(case.ctx_before.current_bet)
        is_terminal = (actual_actions[0]["action"] == "fold")

        structure = classify_structure(street=street, num_active_players=num_active, current_bet_before_window=current_bet, is_terminal_fold=is_terminal, is_first_fold_of_street=False)

        proxy = get_proxy_for_window(predicted_actions, actual_actions)

        proxy_distribution[proxy] += 1
        cell_distribution[(structure, proxy)] += 1

    print("\n" + "="*50)
    print("DISTRIBUIÇÃO GERAL DAS PROXIES (Apenas Cross-Player)")
    print("="*50)
    total_cross = sum(proxy_distribution.values())
    for proxy, count in proxy_distribution.most_common():
        print(f"{proxy.name}: {count} ({(count/total_cross)*100:.1f}%)")

    print("\n" + "="*50)
    print("TOP 5 CÉLULAS ESTRATIFICADAS (Estrutura x Proxy)")
    print("="*50)
    for (struct, proxy), count in cell_distribution.most_common(5):
        print(f"[{struct.value}] x [{proxy.name}]: {count}")

if __name__ == "__main__":
    main()