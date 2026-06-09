import csv
import random
from collections import defaultdict
from pathlib import Path

# Importações do nosso classificador anterior
from poker_vision.corpus.ambiguity_subclassifier import (
    HandStructure, ErrorMode, classify_structure, classify_error_mode_sequence
)
from poker_vision.inference.test_corpus import load_corpus
from poker_vision.inference.opponent_action_inferencer import OpponentActionInferencer
from run_proxy_calibration import get_proxy_for_window, BindingFailureProxy

def main():
    print("Iniciando Verificação C: Amostragem Estratificada e Enumeração...\n")

    corpus_path = Path("data/corpus/test_corpus.json")
    csv_path = Path("output_ab_test/run1_baseline/per_case.csv")
    corpus = {c.case_id: c for c in load_corpus(corpus_path)}

    error_cases = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["sequence_exact_match"].strip().lower() in ("false", "0"):
                error_cases.append(row["case_id"])

    strata_buckets = defaultdict(list)

    print("Classificando o corpus para a amostragem...")
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
        if error_mode != ErrorMode.CROSS_PLAYER_ATTRIBUTION:
            continue

        street = case.metadata["street_at_window_end"]
        num_active = len(case.metadata.get("active_players", [])) if case.metadata.get("active_players") else 2
        current_bet = float(case.ctx_before.current_bet)
        is_terminal = (actual_actions[0]["action"] == "fold")

        structure = classify_structure(street=street, num_active_players=num_active, current_bet_before_window=current_bet, is_terminal_fold=is_terminal, is_first_fold_of_street=False)
        proxy = get_proxy_for_window(predicted_actions, actual_actions)

        strata_buckets[(structure, proxy)].append({
            "case": case,
            "actual": actual_actions,
            "predicted": predicted_actions
        })

    # Definição das cotas da amostragem estratificada
    strata_targets = {
        (HandStructure.POSTFLOP_HEADS_UP, BindingFailureProxy.MULTI_MOTION_WINDOW): 10,
        (HandStructure.PREFLOP_FOLD_TO_OPEN, BindingFailureProxy.NO_PROXY_FIRED): 10,
        (HandStructure.POSTFLOP_HEADS_UP, BindingFailureProxy.NO_PROXY_FIRED): 8,
        (HandStructure.POSTFLOP_HEADS_UP, BindingFailureProxy.TEMPORAL_EDGE): 6,
        (HandStructure.PREFLOP_FOLD_TO_OPEN, BindingFailureProxy.MULTI_MOTION_WINDOW): 6
    }

    sampled_cases = []
    random.seed(42) # Reprodutibilidade

    for key, target in strata_targets.items():
        pool = strata_buckets.get(key, [])
        sample_size = min(target, len(pool))
        sampled_cases.extend([(key, item) for item in random.sample(pool, sample_size)])

    print(f"\nAmostra gerada com {len(sampled_cases)} casos. Gerando dossiê do Tier 1...\n")
    print("="*70)

    # Gerando o relatório para a Revisão Manual / Enumeração
    for i, (stratum_key, data) in enumerate(sampled_cases, 1):
        case = data["case"]
        actual = data["actual"]
        predicted = data["predicted"]
        ctx = case.ctx_before

        struct_name = stratum_key[0].name
        proxy_name = stratum_key[1].name

        print(f"CASO {i}/40 | ID: {case.metadata['hand_id']} | Janela: {case.case_id}")
        print(f"Estrato: [{struct_name}] x [{proxy_name}]")
        print("-" * 70)
        print(">> TIER 1: ESTADO LÓGICO DA MESA ANTES DA JANELA")
        print(f"   - Street: {case.metadata['street_at_window_end']}")
        print(f"   - Pote Inicial: {ctx.pot}")
        print(f"   - Aposta Atual (Current Bet): {ctx.current_bet}")
        print(f"   - Ordem de Ação (Turn Order): {ctx.action_order}")
        if hasattr(ctx, 'player_stacks'):
            print(f"   - Stacks Ativos: {ctx.player_stacks}")
        print("-" * 70)
        print(">> O QUE O DFS ESCOLHEU (Ação Logicamente Válida pelo Algoritmo):")
        for p in predicted: print(f"   [ {p['player']} ] -> {p['action'].upper()}")
        print("\n>> GROUND TRUTH (A Realidade do PokerStars):")
        for a in actual: print(f"   [ {a['player']} ] -> {a['action'].upper()}")
        print("="*70)

if __name__ == "__main__":
    main()