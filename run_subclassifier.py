import csv
from pathlib import Path

from poker_vision.corpus.ambiguity_subclassifier import (
    SubclassifiedError, classify_structure, classify_action_pair, classify_error_mode_sequence,
    build_3d_crosstab, render_crosstab_markdown
)
from poker_vision.inference.test_corpus import load_corpus
from poker_vision.inference.opponent_action_inferencer import OpponentActionInferencer

def main():
    print("Iniciando Verificação 1 & 2 (Diagnóstico Corrigido de Sequências)...")

    corpus_path = Path("data/corpus/test_corpus.json")
    csv_path = Path("output_ab_test/run1_baseline/per_case.csv")

    corpus = {c.case_id: c for c in load_corpus(corpus_path)}

    error_cases = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["sequence_exact_match"].strip().lower() in ("false", "0"):
                error_cases.append(row["case_id"])

    classified_errors = []

    print(f"Processando {len(error_cases)} erros no DFS...")
    for case_id in error_cases:
        if case_id not in corpus: continue
        case = corpus[case_id]
        if not case.expected_actions: continue

        inf = OpponentActionInferencer()
        inf.on_anchor(case.anchor_start, case.ctx_before)
        raw_pred = inf.on_anchor(case.anchor_end, case.ctx_before, non_folders=frozenset(getattr(case, "non_folders", [])))

        # Mapeando sequências inteiras
        actual_actions = [{"player": a.player_id, "action": a.action} for a in case.expected_actions]
        predicted_actions = [{"player": pa.player_id, "action": pa.action} for pa in raw_pred]

        # Para Action Pair, pegamos a primeira divergência como proxy do par confuso principal
        predicted_kind = "unknown"
        actual_kind = actual_actions[0]["action"]
        for pa in predicted_actions:
            if pa["player"] == actual_actions[0]["player"]:
                predicted_kind = pa["action"]
                break
        if predicted_kind == "unknown": predicted_kind = "fold"

        street = case.metadata["street_at_window_end"]
        num_active = len(case.metadata.get("active_players", []))
        if num_active == 0: num_active = 2
        current_bet = float(case.ctx_before.current_bet)
        is_terminal = (actual_kind == "fold")

        structure = classify_structure(street=street, num_active_players=num_active, current_bet_before_window=current_bet, is_terminal_fold=is_terminal, is_first_fold_of_street=False)
        action_pair = classify_action_pair(predicted_kind=predicted_kind, actual_kind=actual_kind)

        # O FIM DO ARTEFATO DE MEDIÇÃO
        error_mode = classify_error_mode_sequence(predicted_actions=predicted_actions, actual_actions=actual_actions)

        err = SubclassifiedError(hand_id=case.metadata["hand_id"], window_id=case_id, structure=structure, action_pair=action_pair, error_mode=error_mode, predicted_kind=predicted_kind, actual_kind=actual_kind, predicted_player="", actual_player="", num_active_players=num_active, street=street)
        classified_errors.append(err)

    crosstab_3d = build_3d_crosstab(classified_errors)
    total_all = len(classified_errors)

    print("\n" + "="*60)
    for mode, matrix in crosstab_3d.items():
        mode_total = sum(matrix.values())
        print(f"\n### Error Mode: {mode.name} ({mode_total} errors total, {(mode_total/total_all)*100:.1f}%)")
        print(render_crosstab_markdown(matrix, total=mode_total))

if __name__ == "__main__":
    main()