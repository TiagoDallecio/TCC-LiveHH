import csv
from pathlib import Path
from poker_vision.corpus.error_taxonomy import classify_error, summarize_categories
from poker_vision.inference.test_corpus import load_corpus, PokerStarsParser

def main():
    print("Iniciando Diagnóstico de Taxonomia dos 662 Erros...")

    # 1. Carregar o corpus e os resultados da Baseline
    corpus_path = Path("data/corpus/test_corpus.json")
    csv_path = Path("output_ab_test/run1_baseline/per_case.csv")

    corpus = {c.case_id: c for c in load_corpus(corpus_path)}

    # Pegar apenas os IDs das janelas que tiveram erro (Acurácia < 1.0)
    # Focando especificamente nos erros de Fold -> Raise
    error_cases = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Verifica se a sequência NÃO foi um match exato
            if row["sequence_exact_match"].strip().lower() in ("false", "0"):
                error_cases.append(row["case_id"])

    print(f"Encontrados {len(error_cases)} casos com erro no Baseline.")

    # 2. Fazer o parse dos arquivos originais para ter as "Ações do Futuro"
    parser = PokerStarsParser()
    txt_files = list(Path("data/hh").glob("*.txt")) # Ajuste o caminho dos seus .txt se precisar
    parsed_hands = {}
    for txt in txt_files:
        for hand in parser.parse_file(txt):
            parsed_hands[hand.hand_id] = hand

    # 3. Classificar cada erro
    classified_errors = []
    for case_id in error_cases:
        if case_id not in corpus: continue
        case = corpus[case_id]
        hand_id = case.metadata["hand_id"]
        if hand_id not in parsed_hands: continue

        hand = parsed_hands[hand_id]

        # Juntar todas as ações do futuro (após a rua atual)
        future_actions = []
        streets_order = ["preflop", "flop", "turn", "river"]
        current_street_idx = streets_order.index(case.metadata["street_at_window_end"])

        for i in range(current_street_idx + 1, len(streets_order)):
            street_name = streets_order[i]
            future_actions.extend(hand.actions_by_street.get(street_name, []))

        showdown_participants = frozenset(hand.showdown_reveals.keys())
        pot_collector = hand.winners[0] if hand.winners else None

        # Para simplificar, vamos assumir o primeiro vilão da janela como o "misattributed"
        # Em um script de produção faríamos um diff exato, mas isso já nos dá a amostragem
        if not case.expected_actions: continue
        misattributed_player = case.expected_actions[0].player_id

        err = classify_error(
            hand_id=hand_id,
            window_id=case_id,
            misattributed_player=misattributed_player,
            predicted_kind="fold",
            actual_kind="raise",
            hand_actions_after_window=future_actions,
            hand_pot_collector=pot_collector,
            hand_showdown_participants=showdown_participants
        )
        classified_errors.append(err)

    # 4. Imprimir o Resumo
    summary = summarize_categories(classified_errors)
    print("\n=== RESULTADO DA TAXONOMIA ===")
    total = sum(summary.values())
    for cat, count in summary.items():
        pct = (count / total * 100) if total > 0 else 0
        print(f"{cat.name}: {count} ({pct:.1f}%)")

if __name__ == "__main__":
    main()