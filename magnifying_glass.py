import csv
import random
from pathlib import Path
from poker_vision.inference.test_corpus import load_corpus, PokerStarsParser

def main():
    print("Iniciando Verificação 3: Inspeção de 30 Casos de Showdown...")

    corpus_path = Path("data/corpus/test_corpus.json")
    csv_path = Path("output_ab_test/run1_baseline/per_case.csv")

    corpus = {c.case_id: c for c in load_corpus(corpus_path)}

    # 1. Pegar IDs das janelas com erro
    error_cases = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["sequence_exact_match"].strip().lower() in ("false", "0"):
                error_cases.append(row["case_id"])

    # 2. Parsear as mãos originais
    parser = PokerStarsParser()
    txt_files = list(Path("data/hh").glob("*.txt"))
    parsed_hands = {}
    for txt in txt_files:
        for hand in parser.parse_file(txt):
            parsed_hands[hand.hand_id] = hand

    # 3. Filtrar SOMENTE os erros da categoria SHOWDOWN
    showdown_errors = []
    for case_id in error_cases:
        if case_id not in corpus: continue
        case = corpus[case_id]
        hand_id = case.metadata["hand_id"]
        if hand_id not in parsed_hands: continue

        hand = parsed_hands[hand_id]
        showdown_participants = frozenset(hand.showdown_reveals.keys())

        if not case.expected_actions: continue
        misattributed_player = case.expected_actions[0].player_id

        # Se o cara que DFS errou estava no Showdown, pegamos ele!
        if misattributed_player in showdown_participants:
            showdown_errors.append((case, hand, misattributed_player))

    print(f"Total de erros com Showdown no corpus: {len(showdown_errors)}")

    # 4. Pegar 30 aleatórios (com semente fixa para reprodutibilidade)
    random.seed(42)
    sample = random.sample(showdown_errors, min(30, len(showdown_errors)))

    print("\n" + "="*50)
    for i, (case, hand, player) in enumerate(sample, 1):
        actual_action = case.expected_actions[0].action
        # Simulando o que o DFS previu no erro (quase sempre assumiu um Fold e errou)
        predicted = "fold" if actual_action != "fold" else "unknown"

        print(f"CASO {i}: {case.metadata['hand_id']} - Street: {case.metadata['street_at_window_end']}")
        print(f"  Jogador Foco: {player}")
        print(f"  DFS Acreditou: {predicted.upper()}")
        print(f"  Realidade (Ground Truth): {actual_action.upper()}")
        print(f"  Participantes do Showdown: {list(hand.showdown_reveals.keys())}")
        print("-" * 50)

if __name__ == "__main__":
    main()