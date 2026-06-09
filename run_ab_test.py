import copy
from pathlib import Path
from unittest.mock import patch

from poker_vision.inference import opponent_action_inferencer
from poker_vision.inference.opponent_action_inferencer import reset_inference_metrics, get_inference_metrics
from poker_vision.inference.evaluation.harness import run_evaluation
from poker_vision.inference.test_corpus import load_corpus

def inject_action_order(corpus):
    """Injeta a ordem de ação sintética baseada no turn_pointer e assentos."""
    injected_corpus = copy.deepcopy(corpus)
    for case in injected_corpus:
        ctx = case.ctx_before
        if not ctx.turn_pointer or ctx.turn_pointer not in ctx.active_players:
            ctx.action_order = ()
            continue

        idx = ctx.seat_order.index(ctx.turn_pointer)
        ordered = []
        for i in range(len(ctx.seat_order)):
            p = ctx.seat_order[(idx + i) % len(ctx.seat_order)]
            if p in ctx.active_players:
                ordered.append(p)

        ctx.action_order = tuple(ordered)
        ctx.legal_actions_per_player = {p: frozenset({"fold", "check", "call", "bet", "raise", "all_in"}) for p in ordered}
    return injected_corpus

def main():
    corpus_path = Path("data/corpus/test_corpus.json")
    print(f"Carregando corpus de: {corpus_path}")
    original_corpus = load_corpus(corpus_path)

    # ---------------------------------------------------------
    # RODADA 1: BASELINE (Flag Desligada, Sem Injeção)
    # ---------------------------------------------------------
    print("\n=== RODADA 1: BASELINE ===")
    reset_inference_metrics()
    with patch.object(opponent_action_inferencer, "ENABLE_FSM_HARD_PRUNING", False):
        out1 = run_evaluation(original_corpus, Path("output_ab_test/run1_baseline"))
    print(f"-> Relatórios gerados em: output_ab_test/run1_baseline")

    # ---------------------------------------------------------
    # RODADA 2: DEGRADAÇÃO GRACIOSA (Flag Ligada, Sem Injeção)
    # ---------------------------------------------------------
    print("\n=== RODADA 2: FLAG ON, SEM FSM ===")
    reset_inference_metrics()
    with patch.object(opponent_action_inferencer, "ENABLE_FSM_HARD_PRUNING", True):
        out2 = run_evaluation(original_corpus, Path("output_ab_test/run2_flag_only"))
    metrics2 = get_inference_metrics()
    print(f"-> Relatórios gerados em: output_ab_test/run2_flag_only")
    print(f"-> Métricas de Atrito: {metrics2}")

    # ---------------------------------------------------------
    # RODADA 3: THE MONEY RUN 💰 (Flag Ligada + FSM Sintética)
    # ---------------------------------------------------------
    print("\n=== RODADA 3: THE MONEY RUN (FLAG ON + FSM) ===")
    injected_corpus = inject_action_order(original_corpus)
    reset_inference_metrics()
    with patch.object(opponent_action_inferencer, "ENABLE_FSM_HARD_PRUNING", True):
        out3 = run_evaluation(injected_corpus, Path("output_ab_test/run3_money_run"))
    metrics3 = get_inference_metrics()
    print(f"-> Relatórios gerados em: output_ab_test/run3_money_run")
    print(f"-> Métricas de Atrito: {metrics3}")

    print("\n=== TESTE A/B CONCLUÍDO ===")
    print("Vá até as pastas output_ab_test/run1, run2 e run3 e compare as matrizes de confusão!")

if __name__ == "__main__":
    main()