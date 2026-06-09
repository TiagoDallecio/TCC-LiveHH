from decimal import Decimal

from poker_vision.corpus.synthetic_fsm import HistoryAction, compute_synthetic_state


def run_spot_checks():
    BB = Decimal("10")
    SB = Decimal("5")

    print("=== INICIANDO SPOT CHECKS ===")

    # ---------------------------------------------------------
    # CENÁRIO 1: Preflop com 1 Raise e 1 Call
    # ---------------------------------------------------------
    # UTG faz fold, MP faz raise para 30, CO e BTN fazem fold.
    # De quem é a vez? (Deve ser a Small Blind, que ainda tem de decidir)
    snap1 = compute_synthetic_state(
        seat_order=["SB", "BB", "UTG", "MP", "CO", "BTN"],
        button_player_id="BTN",
        big_blind=BB,
        small_blind=SB,
        actions_so_far=[
            HistoryAction("UTG", "fold"),
            HistoryAction("MP", "raise", Decimal("30")),
            HistoryAction("CO", "fold"),
            HistoryAction("BTN", "fold"),
        ],
        current_street="preflop",
    )
    print("\n[Cenário 1] Preflop, MP faz raise para 30. Vez da SB agir.")
    print(" -> Turno Esperado: SB")
    print(f" -> Turno Calculado: {snap1.turn_pointer}")
    print(f" -> Ordem de Ação: {snap1.action_order}")

    # ---------------------------------------------------------
    # CENÁRIO 2: Flop (Pós-flop)
    # ---------------------------------------------------------
    # Chegamos ao flop com a SB, BB e o BTN.
    # De quem é a vez de falar primeiro no Flop? (Deve ser a SB, o primeiro à esquerda do botão)
    snap2 = compute_synthetic_state(
        seat_order=["SB", "BB", "UTG", "MP", "CO", "BTN"],
        button_player_id="BTN",
        big_blind=BB,
        small_blind=SB,
        actions_so_far=[
            # Ações do Preflop (Limp do BTN, SB completa, BB faz check)
            HistoryAction("UTG", "fold"),
            HistoryAction("MP", "fold"),
            HistoryAction("CO", "fold"),
            HistoryAction("BTN", "call", BB),
            HistoryAction("SB", "call", SB),  # SB completa para 10
            HistoryAction("BB", "check"),
        ],
        current_street="flop",  # O motor deve reconhecer que o preflop fechou
    )
    print("\n[Cenário 2] Início do Flop. Pote em multi-way (SB, BB, BTN).")
    print(" -> Turno Esperado: SB")
    print(f" -> Turno Calculado: {snap2.turn_pointer}")
    print(f" -> Ordem de Ação: {snap2.action_order}")

    # ---------------------------------------------------------
    # CENÁRIO 3: Flop com Aposta e Fold
    # ---------------------------------------------------------
    # No mesmo flop acima, a SB aposta e a BB faz fold.
    snap3 = compute_synthetic_state(
        seat_order=["SB", "BB", "UTG", "MP", "CO", "BTN"],
        button_player_id="BTN",
        big_blind=BB,
        small_blind=SB,
        actions_so_far=[
            # Preflop
            HistoryAction("UTG", "fold"),
            HistoryAction("MP", "fold"),
            HistoryAction("CO", "fold"),
            HistoryAction("BTN", "call", BB),
            HistoryAction("SB", "call", SB),
            HistoryAction("BB", "check"),
            # Flop começa aqui
            HistoryAction("SB", "bet", Decimal("15")),
            HistoryAction("BB", "fold"),
        ],
        current_street="flop",
    )
    print("\n[Cenário 3] Meio do Flop. SB aposta 15, BB faz fold.")
    print(" -> Turno Esperado: BTN")
    print(f" -> Turno Calculado: {snap3.turn_pointer}")
    print(f" -> Ordem de Ação: {snap3.action_order}")


if __name__ == "__main__":
    run_spot_checks()
