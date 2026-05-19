from decimal import Decimal
from typing import List

from poker_vision.inference.opponent_action_inferencer import TableContext


def create_initial_context(num_players: int, hero_position: str) -> TableContext:
    """
    Cria o contexto inicial da mesa baseado no número de jogadores e na posição do Herói.
    Assume que o Botão (BTN) é sempre o índice 0 na lista circular.
    """
    # Mapeamento genérico de posições (Index 0 = BTN, 1 = SB, 2 = BB...)
    # Para 6-max: 0:BTN, 1:SB, 2:BB, 3:UTG, 4:HJ, 5:CO
    # Para 9-max: 0:BTN, 1:SB, 2:BB, 3:UTG, 4:UTG+1, 5:UTG+2, 6:MP, 7:HJ, 8:CO
    positions_map = {
        "BTN": 0,
        "SB": 1,
        "BB": 2,
        "UTG": 3,
        "CO": num_players - 1,  # CO é sempre o último antes do BTN
        "HJ": num_players - 2,  # HJ é sempre o penúltimo
    }

    # Tratamento para posições flexíveis dependendo do tamanho da mesa
    if num_players == 6 and hero_position not in positions_map:
        positions_map["MP"] = 4  # Em 6-max, as vezes HJ é chamado de MP
    elif num_players > 6:
        positions_map["UTG+1"] = 4
        positions_map["UTG+2"] = 5
        positions_map["MP"] = 6

    hero_position = hero_position.upper()
    if hero_position not in positions_map:
        raise ValueError(f"Posição '{hero_position}' inválida para mesa de {num_players}.")

    hero_seat_index = positions_map[hero_position]

    # Monta a ordem dos assentos ("P0" é o Botão, "P1" é o SB...)
    seat_order: List[str] = []
    for i in range(num_players):
        if i == hero_seat_index:
            seat_order.append("Hero")
        else:
            # Nomeia os vilões baseado na distância deles para o botão
            seat_order.append(f"Villain_{i}")

    print("\n--- Setup da Mesa Iniciado ---")
    print(f"Jogadores: {num_players}")
    print(f"Sua Posição: {hero_position} (Assento {hero_seat_index})")
    print(f"Ordem: {' -> '.join(seat_order)}")
    print("------------------------------\n")

    return TableContext(
        num_players=num_players,
        button_seat=0,  # O BTN sempre começa no índice 0
        hero_seat=hero_seat_index,
        seat_order=seat_order.copy(),
        active_players=seat_order.copy(),
        current_street="preflop",
        current_bet=Decimal("0"),
        last_raise_size=Decimal("0"),
        turn_pointer=seat_order[3 % num_players],  # UTG é sempre o primeiro a falar no preflop
        pot=Decimal("0"),
        contributions_this_street={},
        hero_id="Hero",
    )


def rotate_button(ctx: TableContext) -> None:
    """Avança o botão do Dealer uma posição e reseta as variáveis da mão."""
    ctx.button_seat = (ctx.button_seat + 1) % ctx.num_players

    # Gira a lista de seat_order para o novo botão ficar no índice 0
    ctx.seat_order = ctx.seat_order[1:] + [ctx.seat_order[0]]
    ctx.active_players = ctx.seat_order.copy()

    # Reseta estado de apostas
    ctx.current_street = "preflop"
    ctx.current_bet = Decimal("0")
    ctx.last_raise_size = Decimal("0")
    ctx.pot = Decimal("0")
    ctx.contributions_this_street.clear()

    # O novo UTG (índice 3 na nova ordem) é o primeiro a agir
    ctx.turn_pointer = ctx.seat_order[3 % ctx.num_players]
