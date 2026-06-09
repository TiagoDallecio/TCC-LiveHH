import argparse
import json
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Optional

import requests

from poker_vision.export.json_exporter import (
    ExporterConfig,
    ExportInputs,
    HandHistoryExporter,
    HandMetadata,
    HandStateLike,
    WinnerInfo,
)
from poker_vision.inference.opponent_action_inferencer import (
    AnchorEvent,
    AnchorType,
    InferencerConfig,
    OpponentActionInferencer,
    WindowAttribution,
)
from poker_vision.inference.table_context import TableContext

# Importações REAIS do seu repositório
from poker_vision.logic.models import ActionLogEntry

# ============================================================================
# ADAPTADORES PARA O EXPORTER
# ============================================================================

class MockAmbiguousWindowRef:
    def __init__(self, window_id, street, log_positions, attribution):
        self.window_id = window_id
        self.street = street
        self.log_positions = log_positions
        self.attribution = attribution
        self.primary_selection_method = "dfs_occam"
        self.is_ambiguous = True

class MockHandStateForExport(HandStateLike):
    def __init__(
        self,
        hero_id: str,
        button_seat: int,
        blinds: tuple[Decimal, Decimal],
        players: dict,
        action_log: list[ActionLogEntry],
        pot_final: Decimal,
        board_final: tuple[str, ...],
        ambiguous_windows: list = None
    ):
        self.hero_id = hero_id
        self.button_seat = button_seat
        self.blinds = blinds
        self.players = players
        self.action_log = action_log
        self.pot_final = pot_final
        self.board_final = board_final
        self.ambiguous_windows = ambiguous_windows or []


class MockPlayerForExport:
    def __init__(
        self,
        player_id: str,
        seat: int,
        stack_initial: Decimal,
        stack_final: Decimal,
        is_hero: bool,
        hole_cards: Optional[tuple[str, ...]] = None,
    ):
        self.player_id = player_id
        self.seat = seat
        self.stack_initial = stack_initial
        self.stack_final = stack_final
        self.is_hero = is_hero
        self.hole_cards = hole_cards or tuple()


# ============================================================================
# DEMO COMMAND IMPLEMENTATION
# ============================================================================


def demo_command(args: argparse.Namespace) -> None:
    print("\n" + "=" * 70)
    print("POKER VISION DEMO - End-to-End Pipeline (Com DFS Inferencer)")
    print("=" * 70 + "\n")

    # ------------------------------------------------------------------------
    # PASSO 1: Configurando o Estado Físico (Simulação do CV/YOLO)
    # ------------------------------------------------------------------------
    print(" Passo 1: Configurando a mesa e jogadores...")

    players_dict = {
        "Hero": MockPlayerForExport(
            player_id="Hero",
            seat=0,
            stack_initial=Decimal("1000"),
            stack_final=Decimal("992"),
            is_hero=True,
            hole_cards=("AS", "KS"),
        ),
        "Opp1": MockPlayerForExport(
            player_id="Opp1", seat=1, stack_initial=Decimal("1000"), stack_final=Decimal("999.5"), is_hero=False
        ),
        "Opp2": MockPlayerForExport(
            player_id="Opp2", seat=2, stack_initial=Decimal("1000"), stack_final=Decimal("999"), is_hero=False
        ),
        "Opp3": MockPlayerForExport(
            player_id="Opp3", seat=3, stack_initial=Decimal("1000"), stack_final=Decimal("999"), is_hero=False
        ),
        "Opp4": MockPlayerForExport(
            player_id="Opp4", seat=4, stack_initial=Decimal("1000"), stack_final=Decimal("1000"), is_hero=False
        ),
        "Opp5": MockPlayerForExport(
            player_id="Opp5", seat=5, stack_initial=Decimal("1000"), stack_final=Decimal("1000"), is_hero=False
        ),
    }

    # Log inicial garantido (Os blinds obrigatórios que a FSM já saberia)
    action_log = [
        ActionLogEntry(player_id="Opp1", action="post", amount=Decimal("0.5"), street="preflop"),
        ActionLogEntry(player_id="Opp2", action="post", amount=Decimal("1"), street="preflop"),
    ]

    # ------------------------------------------------------------------------
    # PASSO 2: Preparando o TableContext para o Inferencer
    # ------------------------------------------------------------------------
    print("\n Passo 2: Preparando o TableContext (FSM State)...")

    # Este contexto simula o estado da mesa logo APÓS os blinds serem postados,
    # aguardando as ações dos oponentes até chegar no Hero.
    ctx = TableContext(
        street="preflop",
        pot=Decimal("1.5"),
        current_bet=Decimal("1"),
        last_raise_size=Decimal("0"),
        big_blind=Decimal("1"),
        contributions_this_street={"Opp1": Decimal("0.5"), "Opp2": Decimal("1")},
        active_players=["Hero", "Opp1", "Opp2", "Opp3", "Opp4", "Opp5"],
        seat_order=["Hero", "Opp1", "Opp2", "Opp3", "Opp4", "Opp5"],
        hero_id="Hero",
        turn_pointer="Opp3",  # Ação começa no UTG (Opp3)
        action_order=("Opp3", "Opp4", "Opp5", "Hero", "Opp1", "Opp2"),
    )

    # ------------------------------------------------------------------------
    # PASSO 3: Criando os Anchors e Rodando o Inferencer (O Motor do TCC)
    # ------------------------------------------------------------------------
    print("\n Passo 3: YOLO detectou variação de Pote. Rodando Inferencer DFS...")

    # Evento 1: Abertura da janela (Logo após os blinds)
    start_anchor = AnchorEvent(
        anchor_type=AnchorType.STREET_START,
        timestamp=0.0,
        street="preflop",
        pot_before=Decimal("1.5"),
        pot_after=Decimal("1.5"),
    )

    # Evento 2: Fechamento da janela (YOLO viu que é a vez do Hero e o pote subiu!)
    # Vamos simular que o pote foi de 1.5 para 4.5 (Delta de 3.0)
    end_anchor = AnchorEvent(
        anchor_type=AnchorType.HERO_ACTION,
        timestamp=12.5,
        street="preflop",
        pot_before=Decimal("4.5"),
        pot_after=Decimal("4.5"),
    )

    inferencer = OpponentActionInferencer(InferencerConfig())
    inferencer.reset()

    # Abrindo a janela
    inferencer.on_anchor(start_anchor, ctx)

    # Fechando a janela e recebendo as ações deduzidas pela DFS
    inferred_actions = inferencer.on_anchor(end_anchor, ctx)

    ambiguous_windows = []

    if args.strategy == "calibrated":
        print("    Cálculo de EV calibrado: Exportando pesos e incertezas da DFS.")

        hand_id_demo = "DEMO_CALIBRATED_001"

        attr = WindowAttribution(
            primary=inferred_actions,
            alternatives=[inferred_actions],
            weights=[0.53]
        )

        ambig_ref = MockAmbiguousWindowRef(
            window_id="w_demo_1",
            street="preflop",
            log_positions=[2,3,4],
            attribution=attr
        )
        ambiguous_windows.append(ambig_ref)

    else:
        print("    Calculo de EV estrito: Ocultando a ambiguidade.")
        hand_id_demo = "DEMO_STRICT_001"

    print(f"    Delta de Pote percebido: ${end_anchor.pot_before - start_anchor.pot_after}")
    print(f"    O motor gerou {len(inferred_actions)} ações inferidas:")

    for ia in inferred_actions:
        print(f"      -> {ia.player_id} deu {ia.action.upper()} de ${ia.amount} (Confiança: {ia.confidence:.2f})")
        # Injetamos a ação gerada pelo seu algoritmo real de volta no log!
        action_log.append(ActionLogEntry(player_id=ia.player_id, action=ia.action, amount=ia.amount, street="preflop"))

    # Adicionando a ação do Hero para fechar o mock
    action_log.append(ActionLogEntry(player_id="Hero", action="call", amount=Decimal("1"), street="preflop"))

    # ------------------------------------------------------------------------
    # PASSO 4: Exportação JSON
    # ------------------------------------------------------------------------
    print("\n Passo 4: Criando HandState e rodando HandHistoryExporter...")

    mock_hand_state = MockHandStateForExport(
        hero_id="Hero",
        button_seat=0,
        blinds=(Decimal("0.5"), Decimal("1")),
        players=players_dict,
        action_log=action_log,
        pot_final=Decimal("5.5"),  # Total do preflop mockado
        board_final=(),
        ambiguous_windows=ambiguous_windows
    )

    meta = HandMetadata(
        hand_id=hand_id_demo, table_id="table_TCC", timestamp_start=datetime.now(), timestamp_end=datetime.now()
    )

    winners = [WinnerInfo(player_id="Hero", amount_won=Decimal("5.5"), hand_description="Preflop Take")]

    export_inputs = ExportInputs(hand=mock_hand_state, metadata=meta, winners=winners)

    try:
        exporter = HandHistoryExporter(config=ExporterConfig(minor_unit_scale=100))
        json_payload = exporter.export(export_inputs)
        json_str = json.dumps(json_payload, indent=2)

        export_path = Path("demo_inference_export.json")
        with open(export_path, "w") as f:
            f.write(json_str)

        print(f"    JSON gerado com sucesso ({len(json_str) / 1024:.1f} KB). Salvo em {export_path}")

    except Exception as e:
        print(f"    ERRO durante a exportação: {e}")
        sys.exit(1)

    # ------------------------------------------------------------------------
    # PASSO 5: Envio para o Backend Java
    # ------------------------------------------------------------------------
    print("\n Passo 5: Enviando payload para o microserviço Java...")
    java_endpoint = getattr(args, "java_endpoint", "http://localhost:8080/api/v1/hands")

    try:
        response = requests.post(
            java_endpoint, json=json_payload, headers={"Content-Type": "application/json"}, timeout=10
        )
        if response.status_code in [200, 201]:
            print(f"    ✅ Sucesso ({response.status_code}")
            try:
                print(f"   Retorno do servidor: {response.json()}")
            except requests.exceptions.JSONDecodeError:
                print(f"   (O servidor salvou com sucesso e retornou um corpo vazio).")
        else:
            print(f"    ⚠️ Erro HTTP {response.status_code}: {response.text}")
    except Exception as e:
        print(f"    ⚠️ Backend Java não alcançado: {e}")

    print("\n" + "=" * 70)
    print(" DEMO COMPLETA ")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Poker Vision Demo (With Inference)")
    parser.add_argument("--java-endpoint", type=str, default="http://localhost:8080/api/v1/hands")
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["strict", "calibrated"],
        default="calibrated",
        help="Define se o JSON exportado conterá as janelas de ambiguidade (calibrated) ou apenas os fatos crus (strict)."
    )
    args = parser.parse_args()
    demo_command(args)
