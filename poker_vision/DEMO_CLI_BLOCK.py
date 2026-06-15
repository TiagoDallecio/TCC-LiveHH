import argparse
import json
import requests
import sys
from decimal import Decimal
from datetime import datetime
from pathlib import Path

# Importações REAIS do seu repositório
from poker_vision.logic.models import ActionLogEntry
from poker_vision.export.json_exporter import (
    HandHistoryExporter, ExportInputs, HandMetadata, WinnerInfo, ExporterConfig, HandStateLike
)
from poker_vision.inference.table_context import TableContext
from poker_vision.inference.opponent_action_inferencer import (
    OpponentActionInferencer, InferencerConfig, AnchorEvent, AnchorType
)

# ============================================================================
# BLINDAGEM DE MOCKS (Garante que o json_exporter não ignore a Ambiguidade)
# ============================================================================
class MockWindowAttribution:
    def __init__(self, primary, alternatives, weights):
        self.primary = primary
        self.alternatives = alternatives
        self.weights = weights
        self.is_ambiguous = True # Força o exportador a não ignorar a janela!

class MockAmbiguousWindowRef:
    def __init__(self, window_id, street, log_positions, attribution):
        self.window_id = window_id
        self.street = street
        self.log_positions = log_positions
        self.attribution = attribution
        self.primary_selection_method = "dfs_occam"

class MockHandStateForExport(HandStateLike):
    def __init__(self, hero_id, button_seat, blinds, players, action_log, pot_final, board_final, ambiguous_windows=None):
        self.hero_id = hero_id
        self.button_seat = button_seat
        self.blinds = blinds
        self.players = players
        self.action_log = action_log
        self.pot_final = pot_final
        self.board_final = board_final
        self.ambiguous_windows = ambiguous_windows or []

class MockPlayerForExport:
    def __init__(self, player_id, seat, stack_initial, stack_final, is_hero, hole_cards=None):
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
    print("\n" + "="*70)
    print(f"POKER VISION DEMO - Pipeline TCC (Estratégia: {args.strategy.upper()})")
    print("="*70 + "\n")

    # ------------------------------------------------------------------------
    # PASSO 1 & 2: Mesa, Jogadores e FSM State
    # ------------------------------------------------------------------------
    players_dict = {
        "Hero": MockPlayerForExport("Hero", 0, Decimal("1000"), Decimal("992"), True, ("AS", "KS")),
        "Opp1": MockPlayerForExport("Opp1", 1, Decimal("1000"), Decimal("999.5"), False),
        "Opp2": MockPlayerForExport("Opp2", 2, Decimal("1000"), Decimal("999"), False),
        "Opp3": MockPlayerForExport("Opp3", 3, Decimal("1000"), Decimal("999"), False),
        "Opp4": MockPlayerForExport("Opp4", 4, Decimal("1000"), Decimal("1000"), False),
        "Opp5": MockPlayerForExport("Opp5", 5, Decimal("1000"), Decimal("1000"), False),
    }

    action_log = [
        ActionLogEntry(player_id="Opp1", action="post", amount=Decimal("0.5"), street="preflop"),
        ActionLogEntry(player_id="Opp2", action="post", amount=Decimal("1"), street="preflop"),
    ]

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
        turn_pointer="Opp3",
        action_order=("Opp3", "Opp4", "Opp5", "Hero", "Opp1", "Opp2")
    )

    # ------------------------------------------------------------------------
    # PASSO 3: Rodando Inferencer DFS (YOLO Delta)
    # ------------------------------------------------------------------------
    start_anchor = AnchorEvent(AnchorType.STREET_START, 0.0, "preflop", Decimal("1.5"), Decimal("1.5"))
    end_anchor = AnchorEvent(AnchorType.HERO_ACTION, 12.5, "preflop", Decimal("4.5"), Decimal("4.5"))

    inferencer = OpponentActionInferencer(InferencerConfig(all_in_contribution_multiplier=Decimal("5")))
    inferencer.on_anchor(start_anchor, ctx)
    inferred_actions = inferencer.on_anchor(end_anchor, ctx)

    # Injetando as ações deduzidas (Posições 2, 3 e 4 do log)
    for ia in inferred_actions:
        action_log.append(ActionLogEntry(player_id=ia.player_id, action=ia.action, amount=ia.amount, street="preflop"))

    action_log.append(ActionLogEntry(player_id="Hero", action="call", amount=Decimal("1"), street="preflop"))

    # ========================================================================
    # LÓGICA DE TOGGLE: STRICT vs CALIBRATED
    # ========================================================================
    ambiguous_windows = []

    if args.strategy == "calibrated":
        print("    Modo [CALIBRATED EV] ativo: Exportando Pesos e Cenários Alternativos da DFS...")
        hand_id_demo = "DEMO_CALIBRATED_001"

        # Cenário Matemático Alternativo: Em vez de Raise/Fold/Fold, a DFS acha que
        # existe 30% de chance do YOLO ter perdido frames e na verdade foram 3 Calls.
        alt_actions = [
            ActionLogEntry(player_id="Opp3", action="call", amount=Decimal("1"), street="preflop"),
            ActionLogEntry(player_id="Opp4", action="call", amount=Decimal("1"), street="preflop"),
            ActionLogEntry(player_id="Opp5", action="call", amount=Decimal("1"), street="preflop"),
        ]

        attr = MockWindowAttribution(
            primary=inferred_actions,
            alternatives=[alt_actions],
            weights=[0.70, 0.30] # 70% Primary, 30% Alternative
        )

        ambig_ref = MockAmbiguousWindowRef(
            window_id="w_demo_incerteza",
            street="preflop",
            log_positions=[2, 3, 4], # As posições do Raise/Fold/Fold no action_log
            attribution=attr
        )
        ambiguous_windows.append(ambig_ref)

    else:
        print("    Modo [STRICT EV] ativo: Ocultando a ambiguidade. Simulando visão ingênua...")
        hand_id_demo = "DEMO_STRICT_001"


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
        pot_final=Decimal("5.5"),
        board_final=(),
    )

    meta = HandMetadata(hand_id=hand_id_demo, table_id="table_TCC", timestamp_start=datetime.now(), timestamp_end=datetime.now())
    winners = [WinnerInfo(player_id="Hero", amount_won=Decimal("5.5"), hand_description="Preflop Take")]

    try:
        exporter = HandHistoryExporter(config=ExporterConfig(minor_unit_scale=100))
        json_payload = exporter.export(ExportInputs(hand=mock_hand_state,
                                                    metadata=meta,
                                                    winners=winners,
                                                    ambiguous_windows=ambiguous_windows))

        export_path = Path("demo_inference_export.json")
        with open(export_path, 'w') as f:
            f.write(json.dumps(json_payload, indent=2))
        print(f"    JSON salvo em {export_path}")

    except Exception as e:
        print(f"    ERRO durante a exportação: {e}")
        sys.exit(1)

    # ------------------------------------------------------------------------
    # PASSO 5: Envio para o Backend Java
    # ------------------------------------------------------------------------
    print("\n Passo 5: Enviando payload para o microserviço Java...")
    java_endpoint = getattr(args, "java_endpoint", "http://localhost:8080/api/v1/hands")

    try:
        response = requests.post(java_endpoint, json=json_payload, headers={"Content-Type": "application/json"}, timeout=10)
        if response.status_code in [200, 201]:
            print(f"    ✅ Sucesso ({response.status_code})! Mão salva no banco.")
            # Tratamento atualizado para prevenir o "Expecting Value" de respostas vazias
            try:
                print(f"    Resposta: {response.json()}")
            except ValueError:
                pass
        else:
            print(f"    ⚠️ Erro HTTP {response.status_code}: {response.text}")
    except Exception as e:
        print(f"    ⚠️ Backend Java não alcançado: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Poker Vision Demo")
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