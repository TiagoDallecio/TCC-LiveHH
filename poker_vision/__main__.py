import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from poker_vision.config import load_config
from poker_vision.core.pipeline import Orchestrator
from poker_vision.core.session import create_initial_context
from poker_vision.core.video_stages import DebugVisualizerStage, FrameReaderStage
from poker_vision.geometry.calibration_profile import CalibrationProfile
from poker_vision.geometry.calibration_ui import run_calibration_ui
from poker_vision.geometry.calibrator import TableCalibrator
from poker_vision.geometry.zone_assigner import ZoneAssigner, draw_rois_on_frame
from poker_vision.inference.board_tracker_stage import BoardTrackerStage
from poker_vision.inference.card_detector_stage import CardDetectorStage
from poker_vision.inference.card_stabilizer_stage import CardStabilizerStage
from poker_vision.inference.opponent_action_inferencer import OpponentActionInferencer
from poker_vision.logic.hand_fsm import HandFSM
from poker_vision.logic.replay import run_replay_scenario
from poker_vision.run_manager import setup_run_directory


def main() -> None:
    parser = argparse.ArgumentParser(prog="poker_vision", description="Poker Hand History CV Pipeline - CLI")

    subparsers = parser.add_subparsers(dest="command")

    # Comando 'config show'
    config_parser = subparsers.add_parser("config", help="Gerencia as configurações")
    config_parser.add_argument("action", choices=["show"], help="Ação a executar")

    # Comando: 'run'
    run_parser = subparsers.add_parser("run", help="Inicia o processamento do pipeline ponta a ponta")
    run_parser.add_argument("video", help="Caminho para o vídeo (.mp4)")
    run_parser.add_argument("--profile", required=True, help="Caminho para o YAML de calibração")
    run_parser.add_argument("--skip", type=int, default=1, help="Pulo de frames (frame_skip)")

    # Comando: 'layout'
    layout_parser = subparsers.add_parser("layout", help="Ferramentas de geometria da mesa")
    layout_parser.add_argument("action", choices=["render"], help="Ação a executar")

    # Comando: 'calibrate' e seus subcomandos
    cal_parser = subparsers.add_parser("calibrate", help="Ferramentas de calibração")
    cal_sub = cal_parser.add_subparsers(dest="cal_command")

    ui_parser = cal_sub.add_parser("ui", help="Abre a interface gráfica de calibração")
    ui_parser.add_argument("video", nargs="?", default=None, help="Caminho opcional para um vídeo")

    overlay_parser = cal_sub.add_parser("overlay", help="Testa as RoIs projetadas no vídeo ao vivo")
    overlay_parser.add_argument("video", help="Caminho para o vídeo (.mp4)")
    overlay_parser.add_argument("profile", help="Caminho para o YAML de calibração")

    replay_parser = subparsers.add_parser("replay", help="Executa replay determinístico de cenário YAML")
    replay_parser.add_argument("scenario", help="Caminho para o arquivo de cenário YAML")

    corpus_parser = subparsers.add_parser("corpus", help="Constrói o corpus de testes (YAML + PokerStars)")
    corpus_parser.add_argument(
        "--pokerstars-dir", type=Path, default=None, help="Diretório com arquivos .txt do PokerStars"
    )
    corpus_parser.add_argument("--yaml-dir", type=Path, default=None, help="Diretório com fixtures YAML")
    corpus_parser.add_argument("--output", type=Path, required=True, help="Caminho do JSON de saída")
    corpus_parser.add_argument(
        "--reroot", action="store_true", help="Ativa data augmentation (Hero em todos os assentos)"
    )
    corpus_parser.add_argument("--exclude-trivial", action="store_true", help="Ignora janelas sem ações")
    corpus_parser.add_argument("--max-hands", type=int, default=None, help="Limite máximo de mãos a parsear")

    # Comando: 'evaluate'
    eval_parser = subparsers.add_parser("evaluate", help="Avalia o inferidor de ações contra o corpus gerado")
    eval_parser.add_argument("--corpus", type=Path, required=True, help="Caminho para o test_corpus.json")
    eval_parser.add_argument(
        "--output-dir", type=Path, required=True, help="Pasta para salvar os gráficos e relatórios"
    )

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    try:
        config = load_config()

        if args.command == "config" and args.action == "show":
            print(config.model_dump_json(indent=2))

        elif args.command == "run":
            run_dir = setup_run_directory(config)

            logger = logging.getLogger("poker_vision")
            logger.info(f"Iniciando nova execução. Diretório de logs: {run_dir}")

            # 1. Carrega o Perfil de Calibração (Geometria Base)
            profile = CalibrationProfile.load(Path(args.profile))
            calibrator = TableCalibrator()
            calibrator.H = np.array(profile.homography, dtype=np.float32)
            calibrator.H_inv = np.linalg.inv(calibrator.H)
            logger.info(f"Perfil de calibração carregado: {args.profile}")

            # 2. Wizard de Configuração Inicial (Interativo, usa print/input)
            print("\n♠️ Poker Vision - Nova Sessão ♠️")
            try:
                num_players = int(input("Quantos jogadores na mesa? (ex: 6 ou 9): "))
                hero_pos = input("Qual sua posição inicial? (ex: BTN, SB, BB, UTG, HJ, CO): ").upper()

                ctx = create_initial_context(num_players, hero_pos)
                logger.info(f"Contexto da mesa criado: {num_players} jogadores, Hero={hero_pos}")

                inferencer = OpponentActionInferencer()
                fsm = HandFSM(ctx, inferencer)
                logger.info("HandFSM inicializada. Estado inicial: %s", fsm.state.value)

            except ValueError as e:
                print(f"\n❌ Erro de configuração: {e}")
                logger.error(f"Falha na configuração do Wizard: {e}")
                sys.exit(1)

            zone_assigner = ZoneAssigner(config)

            # 3. Iniciar o Pipeline de Vídeo
            print("\n✅ Pipeline iniciado! Pressione 'Q' na janela do vídeo para sair.")
            logger.info(f"Montando pipeline de vídeo para {args.video} com skip={args.skip}")

            reader = FrameReaderStage(args.video, frame_skip=args.skip)

            detector = CardDetectorStage(
                model_path=config.pipeline.model_paths.cards, calibrator=calibrator, zone_assigner=zone_assigner
            )

            stabilizer = CardStabilizerStage(min_hits=2, max_misses=5)

            tracker = BoardTrackerStage(on_board_change=fsm.handle_board_change)

            visualizer = DebugVisualizerStage(config, calibrator, run_dir)

            orchestrator = Orchestrator([reader, detector, stabilizer, tracker, visualizer])
            orchestrator.start()
            logger.info("Threads do pipeline iniciadas com sucesso.")

            try:
                while reader.is_alive() and visualizer.is_alive():
                    time.sleep(0.5)
            except KeyboardInterrupt:
                print("\n[CLI] Interrupção do usuário detectada (Ctrl+C).")
                logger.warning("Interrupção manual (Ctrl+C) recebida.")
            finally:
                orchestrator.stop()
                logger.info("Pipeline encerrado graciosamente.")

        elif args.command == "replay":
            result = run_replay_scenario(Path(args.scenario))
            print(f"Scenario: {result.scenario_name}")
            for line in result.transition_logs:
                print(line)
            print(f"Final state: {result.final_state}")
            print(f"Needs review: {result.needs_review}")

        elif args.command == "layout" and args.action == "render":
            from poker_vision.geometry.layout import render_layout

            render_layout(config)

        elif args.command == "calibrate":
            if args.cal_command == "overlay":
                profile = CalibrationProfile.load(Path(args.profile))
                calibrator = TableCalibrator()
                calibrator.H = np.array(profile.homography, dtype=np.float32)
                calibrator.H_inv = np.linalg.inv(calibrator.H)

                cap = cv2.VideoCapture(args.video)
                print("Pressione 'q' na janela de vídeo para sair.")
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    overlay = draw_rois_on_frame(frame, config, calibrator)
                    cv2.imshow("Poker Vision - RoI Overlay", overlay)

                    if cv2.waitKey(30) & 0xFF == ord("q"):
                        break
                cap.release()
                cv2.destroyAllWindows()

        elif args.command == "corpus":
            from poker_vision.inference.test_corpus import (
                CorpusBuilder,
                CorpusBuildOptions,
                save_corpus,
            )

            ps_files = list(args.pokerstars_dir.glob("*.txt")) if args.pokerstars_dir else []
            yaml_files = list(args.yaml_dir.glob("*.yaml")) if args.yaml_dir else []

            print("Construindo corpus de testes...")
            builder = CorpusBuilder(
                CorpusBuildOptions(
                    reroot=args.reroot,
                    include_trivial=not args.exclude_trivial,
                    max_hands=args.max_hands,
                )
            )

            cases = builder.build(pokerstars_files=ps_files, yaml_files=yaml_files)

            args.output.parent.mkdir(parents=True, exist_ok=True)
            save_corpus(cases, args.output)

            print(f"Sucesso! {len(cases)} casos de teste gerados e salvos em {args.output}")

            from collections import Counter

            by_complexity = Counter(c.complexity for c in cases)
            print("Distribuição por complexidade:", dict(by_complexity))

        elif args.command == "evaluate":
            from poker_vision.inference.evaluation.harness import run_from_corpus_file

            print(f"Iniciando avaliação do corpus ({args.corpus}). Isso pode levar alguns segundos...")

            out = run_from_corpus_file(args.corpus, args.output_dir)
            m = out.metrics

            print("\n" + "=" * 60)
            print(f"Avaliação Concluída: {m.total_cases:,} janelas avaliadas, {m.total_actions:,} ações.")
            print("=" * 60)
            print(f"Acurácia de Sequência (Exata): {m.sequence_accuracy:.3%}")
            print(f"Acurácia de Ação (Apenas tipo):{m.action_accuracy:.3%}")
            print(f"Acurácia Jogador + Ação:       {m.player_action_accuracy:.3%}")
            print(f"Erro Absoluto Médio (MAE):     {m.amount_mae:.3f} fichas/BB")
            print(f"ECE (Erro de Calibração):      {m.calibration.expected_calibration_error:.4f}")
            print(f"Falsos Positivos (Alucinação): {m.false_positive_rate_trivial:.3%} (Idealmente perto de 0%)")
            print("-" * 60)
            print("\n=== DIAGNÓSTICO PROFUNDO ===")

            # Check 1: Confianças
            from collections import Counter

            all_confidences = [
                round(c.predicted.confidence, 2) for r in out.per_case for c in r.per_action if c.predicted is not None
            ]
            top10 = Counter(all_confidences).most_common(10)
            print("Top 10 valores de confiança emitidos:")
            for conf, n in top10:
                print(f"  {conf:.2f} -> {n:,} ações")

            # Check 2: Racional (Quais funções estão rodando?)
            rationales = Counter(
                c.predicted.rationale for r in out.per_case for c in r.per_action if c.predicted is not None
            )
            print("\nDistribuição de Racional (Top 5):")
            for rat, n in rationales.most_common(5):
                print(f"  {n:>6,}  {rat}")

            # Check 3: Configuração carregada
            try:
                from poker_vision.inference.opponent_action_inferencer import InferencerConfig

                cfg = InferencerConfig()
                print("\nConfiguração Carregada:")
                print(f"  confidence_clean_check = {getattr(cfg, 'confidence_clean_check', 'NAO ENCONTRADO')}")
                print(f"  occam_linear_weight    = {getattr(cfg, 'occam_linear_weight', 'NAO ENCONTRADO')}")
            except Exception as e:
                print(f"\nErro ao carregar InferencerConfig: {e}")
            print("============================\n")
            print("Gráficos e relatórios salvos em:")
            for name, path in out.artifact_paths.items():
                print(f"  - {name}: {path}")

            # --- NOVO DIAGNÓSTICO (Limite de Informação) ---
            print("\n=== DIAGNÓSTICO DE TETO DE INFORMAÇÃO ===")
            from collections import Counter

            fold_to_raise_constraint_counts = []
            check_to_bet_constraint_counts = []

            for r in out.per_case:
                n_constrained = r.metadata.get("non_folders_count", 0)
                for c in r.per_action:
                    if c.expected is None or c.predicted is None:
                        continue
                    if c.expected.action == "fold" and c.predicted.action == "raise":
                        fold_to_raise_constraint_counts.append(n_constrained)
                    if c.expected.action == "check" and c.predicted.action == "bet":
                        check_to_bet_constraint_counts.append(n_constrained)

            print("Erros Fold -> Raise agrupados por # de restrições (Showdown) na janela:")
            for n, count in sorted(Counter(fold_to_raise_constraint_counts).items()):
                print(f"  {n} restrições: {count} erros")

            print("\nErros Check -> Bet agrupados por # de restrições (Showdown) na janela:")
            for n, count in sorted(Counter(check_to_bet_constraint_counts).items()):
                print(f"  {n} restrições: {count} erros")
            print("=================================================\n")

        else:
            video_path = Path(args.video) if hasattr(args, "video") and args.video else None
            sys.exit(run_calibration_ui(video_path))

    except Exception as e:
        print(f"Falha crítica: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
