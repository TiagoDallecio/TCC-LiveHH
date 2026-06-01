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
from poker_vision.geometry.calibrator import TableCalibrator
from poker_vision.geometry.zone_assigner import ZoneAssigner, draw_rois_on_frame
from poker_vision.inference.board_tracker_stage import BoardTrackerStage
from poker_vision.inference.card_detector_stage import CardDetectorStage
from poker_vision.inference.card_stabilizer_stage import CardStabilizerStage
from poker_vision.inference.opponent_action_inferencer import OpponentActionInferencer
from poker_vision.logic.hand_fsm import HandFSM
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

            else:
                from poker_vision.geometry.calibration_ui import run_calibration_ui

                video_path = Path(args.video) if hasattr(args, "video") and args.video else None
                sys.exit(run_calibration_ui(video_path))

    except Exception as e:
        print(f"Falha crítica: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
