import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

from poker_vision.config import load_config
from poker_vision.geometry.calibration_profile import CalibrationProfile
from poker_vision.geometry.calibrator import TableCalibrator
from poker_vision.geometry.zone_assigner import draw_rois_on_frame
from poker_vision.run_manager import setup_run_directory


def main() -> None:
    parser = argparse.ArgumentParser(prog="poker_vision", description="Poker Hand History CV Pipeline - CLI")

    subparsers = parser.add_subparsers(dest="command")

    # Comando 'config show'
    config_parser = subparsers.add_parser("config", help="Gerencia as configurações")
    config_parser.add_argument("action", choices=["show"], help="Ação a executar")

    # Novo comando: 'run'
    subparsers.add_parser("run", help="Inicia o processamento do pipeline")

    # Novo comando: 'layout'
    layout_parser = subparsers.add_parser("layout", help="Ferramentas de geometria da mesa")
    layout_parser.add_argument("action", choices=["render"], help="Ação a executar")

    # Novo comando: 'calibrate' e seus subcomandos
    cal_parser = subparsers.add_parser("calibrate", help="Ferramentas de calibração")
    cal_sub = cal_parser.add_subparsers(dest="cal_command")

    # Comando antigo (UI) virou 'ui' (se não passar nada, também abre a UI)
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
            print(f"✅ Pipeline stub iniciado!\n📁 Diretório da execução: {run_dir}")

            logger = logging.getLogger("poker_vision")
            logger.info("Iniciando carregamento de frames... (Stub)")

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
