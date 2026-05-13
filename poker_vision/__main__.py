import argparse
import logging
import sys

from poker_vision.config import load_config
from poker_vision.run_manager import setup_run_directory


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="poker_vision", description="Poker Hand History CV Pipeline - CLI"
    )

    subparsers = parser.add_subparsers(dest="command")

    # Comando 'config show'
    config_parser = subparsers.add_parser("config", help="Gerencia as configurações")
    config_parser.add_argument("action", choices=["show"], help="Ação a executar")

    # Novo comando: 'run'
    subparsers.add_parser("run", help="Inicia o processamento do pipeline")

    # Novo comando: 'layout'
    layout_parser = subparsers.add_parser(
        "layout", help="Ferramentas de geometria da mesa"
    )
    layout_parser.add_argument("action", choices=["render"], help="Ação a executar")

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    try:
        config = load_config()

        if args.command == "config" and args.action == "show":
            print(config.model_dump_json(indent=2))

        elif args.command == "run":
            # Aqui é onde a mágica do Run Manager acontece
            run_dir = setup_run_directory(config)
            print(f"✅ Pipeline stub iniciado!\n📁 Diretório da execução: {run_dir}")

            # Mandando uma mensagem genérica de log
            logger = logging.getLogger("poker_vision")
            logger.info("Iniciando carregamento de frames... (Stub)")

        elif args.command == "layout" and args.action == "render":
            from poker_vision.layout import render_layout

            render_layout(config)

    except Exception as e:
        print(f"Falha crítica: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
