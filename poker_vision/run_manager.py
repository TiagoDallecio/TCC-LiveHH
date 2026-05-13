import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from poker_vision.config import AppConfig


class JSONFormatter(logging.Formatter):
    """Formata os registros de log nativos do Python para JSON (uma linha por log)."""

    def format(self, record: logging.LogRecord) -> str:
        log_record: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
        }
        # Se houver um erro/exceção atrelado ao log, inclui também
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_record)


def setup_run_directory(config: AppConfig, base_dir: str = "runs") -> Path:
    """Cria a estrutura de pastas da execução e configura o logger JSONL."""
    # Cria a pasta base com timestamp (Ex: runs/20260513_143000)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / timestamp
    output_dir = run_dir / "output"

    # Cria as pastas (parents=True garante que cria 'runs' se não existir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Salva uma cópia exata da configuração usada nesta execução
    config_copy_path = run_dir / "config_resolved.json"
    config_copy_path.write_text(config.model_dump_json(indent=2), encoding="utf-8")

    # Prepara o arquivo de log e limpa logs anteriores (útil para os testes)
    log_file = run_dir / "log.jsonl"
    logger = logging.getLogger("poker_vision")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)

    # Dispara o nosso primeiro log para provar que está funcionando
    logger.info("Run directory criado e logger JSON inicializado com sucesso.")

    return run_dir
