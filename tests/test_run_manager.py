import json
from pathlib import Path

from poker_vision.config import AppConfig, ModelPaths, PipelineConfig, TableLayoutConfig
from poker_vision.run_manager import setup_run_directory


def test_setup_run_directory_creates_structure_and_logs(tmp_path: Path) -> None:
    """Garante que a estrutura de pastas e os logs em formato JSON funcionam."""
    # 1. Configuração fictícia para o teste
    mock_config = AppConfig(
        pipeline=PipelineConfig(
            video_source="test.mp4", model_paths=ModelPaths(cards="c.pt", chips="ch.pt")
        ),
        layout=TableLayoutConfig(
            canonical_size=(1000, 600), blinds=(1.0, 2.0), currency="BRL", rois=[]
        ),
    )

    # Usamos o tmp_path do pytest para não sujar a nossa pasta raiz de projeto
    base_dir = tmp_path / "runs"
    run_dir = setup_run_directory(mock_config, base_dir=str(base_dir))

    # 2. Assertions: Verifica a criação das pastas e o arquivo de config (DoD 1 e 3)
    assert run_dir.exists()
    assert (run_dir / "output").exists()
    assert (run_dir / "output").is_dir()
    assert (run_dir / "config_resolved.json").exists()

    # 3. Assertions: Verifica o arquivo de log e seu formato JSON (DoD 2)
    log_file = run_dir / "log.jsonl"
    assert log_file.exists()

    with open(log_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) > 0  # Deve ter pelo menos o log inicial

        # Pega a primeira linha e tenta decodificar como JSON
        log_entry = json.loads(lines[0])
        assert "timestamp" in log_entry
        assert "level" in log_entry
        assert "message" in log_entry
        assert log_entry["level"] == "INFO"
        assert "Run directory criado" in log_entry["message"]
