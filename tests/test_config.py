from pathlib import Path

import pytest

from poker_vision.config import load_config


def test_load_valid_config(tmp_path: Path) -> None:
    """Garante que configurações válidas são lidas corretamente (DoD 2)."""
    pipe_file = tmp_path / "pipeline.yaml"
    pipe_file.write_text(
        "video_source: 'test.mp4'\nmodel_paths:\n  cards: 'c.pt'\n  chips: 'ch.pt'"
    )

    layout_file = tmp_path / "table_layout.yaml"
    layout_file.write_text(
        "canonical_size: [1000, 600]\nblinds: [1, 2]\ncurrency: 'BRL'\nrois: []"
    )

    config = load_config(str(pipe_file), str(layout_file))

    # Asserts verificando os tipos e valores
    assert config.pipeline.video_source == "test.mp4"
    assert config.layout.currency == "BRL"
    assert isinstance(config.layout.canonical_size, tuple)


def test_malformed_yaml_raises_validation_error(tmp_path: Path) -> None:
    """Garante que YAML sem os campos obrigatórios gera erro (DoD 1)."""
    pipe_file = tmp_path / "pipeline.yaml"
    # Faltando a chave 'model_paths' de propósito
    pipe_file.write_text("video_source: 'test.mp4'")

    layout_file = tmp_path / "table_layout.yaml"
    layout_file.write_text(
        "canonical_size: [1000, 600]\nblinds: [1, 2]\ncurrency: 'BRL'\nrois: []"
    )

    # Verifica se levanta um erro ao tentar criar o modelo
    with pytest.raises(RuntimeError) as exc_info:
        load_config(str(pipe_file), str(layout_file))

    assert "validation error" in str(exc_info.value).lower()
