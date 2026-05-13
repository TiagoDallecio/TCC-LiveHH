from pathlib import Path
from typing import List, Tuple

import yaml
from pydantic import BaseModel


class ModelPaths(BaseModel):
    cards: str
    chips: str


class PipelineConfig(BaseModel):
    video_source: str
    model_paths: ModelPaths


class RoiDefinition(BaseModel):
    name: str
    polygon: List[List[int]]


class TableLayoutConfig(BaseModel):
    canonical_size: Tuple[int, int]
    blinds: Tuple[float, float]
    currency: str
    rois: List[RoiDefinition]


class AppConfig(BaseModel):
    """Agrupa as duas configurações principais da aplicação."""

    pipeline: PipelineConfig
    layout: TableLayoutConfig


def load_yaml(file_path: Path) -> dict:
    """Carrega um arquivo YAML e retorna um dicionário."""
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(
    pipeline_path: str = "configs/pipeline.yaml",
    layout_path: str = "configs/table_layout.yaml",
) -> AppConfig:
    """Lê os arquivos YAML e os converte em modelos validados do Pydantic."""
    try:
        pipeline_data = load_yaml(Path(pipeline_path))
        layout_data = load_yaml(Path(layout_path))

        return AppConfig(
            pipeline=PipelineConfig(**pipeline_data),
            layout=TableLayoutConfig(**layout_data),
        )
    except Exception as e:
        # Se o YAML estiver malformado ou faltando campos, o Pydantic joga um erro
        raise RuntimeError(f"Erro ao carregar configurações: {e}")
