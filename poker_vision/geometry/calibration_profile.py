from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class FiducialEntry(BaseModel):
    name: str
    canonical: tuple[float, float]
    image: tuple[float, float]


class CalibrationProfile(BaseModel):
    profile_id: str
    created_at: str
    canonical_size: tuple[int, int]
    fiducials: list[FiducialEntry]
    homography: list[list[float]] = Field(..., description="3x3 homography matrix (image -> canonical)")
    reprojection_error_median_px: float
    source_video: Optional[str] = None
    source_frame_index: Optional[int] = None

    def save(self, path: Path) -> None:
        """Salva o modelo atual em um arquivo YAML."""
        path.write_text(yaml.safe_dump(self.model_dump(), sort_keys=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CalibrationProfile":
        """Carrega e valida um perfil de calibração a partir de um YAML."""
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
