from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from poker_vision.geometry.calibration_profile import CalibrationProfile, FiducialEntry


def test_profile_roundtrip(tmp_path: Path) -> None:
    # Matriz H simulada com números de ponto flutuante precisos
    mock_h = [[1.23456789, 0.0, 50.1], [0.0, 0.98765432, -10.5], [0.001, -0.002, 1.0]]

    profile = CalibrationProfile(
        profile_id="test_profile",
        created_at="2026-05-14T12:00:00Z",
        canonical_size=(1000, 600),
        fiducials=[FiducialEntry(name="pov_top_left", canonical=(0.0, 0.0), image=(10.5, 10.5))],
        homography=mock_h,
        reprojection_error_median_px=0.42,
    )

    filepath = tmp_path / "test_profile.yaml"
    profile.save(filepath)

    loaded_profile = CalibrationProfile.load(filepath)

    assert loaded_profile.profile_id == "test_profile"

    # Compara as matrizes usando numpy para garantir igualdade exata (byte-for-byte logic)
    original_h_np = np.array(profile.homography, dtype=np.float64)
    loaded_h_np = np.array(loaded_profile.homography, dtype=np.float64)
    np.testing.assert_array_equal(original_h_np, loaded_h_np)


def test_corrupted_profile_raises_error(tmp_path: Path) -> None:
    filepath = tmp_path / "bad_profile.yaml"

    bad_yaml = """
profile_id: corrupted
created_at: '2026-05-14T12:00:00Z'
canonical_size: [1000, 600]
fiducials: []
reprojection_error_median_px: 0.0
    """
    filepath.write_text(bad_yaml, encoding="utf-8")

    with pytest.raises(ValidationError):
        CalibrationProfile.load(filepath)
