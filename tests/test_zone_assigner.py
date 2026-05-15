import numpy as np

from poker_vision.config import load_config
from poker_vision.geometry.calibrator import TableCalibrator
from poker_vision.geometry.zone_assigner import ZoneAssigner


def test_assign_zone_correctly_maps_image_points() -> None:
    config = load_config()
    assigner = ZoneAssigner(config)
    calibrator = TableCalibrator()

    calibrator.H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    calibrator.H_inv = calibrator.H

    point_inside = (500.0, 400.0)
    zone = assigner.assign_zone(point_inside, calibrator)
    assert zone == "hero_bet_area", f"Deveria ser 'hero_bet_area', retornou {zone}"

    point_outside = (0.0, 0.0)
    zone_out = assigner.assign_zone(point_outside, calibrator)
    assert zone_out is None, f"Deveria ser None, retornou {zone_out}"
