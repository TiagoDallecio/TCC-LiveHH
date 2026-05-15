from typing import Optional, Tuple

import cv2
import numpy as np
from shapely.geometry import Point, Polygon

from poker_vision.config import AppConfig
from poker_vision.geometry.calibrator import TableCalibrator


class ZoneAssigner:
    """Responsável por mapear pontos do vídeo para as zonas canônicas da mesa."""

    def __init__(self, config: AppConfig) -> None:
        self.rois = {roi.name: Polygon(roi.polygon) for roi in config.layout.rois}

    def assign_zone(self, image_xy: Tuple[float, float], calibrator: TableCalibrator) -> Optional[str]:
        """Projeta um ponto da imagem real e verifica em qual zona (RoI) ele caiu."""
        canonical_pt = calibrator.image_to_canonical(image_xy)
        if canonical_pt is None:
            return None

        pt = Point(canonical_pt)
        for name, polygon in self.rois.items():
            if polygon.contains(pt):
                return name
        return None


def draw_rois_on_frame(frame: np.ndarray, config: AppConfig, calibrator: TableCalibrator) -> np.ndarray:
    """Desenha os polígonos canônicos deformados por cima do frame original do vídeo."""
    overlay = frame.copy()

    colors = {
        "pot": (0, 255, 255),
        "board": (255, 255, 0),
        "muck": (0, 0, 255),
        "stack": (255, 100, 100),
        "bet": (100, 255, 100),
    }

    for roi in config.layout.rois:
        image_pts = []
        for pt in roi.polygon:
            img_pt = calibrator.canonical_to_image((float(pt[0]), float(pt[1])))
            if img_pt is not None:
                image_pts.append([int(img_pt[0]), int(img_pt[1])])

        if len(image_pts) < 3:
            continue

        pts_array = np.array(image_pts, np.int32).reshape((-1, 1, 2))

        color = (255, 255, 255)
        for key, c in colors.items():
            if key in roi.name:
                color = c
                break

        cv2.polylines(overlay, [pts_array], isClosed=True, color=color, thickness=2)

        M = cv2.moments(pts_array)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(overlay, roi.name, (cx - 40, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return overlay
