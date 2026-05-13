from typing import Dict, Optional, Tuple

import cv2
import numpy as np


class TableCalibrator:
    """
    Calcula a transformação matemática (Homografia) entre a câmera real
    do vídeo e a nossa mesa ideal (canônica).
    """

    def __init__(self) -> None:
        self.H: Optional[np.ndarray] = None
        self.H_inv: Optional[np.ndarray] = None
        self.median_error: float = -1.0

    def calibrate_from_fiducials(
        self,
        image_points: Dict[str, Tuple[float, float]],
        canonical_points: Dict[str, Tuple[float, float]],
    ) -> bool:
        """
        Calcula a matriz de homografia recebendo os pontos clicados no vídeo
        e os respectivos pontos na mesa ideal.
        """
        # Verifica se temos pelo menos 4 pontos em comum
        common_keys = set(image_points.keys()).intersection(set(canonical_points.keys()))
        if len(common_keys) < 4:
            return False

        # Prepara os arrays para o OpenCV
        img_pts_list = [image_points[k] for k in common_keys]
        can_pts_list = [canonical_points[k] for k in common_keys]

        src_pts = np.array(img_pts_list, dtype=np.float32).reshape(-1, 1, 2)
        dst_pts = np.array(can_pts_list, dtype=np.float32).reshape(-1, 1, 2)

        # Calcula a matriz H usando RANSAC (robusto a pequenos erros de clique)
        h_matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if h_matrix is None:
            return False

        self.H = h_matrix
        # Calcula a matriz inversa (para voltar do canônico para a imagem)
        self.H_inv = np.linalg.inv(self.H)

        # Calcula o erro de reprojeção
        self._calculate_reprojection_error(src_pts, dst_pts)

        return True

    def image_to_canonical(self, point: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """Transforma um ponto do vídeo real (x, y) para coordenadas da mesa ideal."""
        if self.H is None:
            return None

        pt = np.array([[point]], dtype=np.float32)
        warped_pt = cv2.perspectiveTransform(pt, self.H)
        return (float(warped_pt[0][0][0]), float(warped_pt[0][0][1]))

    def canonical_to_image(self, point: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """Transforma um ponto da mesa ideal (x, y) para as coordenadas do vídeo real."""
        if self.H_inv is None:
            return None

        pt = np.array([[point]], dtype=np.float32)
        warped_pt = cv2.perspectiveTransform(pt, self.H_inv)
        return (float(warped_pt[0][0][0]), float(warped_pt[0][0][1]))

    def warp_frame(self, frame: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
        """Pega o frame do vídeo e "estica" ele para ficar na visão de cima (bird's-eye view)."""
        if self.H is None:
            return frame
        return cv2.warpPerspective(frame, self.H, output_size)

    def _calculate_reprojection_error(self, src_pts: np.ndarray, dst_pts: np.ndarray) -> None:
        """Calcula o quão preciso foi o clique do usuário."""
        if self.H is None:
            return

        projected_pts = cv2.perspectiveTransform(src_pts, self.H)

        # Calcula a distância (erro) entre o ponto projetado e onde ele deveria estar
        errors = np.linalg.norm(projected_pts - dst_pts, axis=2)
        self.median_error = float(np.median(errors))
