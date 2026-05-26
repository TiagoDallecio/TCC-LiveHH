from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from poker_vision.config import load_config
from poker_vision.core.video_stages import FramePacket
from poker_vision.geometry.calibrator import TableCalibrator
from poker_vision.geometry.zone_assigner import ZoneAssigner
from poker_vision.inference.card_detector_stage import CardDetectorStage


@pytest.fixture
def mock_dependencies():
    """Cria instâncias reais de calibração para testar a integração geométrica."""
    config = load_config()
    calibrator = TableCalibrator()
    calibrator.H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    calibrator.H_inv = calibrator.H

    assigner = ZoneAssigner(config)
    return calibrator, assigner


@pytest.fixture
def dummy_packet():
    """Gera um frame vazio para o teste."""
    return FramePacket(idx=0, frame=np.zeros((600, 800, 3), dtype=np.uint8), timestamp=0.0)


@patch("poker_vision.inference.card_detector_stage.YOLO")
def test_card_detector_geometry_integration(mock_yolo_class, mock_dependencies, dummy_packet):
    """Valida que o estágio processa as bboxes, acha o centroide e mapeia a zona correta."""
    calibrator, assigner = mock_dependencies

    mock_model_instance = MagicMock()
    mock_model_instance.names = {0: "Ah"}

    mock_box = MagicMock()
    mock_box.xyxy = [np.array([380, 480, 420, 560])]
    mock_box.conf = [np.array([0.95])]
    mock_box.cls = [np.array([0])]

    mock_result = MagicMock()
    mock_result.boxes = [mock_box]
    mock_model_instance.return_value = [mock_result]
    mock_yolo_class.return_value = mock_model_instance

    stage = CardDetectorStage(model_path="dummy.pt", calibrator=calibrator, zone_assigner=assigner)

    processed_packet = stage.process(dummy_packet)

    assert len(processed_packet.card_detections) == 1
    det = processed_packet.card_detections[0]

    assert det.label == "Ah"
    assert det.confidence == 0.95
    assert det.centroid == (400, 520)
    assert det.zone == "hero_hole_cards", f"A carta deveria estar no hero_hole_cards, mas caiu em {det.zone}"
