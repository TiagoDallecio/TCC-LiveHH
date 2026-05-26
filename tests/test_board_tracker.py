import numpy as np

from poker_vision.core.video_stages import CardDetection, FramePacket
from poker_vision.inference.board_tracker_stage import BoardTrackerStage


def create_mock_packet(labels: list[str]) -> FramePacket:
    """Cria um pacote simulado com as labels fornecidas na zona board."""
    detections = [
        CardDetection(label=lbl, confidence=0.9, bbox=(0, 0, 0, 0), centroid=(0, 0), zone="board") for lbl in labels
    ]
    return FramePacket(idx=0, frame=np.zeros(1), timestamp=0.0, card_detections=detections)


def test_board_state_transitions():
    """Valida a transição exata 0 -> 3 -> 4 -> 5 ignorando ruídos."""
    tracker = BoardTrackerStage()

    # 0 cartas (Preflop)
    p0 = tracker.process(create_mock_packet([]))
    assert len(p0.current_board) == 0

    # Dealer joga 2 cartas do Flop (Incompleto - Ruído). Rastreador deve ignorar.
    p_ruido1 = tracker.process(create_mock_packet(["Ah", "Kd"]))
    assert len(p_ruido1.current_board) == 0  # Mantém o estado anterior

    # Dealer joga a 3ª carta (Flop completo)
    p3 = tracker.process(create_mock_packet(["Ah", "Kd", "2s"]))
    assert len(p3.current_board) == 3

    # Turn
    p4 = tracker.process(create_mock_packet(["Ah", "Kd", "2s", "7d"]))
    assert len(p4.current_board) == 4

    # River
    p5 = tracker.process(create_mock_packet(["Ah", "Kd", "2s", "7d", "Jc"]))
    assert len(p5.current_board) == 5
