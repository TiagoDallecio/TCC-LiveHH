import numpy as np

from poker_vision.core.video_stages import CardDetection, FramePacket
from poker_vision.inference.card_stabilizer_stage import CardStabilizerStage


def test_stabilizer_activation_and_patience():
    """Valida a ativação de uma carta e a sua resistência a falhas do YOLO (paciência)."""
    stabilizer = CardStabilizerStage(min_hits=2, max_misses=2)

    # Simula a mesma carta a ser vista pelo YOLO
    det_ah = CardDetection(label="Ah", confidence=0.9, bbox=(0, 0, 0, 0), centroid=(0, 0), zone="board")

    # Frame 1: YOLO vê a carta pela 1ª vez. (Ainda não passou do min_hits=2)
    p1 = FramePacket(idx=1, frame=np.zeros(1), timestamp=0.0, card_detections=[det_ah])
    res1 = stabilizer.process(p1)
    assert len(res1.card_detections) == 0, "A carta não deveria ativar no 1º frame (min_hits=2)"

    # Frame 2: YOLO vê novamente.
    p2 = FramePacket(idx=2, frame=np.zeros(1), timestamp=0.1, card_detections=[det_ah])
    res2 = stabilizer.process(p2)
    assert len(res2.card_detections) == 1, "A carta deveria ativar no 2º frame consecutivo"

    # Frame 3: A mão do dealer passou por cima e o YOLO ficou cego (ausência).
    p3 = FramePacket(idx=3, frame=np.zeros(1), timestamp=0.2, card_detections=[])
    res3 = stabilizer.process(p3)
    assert len(res3.card_detections) == 1, "A carta deveria ser mantida viva devido à paciência"

    # Frame 4 e 5: YOLO continua cego. Paciência esgota (misses > 2).
    stabilizer.process(FramePacket(idx=4, frame=np.zeros(1), timestamp=0.3, card_detections=[]))
    res5 = stabilizer.process(FramePacket(idx=5, frame=np.zeros(1), timestamp=0.4, card_detections=[]))

    assert len(res5.card_detections) == 0, "A carta deveria ser dropada após exceder os max_misses"
