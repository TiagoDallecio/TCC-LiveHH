from typing import Dict, Optional, Tuple

from poker_vision.core.pipeline import Stage
from poker_vision.core.video_stages import FramePacket


class CardStabilizerStage(Stage):
    """
    Filtro temporal para evitar flickering (piscar) das deteções.
    Mecânicas:
    - Ativação (min_hits): Exige que a carta apareça N vezes antes de ser validada.
    - Paciência (max_misses): Mantém a carta viva por N frames se ela sumir temporariamente.
    """

    def __init__(self, min_hits: int = 2, max_misses: int = 5, max_q_size: int = 30) -> None:
        super().__init__(max_q_size=max_q_size)
        self.min_hits = min_hits
        self.max_misses = max_misses

        # Dicionário de rastreamento.
        # Chave: (label, zone) -> Ex: ("Ah", "hero_hole_cards")
        # Valor: dict com o estado de vida da carta
        self.tracked_cards: Dict[Tuple[str, str], dict] = {}

    def process(self, packet: FramePacket) -> Optional[FramePacket]:
        # 1. Envelhece todas as cartas rastreadas (aumenta o tempo em que não foram vistas)
        for key in self.tracked_cards:
            self.tracked_cards[key]["misses"] += 1

        # 2. Processa as deteções brutas do frame atual vindas do YOLO
        for det in packet.card_detections:
            # Ignora cartas que não caíram em nenhuma zona útil (ex: muck)
            if not det.zone:
                continue

            key = (det.label, det.zone)
            if key not in self.tracked_cards:
                self.tracked_cards[key] = {
                    "hits": 1,
                    "misses": 0,
                    "det": det,  # Guarda o objeto CardDetection para ter o bbox e centroid
                }
            else:
                self.tracked_cards[key]["hits"] += 1
                self.tracked_cards[key]["misses"] = 0
                self.tracked_cards[key]["det"] = det  # Atualiza a posição espacial

        # 3. Esquecimento (Drop): Limpa cartas que ficaram ausentes além da paciência
        keys_to_remove = [k for k, v in self.tracked_cards.items() if v["misses"] > self.max_misses]
        for k in keys_to_remove:
            del self.tracked_cards[k]

        # 4. Reconstrói a lista do pacote apenas com as cartas "Estáveis"
        stable_detections = []
        for v in self.tracked_cards.values():
            if v["hits"] >= self.min_hits:
                # Mesmo que misses > 0 (esteja invisível agora), a carta continua a ser repassada
                # com a sua última posição conhecida (v["det"])
                stable_detections.append(v["det"])

        packet.card_detections = stable_detections
        return packet
