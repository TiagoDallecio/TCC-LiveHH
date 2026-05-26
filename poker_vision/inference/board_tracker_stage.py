from typing import Callable, List, Optional

from poker_vision.core.pipeline import Stage
from poker_vision.core.video_stages import FramePacket, cards_in_zone


class BoardTrackerStage(Stage):
    """
    Rastreador semântico. Lê as cartas estabilizadas e emite eventos
    apenas quando o board muda para um tamanho válido no Texas Hold'em.
    """

    def __init__(self, on_board_change: Optional[Callable[[List[str]], None]] = None, max_q_size: int = 30) -> None:
        super().__init__(max_q_size=max_q_size)
        self.current_board: List[str] = []

        self.on_board_change = on_board_change

    def process(self, packet: FramePacket) -> Optional[FramePacket]:
        board_detections = cards_in_zone(packet.card_detections, "board")
        detected_labels = sorted([det.label for det in board_detections])

        valid_lengths = {0, 3, 4, 5}

        if len(detected_labels) in valid_lengths:
            if detected_labels != self.current_board:
                self.current_board = detected_labels

                # EMITE O EVENTO SEMÂNTICO
                if self.on_board_change:
                    self.on_board_change(self.current_board.copy())
                else:
                    # Fallback apenas para debug se ninguém estiver a escutar
                    print(f"[{self.name}] BoardChanged Event! Novo Board: {self.current_board}")

        packet.current_board = self.current_board.copy()
        return packet
