import queue
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import cv2
import numpy as np

from poker_vision.config import AppConfig
from poker_vision.core.pipeline import Stage
from poker_vision.geometry.calibrator import TableCalibrator
from poker_vision.geometry.zone_assigner import draw_rois_on_frame


@dataclass
class CardDetection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]
    centroid: tuple[int, int]
    zone: Optional[str] = None


@dataclass
class FramePacket:
    """Objeto que trafega pela esteira contendo o frame e seus metadados."""

    idx: int
    frame: np.ndarray
    timestamp: float
    card_detections: List[CardDetection] = field(default_factory=list)
    current_board: List[str] = field(default_factory=list)


def cards_in_zone(detections: List[CardDetection], zone_name: str) -> List[CardDetection]:
    """
    Filtra uma lista de detecções, retornando apenas as cartas que pertencem à zona especificada.
    """
    return [det for det in detections if det.zone == zone_name]


class FrameReaderStage(Stage):
    """Estágio Produtor: Lê o vídeo e emite FramePackets respeitando o frame_skip."""

    def __init__(self, video_path: str, frame_skip: int = 1, max_q_size: int = 30) -> None:
        super().__init__(max_q_size=max_q_size)
        self.video_path = video_path
        self.frame_skip = frame_skip

    def run(self) -> None:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"[Erro no {self.name}] Não foi possível abrir: {self.video_path}")
            return

        frame_idx = 0

        while not (self.stop_event and self.stop_event.is_set()):
            ret, frame = cap.read()
            if not ret:
                break

            # Aplica a lógica de frame_skip
            if frame_idx % self.frame_skip == 0:
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                packet = FramePacket(idx=frame_idx, frame=frame, timestamp=timestamp)

                # Log exigido no DoD: "frame 0", "frame 1"...
                print(f"[{self.name}] Lendo frame {frame_idx} (timestamp: {timestamp:.2f}s)")

                if self.out_q is not None:
                    while not (self.stop_event and self.stop_event.is_set()):
                        try:
                            self.out_q.put(packet, timeout=0.1)
                            break
                        except queue.Full:
                            pass

            frame_idx += 1

        cap.release()

    def process(self, item: Any) -> Any:
        pass


class DebugVisualizerStage(Stage):
    """Estágio Terminal: Exibe o frame com overlays alternáveis via teclado."""

    def __init__(self, config: AppConfig, calibrator: TableCalibrator, run_dir: Path) -> None:
        super().__init__()
        self.config = config
        self.calibrator = calibrator
        self.log_file = run_dir / "timestamps_log.txt"

        self.show_rois = True
        self.show_detections = True

        self.log_file.write_text("Frame Index | Timestamp (s)\n")

    def process(self, packet: FramePacket) -> Optional[FramePacket]:
        with open(self.log_file, "a") as f:
            f.write(f"Frame {packet.idx:05d} | {packet.timestamp:.3f}\n")

        display_frame = packet.frame.copy()

        # 2. Desenha as RoIs se estiver ativado
        if self.show_rois:
            display_frame = draw_rois_on_frame(display_frame, self.config, self.calibrator)

        # 3. Desenha as deteções se estiver ativado
        if self.show_detections and hasattr(packet, "card_detections"):
            for det in packet.card_detections:
                x1, y1, x2, y2 = det.bbox

                if det.zone == "hero_hole_cards":
                    color = (255, 150, 50)
                elif det.zone == "board":
                    color = (0, 255, 255)
                else:
                    color = (0, 0, 255)

                # Desenha a caixa e o centróide
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(display_frame, det.centroid, 3, color, -1)

                # Texto com a Label e Confiança (ex: "Ah 0.94")
                label_text = f"{det.label} {det.confidence:.2f}"
                cv2.putText(display_frame, label_text, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 4. HUD do Board Tracker
        if hasattr(packet, "current_board"):
            hud_text = f"BOARD CONFIRMADO: [{' '.join(packet.current_board)}]"
            cv2.rectangle(display_frame, (10, 50), (450, 90), (0, 0, 0), -1)  # Fundo preto
            cv2.putText(
                display_frame, hud_text, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2  # Texto Amarelo
            )
        cv2.putText(
            display_frame, "Atalhos: [R] RoIs | [Q] Sair", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

        cv2.imshow("Poker Vision - Pipeline Debug", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            if self.stop_event:
                self.stop_event.set()
        elif key == ord("r"):
            self.show_rois = not self.show_rois
            print(f"[{self.name}] Toggle RoIs: {self.show_rois}")
        elif key == ord("d"):
            self.show_detections = not self.show_detections
            print(f"[{self.name}] Toggle Detections: {self.show_detections}")

        return None
