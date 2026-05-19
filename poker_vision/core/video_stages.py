import queue
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from poker_vision.config import AppConfig
from poker_vision.core.pipeline import Stage
from poker_vision.geometry.calibrator import TableCalibrator
from poker_vision.geometry.zone_assigner import draw_rois_on_frame


@dataclass
class FramePacket:
    """Objeto que trafega pela esteira contendo o frame e seus metadados."""

    idx: int
    frame: np.ndarray
    timestamp: float


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

            # Aplica a lógica de frame_skip (DoD)
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

        # Estado dos overlays (Teclado)
        self.show_rois = True
        self.show_detections = True

        # Cria o arquivo de log vazio
        self.log_file.write_text("Frame Index | Timestamp (s)\n")

    def process(self, packet: FramePacket) -> Optional[FramePacket]:
        # 1. Loga o timestamp na pasta da execução (DoD 2.4)
        with open(self.log_file, "a") as f:
            f.write(f"Frame {packet.idx:05d} | {packet.timestamp:.3f}\n")

        display_frame = packet.frame.copy()

        # 2. Desenha as RoIs se estiver ativado
        if self.show_rois:
            display_frame = draw_rois_on_frame(display_frame, self.config, self.calibrator)

        # (No futuro, adicionaremos a lógica de self.show_detections aqui)

        # Adiciona atalhos de teclado na tela
        cv2.putText(
            display_frame, "Atalhos: [R] RoIs | [Q] Sair", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

        cv2.imshow("Poker Vision - Pipeline Debug", display_frame)

        # 3. Lógica do Teclado (DoD 2.3)
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

        return None  # É um estágio terminal, não repassa o pacote
