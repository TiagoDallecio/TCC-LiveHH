from typing import Optional

from ultralytics import YOLO

from poker_vision.core.pipeline import Stage
from poker_vision.core.video_stages import CardDetection, FramePacket
from poker_vision.geometry.calibrator import TableCalibrator
from poker_vision.geometry.zone_assigner import ZoneAssigner


class CardDetectorStage(Stage):
    """
    Estágio Consumidor/Produtor: Recebe frames, detecta cartas com YOLOv8,
    calcula centroides e atribui zonas espaciais.
    """

    def __init__(
        self,
        model_path: str,
        calibrator: TableCalibrator,
        zone_assigner: ZoneAssigner,
        confidence_threshold: float = 0.7,
        max_q_size: int = 30,
    ) -> None:
        super().__init__(max_q_size=max_q_size)

        self.model_path = model_path
        self.calibrator = calibrator
        self.zone_assigner = zone_assigner
        self.conf_thresh = confidence_threshold

        # Lazy loading: o modelo só sobe para a memória (VRAM/RAM) dentro da thread
        # quando o primeiro pacote chegar, evitando problemas no multiprocessamento.
        self._model: Optional[YOLO] = None

    def process(self, packet: FramePacket) -> Optional[FramePacket]:
        if self._model is None:
            print(f"[{self.name}] Carregando pesos do YOLO: {self.model_path}")
            self._model = YOLO(self.model_path)

        import cv2
        import numpy as np

        # 1. Máscara de Atenção (Pre-processing) para evitar alucinações fora das zonas de jogo
        mask = np.zeros(packet.frame.shape[:2], dtype=np.uint8)
        zonas_permitidas = ["board", "hero_hole_cards"]

        for roi in self.zone_assigner.rois:
            if any(z in roi for z in zonas_permitidas):
                image_pts = []
                for pt in self.zone_assigner.rois[roi].exterior.coords:
                    img_pt = self.calibrator.canonical_to_image((float(pt[0]), float(pt[1])))
                    if img_pt is not None:
                        image_pts.append([int(img_pt[0]), int(img_pt[1])])

                if len(image_pts) >= 3:
                    pts_array = np.array(image_pts, np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(mask, [pts_array], 255)

        masked_frame = cv2.bitwise_and(packet.frame, packet.frame, mask=mask)

        # 2. Inferência em Alta Resolução
        results = self._model(masked_frame, verbose=False, conf=self.conf_thresh, imgsz=1280, iou=0.4)

        # 3. Empacotamento de Resultados (Sem prints de Raio-X)
        if len(results) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0].item())
                cls_id = int(box.cls[0].item())
                label = self._model.names[cls_id]

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                try:
                    assigned_zone = self.zone_assigner.assign_zone(
                        image_xy=(float(cx), float(cy)), calibrator=self.calibrator
                    )
                except Exception:
                    assigned_zone = None

                packet.card_detections.append(
                    CardDetection(
                        label=label, confidence=conf, bbox=(x1, y1, x2, y2), centroid=(cx, cy), zone=assigned_zone
                    )
                )

        return packet
