import cv2
import numpy as np

from poker_vision.config import AppConfig


def render_layout(config: AppConfig, output_path: str = "layout_preview.png") -> None:
    """Desenha o layout canônico em uma imagem e salva no disco."""
    width, height = config.layout.canonical_size

    # Cria uma imagem de fundo verde escuro (mesa de poker)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (50, 100, 50)  # Cor BGR

    # Cores fixas para cada tipo de zona do POV (BGR)
    colors = {
        "pot": (0, 255, 255),  # Amarelo (Fundo da mesa)
        "board": (255, 255, 0),  # Ciano (Centro)
        "hero_bet_area": (100, 255, 100),  # Verde claro (Apostas)
        "hero_hole_cards": (200, 100, 255),  # Rosa/Roxo (Suas cartas, base da tela)
    }

    for roi in config.layout.rois:
        # Define a cor baseada no nome
        color = (255, 255, 255)  # Branco padrão
        for key, c in colors.items():
            if key in roi.name:
                color = c
                break

        # Converte a lista de pontos para o formato do OpenCV
        pts = np.array(roi.polygon, np.int32).reshape((-1, 1, 2))

        # Desenha o contorno do polígono
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)

        # Adiciona o nome da zona no meio do polígono
        M = cv2.moments(pts)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(
                img,
                roi.name,
                (cx - 40, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

    cv2.imwrite(output_path, img)
    print(f"✅ Layout canônico renderizado salvo em: {output_path}")
