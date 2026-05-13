from poker_vision.geometry.calibrator import TableCalibrator


def test_homography_roundtrip_and_error() -> None:
    """Garante que a calibração faz a ida e volta com precisão < 1px."""
    calibrator = TableCalibrator()

    # Simulando um retângulo torto na imagem (como se fosse a perspectiva da câmera)
    image_pts = {
        "tl": (100.0, 150.0),  # Top-Left
        "tr": (900.0, 120.0),  # Top-Right
        "br": (1100.0, 600.0),  # Bottom-Right
        "bl": (50.0, 650.0),  # Bottom-Left
    }

    # Onde esses pontos deveriam estar perfeitamente na nossa mesa 1000x600
    canonical_pts = {
        "tl": (0.0, 0.0),
        "tr": (1000.0, 0.0),
        "br": (1000.0, 600.0),
        "bl": (0.0, 600.0),
    }

    # 1. Tenta calibrar
    success = calibrator.calibrate_from_fiducials(image_pts, canonical_pts)
    assert success is True
    assert calibrator.H is not None

    # 2. Testa o erro (DoD: Erro deve ser < 1px em dados perfeitos/sintéticos)
    assert calibrator.median_error < 1.0, f"Erro de reprojeção alto: {calibrator.median_error}"

    # 3. Testa a ida e volta de um ponto (DoD: image -> canonical -> image)
    test_pt = (500.0, 300.0)
    canonical_pt = calibrator.image_to_canonical(test_pt)
    assert canonical_pt is not None

    roundtrip_pt = calibrator.canonical_to_image(canonical_pt)
    assert roundtrip_pt is not None

    # A diferença entre o ponto original e a volta não pode passar de 1 pixel
    diff_x = abs(test_pt[0] - roundtrip_pt[0])
    diff_y = abs(test_pt[1] - roundtrip_pt[1])
    assert diff_x < 1.0 and diff_y < 1.0
