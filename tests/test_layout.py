from shapely.geometry import Polygon

from poker_vision.config import load_config


def test_no_overlapping_rois() -> None:
    """Garante que nenhuma Região de Interesse (RoI) se sobrepõe a outra (DoD 2)."""
    config = load_config()  # Carrega o YAML real

    # Converte os dados do Pydantic para Polígonos do Shapely
    polygons = {roi.name: Polygon(roi.polygon) for roi in config.layout.rois}

    names = list(polygons.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            name_a, name_b = names[i], names[j]
            poly_a, poly_b = polygons[name_a], polygons[name_b]

            # O interior das zonas não pode se sobrepor
            # (Elas podem apenas tocar as bordas, mas aqui nem isso devem fazer)
            assert not poly_a.overlaps(poly_b), f"Erro crítico: {name_a} se sobrepõe com {name_b}!"
            assert not poly_a.within(poly_b), f"Erro crítico: {name_a} está dentro de {name_b}!"


def test_seats_have_stack_and_bet_zones() -> None:
    """Garante que todos os assentos definidos possuem zona de bet e stack (DoD 2)."""
    config = load_config()
    roi_names = [roi.name for roi in config.layout.rois]

    # Descobre quais assentos foram definidos (ex: pega o '1' de 'seat_1_stack')
    seats_found = set()
    for name in roi_names:
        if name.startswith("seat_"):
            parts = name.split("_")
            if len(parts) >= 2 and parts[1].isdigit():
                seats_found.add(parts[1])

    # Verifica se existe o par (stack e bet) para cada assento encontrado
    for seat in seats_found:
        assert f"seat_{seat}_stack" in roi_names, f"Assento {seat} não tem zona de stack!"
        assert f"seat_{seat}_bet" in roi_names, f"Assento {seat} não tem zona de bet!"
