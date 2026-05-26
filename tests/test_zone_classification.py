from poker_vision.core.video_stages import CardDetection, cards_in_zone


def test_cards_in_zone_grouping():
    """
    Valida que a função agrupa corretamente 3 cartas comunitárias
    e 2 cartas de mão (fixture estático).
    """
    mock_detections = [
        CardDetection(label="Ah", confidence=0.95, bbox=(0, 0, 0, 0), centroid=(400, 500), zone="hero_hole_cards"),
        CardDetection(label="Kh", confidence=0.92, bbox=(0, 0, 0, 0), centroid=(420, 500), zone="hero_hole_cards"),
        CardDetection(label="2s", confidence=0.88, bbox=(0, 0, 0, 0), centroid=(300, 200), zone="board"),
        CardDetection(label="7d", confidence=0.91, bbox=(0, 0, 0, 0), centroid=(400, 200), zone="board"),
        CardDetection(label="Jc", confidence=0.89, bbox=(0, 0, 0, 0), centroid=(500, 200), zone="board"),
        CardDetection(label="9h", confidence=0.40, bbox=(0, 0, 0, 0), centroid=(400, 50), zone="pot"),
    ]

    board_cards = cards_in_zone(mock_detections, "board")
    hero_cards = cards_in_zone(mock_detections, "hero_hole_cards")
    pot_noise = cards_in_zone(mock_detections, "pot")
    muck_cards = cards_in_zone(mock_detections, "muck")

    assert len(board_cards) == 3, "Deveria ter encontrado exatamente 3 cartas no board"
    assert len(hero_cards) == 2, "Deveria ter encontrado exatamente 2 cartas na mão do herói"
    assert len(pot_noise) == 1, "Deveria ter encontrado o ruído no pote"
    assert len(muck_cards) == 0, "Não deveria haver cartas no muck"

    assert board_cards[0].label == "2s"
