from src.extraction.text_encoding import repair_mojibake


def test_repair_mojibake_repairs_common_spanish():
    assert (
        repair_mojibake(
            "MÃ©dico que presuntamente encubriÃ³ homicidio"
        )
        ==
        "Médico que presuntamente encubrió homicidio"
    )


def test_repair_mojibake_repairs_mexico():
    assert (
        repair_mojibake("Ciudad de MÃ©xico")
        ==
        "Ciudad de México"
    )


def test_repair_mojibake_preserves_valid_spanish():
    text = (
        "Las víctimas acudieron a la fiscalía "
        "y hablaron con policías."
    )

    assert repair_mojibake(text) == text


def test_repair_mojibake_preserves_accents():
    examples = [
        "será",
        "más",
        "negó",
        "tíos",
        "terminó",
        "Córdoba",
        "milímetros",
        "víctima",
        "víctimas",
        "delegación",
        "además",
        "séxtuple",
        "policía",
        "policías",
        "corazón",
        "Sánchez",
        "también",
        "fiscalía",
        "alcaldía",
        "crímenes",
        "había",
        "México",
    ]

    for text in examples:
        assert repair_mojibake(text) == text


def test_repair_mojibake_preserves_enye():
    examples = [
        "niña",
        "niño",
        "año",
        "señora",
        "compañera",
    ]

    for text in examples:
        assert repair_mojibake(text) == text


def test_repair_mojibake_does_not_drop_characters():
    original = "La víctima declaró ante la Fiscalía."

    repaired = repair_mojibake(original)

    assert repaired == original
    assert "víctima" in repaired
    assert "Fiscalía" in repaired


def test_repair_mojibake_empty_values():
    assert repair_mojibake("") == ""
    assert repair_mojibake(None) == ""