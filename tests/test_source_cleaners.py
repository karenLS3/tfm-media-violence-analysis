from src.extraction.source_cleaners import (
    clean_text_by_source,
)


def test_eluniversal_repairs_mojibake():
    raw = (
        "MÃ©dico que presuntamente encubriÃ³ "
        "homicidio seguirÃ¡ proceso en libertad"
    )

    clean = clean_text_by_source(
        raw,
        "eluniversal_mx",
    )

    assert "Médico" in clean
    assert "encubrió" in clean
    assert "seguirá" in clean


def test_cleaner_preserves_correct_accents():
    raw = (
        "Las víctimas acudieron a la fiscalía "
        "y hablaron con policías."
    )

    clean = clean_text_by_source(
        raw,
        "pagina12",
    )

    assert "víctimas" in clean
    assert "fiscalía" in clean
    assert "policías" in clean