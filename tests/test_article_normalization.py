import pandas as pd

from src.analysis.article_normalization import (
    canonicalize_url_key,
    classify_page_type,
    extract_date_from_url,
    resolve_analysis_date,
)


def test_canonicalizes_hostname_variants():
    a = canonicalize_url_key(
        "https://www.lanacion.com.ar/tema/violencia-de-genero-tid47009/"
    )
    b = canonicalize_url_key(
        "http://lanacion.com.ar./tema/violencia-de-genero-tid47009"
    )
    assert a == b
    assert a == "lanacion.com.ar/tema/violencia-de-genero-tid47009"


def test_extracts_date_from_jornada_url():
    value = extract_date_from_url(
        "http://jornada.com.mx/notas/2020/08/13/estados/ejemplo"
    )
    assert value == pd.Timestamp("2020-08-13")


def test_does_not_infer_isolated_year():
    value = extract_date_from_url("https://example.com/noticia-2020-femicidio")
    assert pd.isna(value)


def test_topic_pages():
    cases = [
        ("clarin_arg", "http://clarin.com/tema/femicidio.html"),
        ("pagina12", "http://pagina12.com.ar/temas/660-femicidios"),
        (
            "lanacion",
            "http://lanacion.com.ar/tema/violencia-de-genero-tid47009",
        ),
        ("milenio", "http://milenio.com/temas/violencia-de-genero"),
    ]
    for source, url in cases:
        page_type, _ = classify_page_type(source, url)
        assert page_type == "topic_page"


def test_milenio_author_page():
    page_type, reason = classify_page_type(
        "milenio", "http://milenio.com/opinion/valeria-moy"
    )
    assert page_type == "author_page"
    assert reason == "milenio_opinion_author"


def test_clarin_opinion_article_is_not_excluded():
    page_type, _ = classify_page_type(
        "clarin_arg",
        "http://clarin.com/opinion/violencia-desigualdad-gran-desafio_0_ABC.html",
    )
    assert page_type == "article_or_unknown"


def test_jornada_dated_section_landing():
    page_type, _ = classify_page_type(
        "jornada", "http://jornada.com.mx/2018/07/25/correo"
    )
    assert page_type == "section_page"


def test_date_precedence_publication_over_url():
    analysis_date, year, source = resolve_analysis_date(
        publication_date="2021-03-02",
        url_date=pd.Timestamp("2020-08-13"),
        archive_year=2022,
    )
    assert analysis_date == pd.Timestamp("2021-03-02")
    assert year == 2021
    assert source == "publication_date"


def test_date_precedence_url_over_archive():
    analysis_date, year, source = resolve_analysis_date(
        publication_date=None,
        url_date=pd.Timestamp("2020-08-13"),
        archive_year=2021,
    )
    assert analysis_date == pd.Timestamp("2020-08-13")
    assert year == 2020
    assert source == "url_date"


def test_archive_year_is_last_resort():
    analysis_date, year, source = resolve_analysis_date(
        publication_date=None,
        url_date=pd.NaT,
        archive_year=2021,
    )
    assert pd.isna(analysis_date)
    assert year == 2021
    assert source == "archive_year"
