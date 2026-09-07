import pandas as pd

from src.analysis.build_analysis_corpus import (
    build_pre_dedup_dataframe,
    deduplicate_articles,
)


CONFIG = {
    "text": {
        "title_columns": ["title_clean", "title", "title_raw"],
        "body_columns": ["text_clean", "text", "text_raw", "text_excerpt"],
    },
    "dates": {
        "publication_date_columns": ["publication_date"],
        "archive_year_column": "archive_year",
        "snapshot_timestamp_column": "snapshot_timestamp",
    },
    "page_filter": {
        "excluded_page_types": ["topic_page", "author_page", "section_page"]
    },
    "periods": [
        {"label": "2015-2018", "start_year": 2015, "end_year": 2018},
        {"label": "2019-2021", "start_year": 2019, "end_year": 2021},
    ],
}


def test_url_date_corrects_archive_year():
    raw = pd.DataFrame(
        {
            "article_key": ["a"],
            "country": ["México"],
            "source": ["jornada"],
            "archive_year": [2021],
            "normalized_url": [
                "http://jornada.com.mx/notas/2020/08/13/estados/noticia"
            ],
            "title_clean": ["Título"],
            "text_clean": ["Texto editorial suficiente."],
        }
    )
    out = build_pre_dedup_dataframe(raw, CONFIG, 2015, 2021)
    assert int(out.iloc[0]["analysis_year"]) == 2020
    assert out.iloc[0]["analysis_year_source"] == "url_date"


def test_topic_page_is_marked_for_exclusion():
    raw = pd.DataFrame(
        {
            "article_key": ["a"],
            "country": ["Argentina"],
            "source": ["clarin_arg"],
            "archive_year": [2020],
            "normalized_url": ["http://clarin.com/tema/femicidio.html"],
            "title_clean": ["Femicidio"],
            "text_clean": ["Página temática"],
        }
    )
    out = build_pre_dedup_dataframe(raw, CONFIG, 2015, 2021)
    assert out.iloc[0]["page_type"] == "topic_page"
    assert not bool(out.iloc[0]["include_in_article_corpus"])


def test_cross_year_duplicate_becomes_one_article():
    raw = pd.DataFrame(
        {
            "article_key": ["a", "a"],
            "country": ["México", "México"],
            "source": ["milenio", "milenio"],
            "archive_year": [2020, 2021],
            "normalized_url": [
                "https://www.milenio.com/policia/noticia-x",
                "http://milenio.com/policia/noticia-x/",
            ],
            "title_clean": ["Título", "Título"],
            "text_clean": [
                "Texto corto.",
                "Texto mucho más largo con más palabras para seleccionar esta captura.",
            ],
        }
    )
    pre = build_pre_dedup_dataframe(raw, CONFIG, 2015, 2021)
    included = pre[pre["include_in_article_corpus"]].copy()
    final = deduplicate_articles(included)
    assert len(final) == 1
    assert int(final.iloc[0]["archive_year_count"]) == 2
    assert int(final.iloc[0]["snapshot_row_count"]) == 2
    assert "mucho más largo" in final.iloc[0]["analysis_text"]


def test_anchor_text_does_not_enter_analysis_text():
    raw = pd.DataFrame(
        {
            "article_key": ["a"],
            "country": ["Argentina"],
            "source": ["clarin_arg"],
            "archive_year": [2020],
            "normalized_url": ["http://clarin.com/sociedad/noticia.html"],
            "title_clean": ["Título"],
            "text_clean": ["Contenido editorial."],
            "anchor_text": ["Texto de portada que no debe analizarse."],
        }
    )
    out = build_pre_dedup_dataframe(raw, CONFIG, 2015, 2021)
    assert out.iloc[0]["analysis_text"] == "Contenido editorial."
    assert "portada" not in out.iloc[0]["analysis_text"]
