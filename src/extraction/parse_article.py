from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
import trafilatura


# =========================================================
# Limpieza básica
# =========================================================

MOJIBAKE_MARKERS = (
    "Ã",
    "Â",
    "â€",
    "â€™",
    "â€œ",
    "â€\x9d",
    "ðŸ",
)


def clean_value(value: object) -> str:
    if value is None:
        return ""

    return " ".join(
        str(value).split()
    ).strip()


def mojibake_score(text: str) -> int:
    return sum(
        text.count(marker)
        for marker in MOJIBAKE_MARKERS
    )


def fix_mojibake(value: object) -> str:
    """
    Repara de forma conservadora mojibake típico producido
    cuando texto UTF-8 fue interpretado como Latin-1 o
    Windows-1252.

    Una transformación solo se conserva si reduce la cantidad
    de indicadores de mojibake.
    """
    text = clean_value(value)

    if not text:
        return ""

    best = text
    best_score = mojibake_score(best)

    # Se permiten dos pasadas por si existe doble codificación.
    for _ in range(2):
        candidates: list[str] = []

        for encoding in (
            "latin1",
            "cp1252",
        ):
            try:
                candidate = (
                    best
                    .encode(encoding)
                    .decode("utf-8")
                )

                candidates.append(candidate)

            except (
                UnicodeEncodeError,
                UnicodeDecodeError,
            ):
                continue

        if not candidates:
            break

        candidate = min(
            candidates,
            key=mojibake_score,
        )

        candidate_score = mojibake_score(
            candidate
        )

        if candidate_score >= best_score:
            break

        best = candidate
        best_score = candidate_score

    return best


# =========================================================
# JSON-LD
# =========================================================

def iter_json_ld_objects(value):
    """
    Recorre estructuras JSON-LD, incluyendo listas y @graph.
    """
    if isinstance(value, dict):
        yield value

        graph = value.get("@graph")

        if isinstance(graph, (list, dict)):
            yield from iter_json_ld_objects(
                graph
            )

    elif isinstance(value, list):
        for item in value:
            yield from iter_json_ld_objects(
                item
            )


def load_json_ld_objects(
    soup: BeautifulSoup,
):
    """
    Recupera objetos definidos en bloques
    application/ld+json.
    """
    for tag in soup.find_all(
        "script",
        attrs={
            "type": "application/ld+json"
        },
    ):
        raw = (
            tag.string
            or tag.get_text()
        )

        if not raw:
            continue

        raw = raw.strip()

        if not raw:
            continue

        try:
            value = json.loads(raw)

        except (
            json.JSONDecodeError,
            TypeError,
        ):
            continue

        yield from iter_json_ld_objects(
            value
        )


def extract_json_ld_value(
    soup: BeautifulSoup,
    keys: list[str],
) -> str:
    """
    Devuelve el primer valor string no vacío encontrado
    para cualquiera de las claves indicadas.
    """
    for obj in load_json_ld_objects(
        soup
    ):
        for key in keys:
            value = obj.get(key)

            if (
                isinstance(value, str)
                and value.strip()
            ):
                return clean_value(
                    value
                )

    return ""


# =========================================================
# Meta tags
# =========================================================

def meta_content(
    soup: BeautifulSoup,
    *,
    property_name: str | None = None,
    name: str | None = None,
) -> str:
    attrs = {}

    if property_name:
        attrs["property"] = property_name

    if name:
        attrs["name"] = name

    tag = soup.find(
        "meta",
        attrs=attrs,
    )

    if not tag:
        return ""

    return clean_value(
        tag.get("content")
    )


# =========================================================
# Título
# =========================================================

def normalize_title_by_source(
    source: object,
    title_source: str,
    title: object,
) -> tuple[str, str]:
    """
    Aplica únicamente normalizaciones que dependen de la
    estructura conocida de una fuente.

    Página/12 histórico puede presentar:

        Página/12 :: Sociedad :: Titular

    donde el último componente corresponde al titular.

    En cambio:

        Página/12 :: rosario
        Página/12 :: soy
        Página/12

    son títulos genéricos del sitio o de una sección.
    """
    source_value = clean_value(
        source
    ).lower()

    title_value = fix_mojibake(
        title
    )

    if not title_value:
        return "", title_source

    if (
        source_value == "pagina12"
        and title_source == "html_title"
    ):
        parts = [
            part.strip()
            for part in title_value.split("::")
            if part.strip()
        ]

        if (
            len(parts) >= 3
            and parts[0]
            .lower()
            .startswith("página/12")
        ):
            headline = parts[-1]

            if headline:
                return (
                    headline,
                    "html_title_pagina12_headline",
                )

        return (
            "",
            "html_title_pagina12_generic_rejected",
        )

    return (
        title_value,
        title_source,
    )


def extract_title(
    soup: BeautifulSoup,
    source: object = "",
) -> tuple[str, str]:
    """
    Jerarquía de extracción del título:

    1. JSON-LD headline
    2. OpenGraph og:title
    3. twitter:title
    4. h1
    5. <title>
    """

    value = extract_json_ld_value(
        soup,
        ["headline"],
    )

    if value:
        return normalize_title_by_source(
            source,
            "jsonld_headline",
            value,
        )

    value = meta_content(
        soup,
        property_name="og:title",
    )

    if value:
        return normalize_title_by_source(
            source,
            "og_title",
            value,
        )

    value = meta_content(
        soup,
        name="twitter:title",
    )

    if value:
        return normalize_title_by_source(
            source,
            "twitter_title",
            value,
        )

    h1 = soup.find("h1")

    if h1:
        value = clean_value(
            h1.get_text(
                " ",
                strip=True,
            )
        )

        if value:
            return normalize_title_by_source(
                source,
                "h1",
                value,
            )

    if soup.title:
        value = clean_value(
            soup.title.get_text(
                " ",
                strip=True,
            )
        )

        if value:
            return normalize_title_by_source(
                source,
                "html_title",
                value,
            )

    return "", "missing"


# =========================================================
# Fecha de publicación
# =========================================================

def extract_publication_date_candidate(
    soup: BeautifulSoup,
) -> tuple[str, str]:
    """
    Recupera únicamente métodos cuya semántica representa
    explícitamente una fecha de publicación.

    Se excluyen deliberadamente:
    - meta name="date"
    - pubdate
    - publish-date
    - publication_date
    - time[datetime]

    porque las auditorías mostraron que estos campos pueden
    representar otras fechas de la página.
    """

    value = extract_json_ld_value(
        soup,
        ["datePublished"],
    )

    if value:
        return (
            value,
            "jsonld_datePublished",
        )

    value = meta_content(
        soup,
        property_name="article:published_time",
    )

    if value:
        return (
            value,
            "article_published_time",
        )

    return "", "missing"


def parse_snapshot_timestamp(
    value: object,
) -> pd.Timestamp:
    """
    Convierte timestamps Wayback YYYYMMDD o YYYYMMDDHHMMSS
    en timestamps UTC.
    """
    raw = clean_value(
        value
    )

    if not raw:
        return pd.NaT

    # Algunos Parquet antiguos pueden convertir el timestamp
    # numérico en algo similar a 20150101123000.0.
    if raw.endswith(".0"):
        raw = raw[:-2]

    digits = "".join(
        character
        for character in raw
        if character.isdigit()
    )

    if len(digits) >= 14:
        digits = digits[:14]

        return pd.to_datetime(
            digits,
            format="%Y%m%d%H%M%S",
            errors="coerce",
            utc=True,
        )

    if len(digits) >= 8:
        digits = digits[:8]

        return pd.to_datetime(
            digits,
            format="%Y%m%d",
            errors="coerce",
            utc=True,
        )

    return pd.NaT


def validate_publication_date(
    value: object,
    method: str,
    *,
    source: object = "",
    snapshot_timestamp: object = "",
) -> tuple[str, str]:
    """
    Valida una fecha editorial antes de incorporarla al
    dataset principal.

    Reglas:

    - Solo se aceptan métodos explícitos de publicación.
    - Las fechas HTML de Página/12 no se utilizan.
    - Si existe timestamp Wayback, la fecha editorial no
      puede ser posterior a la captura en más de un día.
    """

    raw = clean_value(
        value
    )

    if not raw:
        return "", "missing"

    trusted_methods = {
        "jsonld_datePublished",
        "article_published_time",
    }

    if method not in trusted_methods:
        return "", "missing"

    source_value = clean_value(
        source
    ).lower()

    # La auditoría temporal mostró resultados suficientemente
    # inconsistentes para no utilizar estas fechas en Página/12.
    if source_value == "pagina12":
        return "", "missing"

    publication_dt = pd.to_datetime(
        raw,
        errors="coerce",
        utc=True,
        format="mixed",
    )

    if pd.isna(publication_dt):
        return "", "missing"

    snapshot_dt = parse_snapshot_timestamp(
        snapshot_timestamp
    )

    # Cuando el caller conoce la captura Wayback, validamos
    # coherencia temporal.
    if pd.notna(snapshot_dt):
        maximum_publication_dt = (
            snapshot_dt
            + pd.Timedelta(days=1)
        )

        if publication_dt > maximum_publication_dt:
            return "", "missing"

    return (
        raw,
        method,
    )


def extract_publication_date(
    soup: BeautifulSoup,
    *,
    source: object = "",
    snapshot_timestamp: object = "",
) -> tuple[str, str]:
    candidate, candidate_source = (
        extract_publication_date_candidate(
            soup
        )
    )

    return validate_publication_date(
        candidate,
        candidate_source,
        source=source,
        snapshot_timestamp=snapshot_timestamp,
    )


# =========================================================
# Extracción
# =========================================================

def read_html(
    html_path: Path,
) -> str:
    return html_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )


def extract_article_metadata(
    html_path: Path,
    *,
    source: object = "",
    snapshot_timestamp: object = "",
) -> dict:
    """
    Extrae únicamente título y fecha editorial.

    No modifica archivos ni genera sidecars.
    """
    html = read_html(
        html_path
    )

    soup = BeautifulSoup(
        html,
        "lxml",
    )

    title, title_source = extract_title(
        soup,
        source=source,
    )

    (
        publication_date,
        publication_date_source,
    ) = extract_publication_date(
        soup,
        source=source,
        snapshot_timestamp=snapshot_timestamp,
    )

    return {
        "title": title,
        "title_source": title_source,
        "publication_date": publication_date,
        "publication_date_source": (
            publication_date_source
        ),
    }


def parse_article_html(
    html_path: Path,
    *,
    source: object = "",
    snapshot_timestamp: object = "",
) -> dict:
    """
    Extrae en una única pasada los campos canónicos de un
    artículo archivado.

    Devuelve:
        title
        title_source
        publication_date
        publication_date_source
        text
        text_length
    """
    html = read_html(
        html_path
    )

    soup = BeautifulSoup(
        html,
        "lxml",
    )

    title, title_source = extract_title(
        soup,
        source=source,
    )

    (
        publication_date,
        publication_date_source,
    ) = extract_publication_date(
        soup,
        source=source,
        snapshot_timestamp=snapshot_timestamp,
    )

    text = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False,
        favor_precision=True,
    )

    if not text:
        text = soup.get_text(
            " ",
            strip=True,
        )

    text = text or ""

    return {
        "title": title,
        "title_source": title_source,
        "publication_date": publication_date,
        "publication_date_source": (
            publication_date_source
        ),
        "text": text,
        "text_length": len(text),
    }