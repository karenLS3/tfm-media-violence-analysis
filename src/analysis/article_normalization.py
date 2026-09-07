from __future__ import annotations

import re
from urllib.parse import parse_qsl, urlencode, urlsplit

import pandas as pd


TRACKING_QUERY_KEYS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term",
    "utm_content", "utm_id", "fbclid", "gclid",
}


def _clean_value(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def canonicalize_url_key(url: object) -> str:
    """Construye una identidad URL canónica para deduplicación analítica."""
    value = _clean_value(url)
    if not value:
        return ""

    parse_value = value if "://" in value else "//" + value

    try:
        parts = urlsplit(parse_value)
        hostname = (parts.hostname or "").lower().rstrip(".")
        if hostname.startswith("www."):
            hostname = hostname[4:]
        if not hostname:
            return value

        port = parts.port
        if port and port not in {80, 443}:
            hostname = f"{hostname}:{port}"

        path = re.sub(r"/{2,}", "/", parts.path or "/")
        if path != "/":
            path = path.rstrip("/")

        query_items = [
            (key, val)
            for key, val in parse_qsl(parts.query, keep_blank_values=True)
            if key.lower() not in TRACKING_QUERY_KEYS
        ]
        query_items.sort()
        query = urlencode(query_items, doseq=True)

        result = hostname + path
        if query:
            result += "?" + query
        return result
    except (TypeError, ValueError):
        return value


def extract_date_from_url(url: object) -> pd.Timestamp:
    """Extrae solo fechas inequívocas YYYY/MM/DD o YYYY-MM-DD de la ruta URL."""
    value = _clean_value(url)
    if not value:
        return pd.NaT

    try:
        parse_value = value if "://" in value else "//" + value
        path = urlsplit(parse_value).path
    except (TypeError, ValueError):
        path = value

    patterns = (
        r"/(20\d{2})/(\d{2})/(\d{2})(?:/|$)",
        r"/(20\d{2})-(\d{2})-(\d{2})(?:/|$)",
    )
    for expression in patterns:
        match = re.search(expression, path)
        if not match:
            continue
        year, month, day = map(int, match.groups())
        try:
            return pd.Timestamp(year=year, month=month, day=day)
        except ValueError:
            continue
    return pd.NaT


def parse_wayback_date(value: object) -> pd.Timestamp:
    text = _clean_value(value)
    if not text:
        return pd.NaT
    match = re.match(r"^(\d{4})(\d{2})(\d{2})", text)
    if not match:
        return pd.NaT
    year, month, day = map(int, match.groups())
    try:
        return pd.Timestamp(year=year, month=month, day=day)
    except ValueError:
        return pd.NaT


def classify_page_type(source: object, url: object) -> tuple[str, str]:
    """Clasifica patrones de no-artículo conocidos; ante duda, no excluye."""
    source_value = _clean_value(source).lower()
    url_value = _clean_value(url)

    if not url_value:
        return "article_or_unknown", "missing_url_not_excluded"

    try:
        parse_value = url_value if "://" in url_value else "//" + url_value
        path = urlsplit(parse_value).path.lower().rstrip("/")
    except (TypeError, ValueError):
        return "article_or_unknown", "unparseable_url_not_excluded"

    if source_value == "clarin_arg" and path.startswith("/tema/"):
        return "topic_page", "clarin_tema"
    if source_value == "pagina12" and path.startswith("/temas/"):
        return "topic_page", "pagina12_temas"
    if source_value == "lanacion" and path.startswith("/tema/"):
        return "topic_page", "lanacion_tema"

    if source_value == "milenio":
        if path.startswith("/temas/"):
            return "topic_page", "milenio_temas"
        if path == "/policia/violencia-genero":
            return "topic_page", "milenio_violencia_genero_landing"
        parts = [part for part in path.split("/") if part]
        if len(parts) == 2 and parts[0] == "opinion":
            return "author_page", "milenio_opinion_author"

    if source_value == "jornada":
        if re.fullmatch(r"/20\d{2}/\d{2}/\d{2}/[^/]+", path):
            return "section_page", "jornada_dated_section_landing"

    return "article_or_unknown", "no_non_article_pattern"


def parse_publication_date(value: object) -> pd.Timestamp:
    text = _clean_value(value)
    if not text:
        return pd.NaT
    try:
        parsed = pd.to_datetime(text, errors="coerce", utc=True)
    except (TypeError, ValueError):
        return pd.NaT
    if pd.isna(parsed):
        return pd.NaT
    if getattr(parsed, "tzinfo", None) is not None:
        parsed = parsed.tz_convert(None)
    return pd.Timestamp(parsed)


def resolve_analysis_date(
    publication_date: object,
    url_date: object,
    archive_year: object,
) -> tuple[pd.Timestamp, object, str]:
    """Prioridad: publication_date > url_date > archive_year."""
    publication_ts = parse_publication_date(publication_date)
    if pd.notna(publication_ts):
        return publication_ts, int(publication_ts.year), "publication_date"

    if url_date is not None and not pd.isna(url_date):
        url_ts = pd.Timestamp(url_date)
        return url_ts, int(url_ts.year), "url_date"

    try:
        if archive_year is not None and not pd.isna(archive_year):
            year = int(archive_year)
            if 1900 <= year <= 2100:
                return pd.NaT, year, "archive_year"
    except (TypeError, ValueError):
        pass

    return pd.NaT, pd.NA, "unresolved"
