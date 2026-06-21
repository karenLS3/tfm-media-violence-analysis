from __future__ import annotations

from urllib.parse import urlparse, urlunparse

import pandas as pd


BAD_PATTERNS = [
    "#",
    "/rss",
    "/feed",
    "/feeds",
    "/xml",
    "/clima",
    "/contactenos",
    "/contacto",
    "/dinreq/",
    "modal_v3",
    "ns_campaign=",

    # PDFs / documentos / archivos
    ".pdf",
    "/pdf/",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".zip",
    "codigodeetica",
    "codigo-de-etica",

    # Autoría / staff / perfiles
    "/autor/",
    "/autor",
    "/autores/",
    "/autores",
    "/author/",
    "/author",
    "/authors/",
    "/authors",
    "/staff",
    "/firma/",
    "/firmas/",

    # Páginas institucionales / legales
    "/ayuda/",
    "/ayuda",
    "/terminos",
    "terminos-y-condiciones",
    "términos-y-condiciones",
    "politica-privacidad",
    "politica-de-privacidad",
    "política-de-privacidad",
    "proteccion-datos",
    "protección-datos",
    "normas-confidencialidad",
    "aviso-legal",

    # Newsletter / login / suscripciones
    "newsletter",
    "newsletters",
    "/login",
    "/registro",
    "/suscrib",
    "/suscripciones",
    "/mi-cuenta",
    "/account",

    # Secciones que suelen ser navegación, no artículos
    "/tag/",
    "/tags/",
    "/buscar",
    "/busqueda",
    "/search",
    "/archivo",
    "/hemeroteca",
    "/edicion-impresa",
]


BAD_EXTENSIONS = (
    ".pdf",
    ".xml",
    ".rss",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".svg",
    ".mp4",
    ".mp3",
    ".zip",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
)


def normalize_url(url: str) -> str:
    parsed = urlparse(str(url or ""))

    scheme = "http"

    hostname = parsed.hostname or parsed.netloc
    netloc = hostname.lower().replace("www.", "") if hostname else ""

    path = parsed.path.rstrip("/")
    query = ""

    return urlunparse((scheme, netloc, path, "", query, ""))


def is_obvious_non_article_url(url: str) -> bool:
    url = str(url or "").strip().lower()

    if not url:
        return True

    parsed = urlparse(url)
    path = parsed.path.lower().rstrip("/")

    if any(pattern in url for pattern in BAD_PATTERNS):
        return True

    if path.endswith(BAD_EXTENSIONS):
        return True

    # Excluir homepages puras o secciones demasiado generales.
    if path in {"", "/", "/home", "/index.html"}:
        return True

    return False


def looks_like_article(url: str, anchor_text: str) -> bool:
    if not url:
        return False

    if is_obvious_non_article_url(url):
        return False

    parsed = urlparse(str(url))
    path = parsed.path.lower().rstrip("/")

    # Muy corto suele ser portada o sección: /politica, /sociedad, /opinion
    if path.count("/") < 2:
        return False

    anchor_text = str(anchor_text or "").strip()

    # Evita links de menú: "Inicio", "Sociedad", "Opinión", etc.
    if len(anchor_text) < 10:
        return False

    # Evita anchors que son claramente navegación.
    bad_anchor_texts = {
        "inicio",
        "home",
        "sociedad",
        "política",
        "politica",
        "opinión",
        "opinion",
        "mundo",
        "deportes",
        "economía",
        "economia",
        "cultura",
        "newsletter",
        "suscribite",
        "suscríbete",
        "registrarse",
        "ingresar",
        "login",
    }

    if anchor_text.lower() in bad_anchor_texts:
        return False

    return True


def filter_candidates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    mask = df.apply(
        lambda row: looks_like_article(
            row.get("candidate_url", ""),
            row.get("anchor_text", ""),
        ),
        axis=1,
    )

    clean = df[mask].copy()

    if clean.empty:
        return clean

    clean["normalized_url"] = clean["candidate_url"].apply(normalize_url)

    # Segunda pasada: filtrar también después de normalizar.
    clean = clean[
        ~clean["normalized_url"].apply(is_obvious_non_article_url)
    ].copy()

    clean = clean.drop_duplicates(
        subset=["source", "normalized_url"]
    )

    return clean