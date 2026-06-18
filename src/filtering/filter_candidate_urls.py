from __future__ import annotations

from urllib.parse import urlparse, urlunparse
import pandas as pd


BAD_PATTERNS = [
    "#",
    "/rss",
    "/clima",
    "/contactenos",
    "/dinreq/",
    "modal_v3",
    "ns_campaign=",
]


def normalize_url(url: str) -> str:
    parsed = urlparse(url)

    scheme = "http"

    hostname = parsed.hostname or parsed.netloc
    netloc = hostname.lower().replace("www.", "")

    path = parsed.path.rstrip("/")
    query = ""

    return urlunparse((scheme, netloc, path, "", query, ""))


def looks_like_article(url: str, anchor_text: str) -> bool:
    if not url:
        return False

    lower_url = url.lower()

    for bad in BAD_PATTERNS:
        if bad in lower_url:
            return False

    parsed = urlparse(url)
    path = parsed.path.lower()

    if path.count("/") < 2:
        return False

    if not anchor_text or len(anchor_text.strip()) < 10:
        return False

    return True


def filter_candidates(df: pd.DataFrame) -> pd.DataFrame:
    mask = df.apply(
        lambda row: looks_like_article(row["candidate_url"], row["anchor_text"]),
        axis=1,
    )

    clean = df[mask].copy()
    clean["normalized_url"] = clean["candidate_url"].apply(normalize_url)

    clean = clean.drop_duplicates(
        subset=["source", "normalized_url"]
    )

    return clean