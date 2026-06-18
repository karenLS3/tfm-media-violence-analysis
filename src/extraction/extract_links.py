from __future__ import annotations

from pathlib import Path
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import pandas as pd


def clean_wayback_link(href: str) -> str | None:
    if not href:
        return None

    if href.startswith("/web/"):
        parts = href.split("/", 3)
        if len(parts) == 4:
            href = parts[3]

    if href.startswith("http://") or href.startswith("https://"):
        return href

    return href


def extract_internal_links_from_html(
    html_path: Path,
    base_url: str,
    source: str,
    country: str,
    snapshot_timestamp: str,
) -> pd.DataFrame:
    html = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")

    rows = []
    base_netloc = urlparse(base_url).netloc.replace("www.", "")

    for a in soup.find_all("a"):
        href = clean_wayback_link(a.get("href"))
        text = a.get_text(" ", strip=True)

        if not href:
            continue

        full_url = urljoin(base_url, href)
        parsed = urlparse(full_url)

        if parsed.scheme not in {"http", "https"}:
            continue

        link_netloc = parsed.netloc.replace("www.", "")

        if link_netloc != base_netloc:
            continue

        rows.append({
            "source": source,
            "country": country,
            "snapshot_timestamp": snapshot_timestamp,
            "homepage_url": base_url,
            "candidate_url": full_url,
            "anchor_text": text,
        })

    return pd.DataFrame(rows)