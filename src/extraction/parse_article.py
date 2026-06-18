from __future__ import annotations

from pathlib import Path
from bs4 import BeautifulSoup
import trafilatura


def parse_article_html(html_path: Path) -> dict:
    html = html_path.read_text(encoding="utf-8", errors="ignore")

    soup = BeautifulSoup(html, "lxml")

    h1 = soup.find("h1")
    title = h1.get_text(" ", strip=True) if h1 else ""

    text = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False,
        favor_precision=True,
    )

    if not text:
        text = soup.get_text(" ", strip=True)

    return {
        "title": title,
        "text": text or "",
        "text_length": len(text or ""),
    }