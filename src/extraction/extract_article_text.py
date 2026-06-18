from bs4 import BeautifulSoup
from pathlib import Path


def extract_text_from_html(html_file: str) -> dict:
    html = Path(html_file).read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "aside"]):
        tag.decompose()

    title = soup.title.get_text(" ", strip=True) if soup.title else ""

    paragraphs = [
        p.get_text(" ", strip=True)
        for p in soup.find_all("p")
        if len(p.get_text(" ", strip=True)) > 40
    ]

    text = "\n".join(paragraphs)

    return {
        "title": title,
        "text": text,
    }