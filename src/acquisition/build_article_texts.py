from __future__ import annotations

from pathlib import Path
import hashlib
import pandas as pd
from tqdm import tqdm

from src.acquisition.fetch_html import fetch_wayback_html
from src.extraction.parse_article import parse_article_html
from src.utils.logging_config import setup_logger


ROOT = Path(__file__).resolve().parents[2]


def safe_name(url: str) -> str:
    return hashlib.md5(url.encode("utf-8")).hexdigest()


def main() -> None:
    logger = setup_logger("build_article_texts", ROOT / "data" / "logs")

    in_path = ROOT / "data" / "candidates" / "clean_candidates.parquet"
    out_path = ROOT / "data" / "extracted_text" / "articles_text.parquet"
    html_dir = ROOT / "data" / "raw_html" / "articles"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_parquet(in_path)

    records = []

    for row in tqdm(candidates.itertuples(index=False), total=len(candidates)):
        article_url = row.normalized_url
        timestamp = row.snapshot_timestamp

        article_dir = html_dir / row.source
        filename = f"{timestamp}_{safe_name(article_url)}.html"
        html_path = article_dir / filename

        html_file, error = fetch_wayback_html(
            timestamp=timestamp,
            original_url=article_url,
            out_dir=article_dir,
            sleep_seconds=1.0,
        )

        if error:
            logger.error(f"Fetch error | {article_url} | {error}")
            records.append({
                **row._asdict(),
                "fetch_error": error,
                "title": "",
                "text": "",
                "text_length": 0,
            })
            continue

        parsed = parse_article_html(html_file)

        records.append({
            **row._asdict(),
            "fetch_error": None,
            **parsed,
        })

    df = pd.DataFrame(records)
    df.to_parquet(out_path, index=False)

    logger.info(f"Articles processed: {len(df):,}")
    logger.info(f"Saved: {out_path}")


if __name__ == "__main__":
    main()