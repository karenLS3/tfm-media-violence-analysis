from __future__ import annotations

from pathlib import Path
import pandas as pd
from tqdm import tqdm

from src.acquisition.fetch_html import fetch_wayback_html
from src.extraction.parse_article import parse_article_html
from src.utils.logging_config import setup_logger


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    logger = setup_logger("retry_failed_article_texts", ROOT / "data" / "logs")

    articles_path = ROOT / "data" / "extracted_text" / "articles_text.parquet"
    html_dir = ROOT / "data" / "raw_html" / "articles"

    df = pd.read_parquet(articles_path)

    failed_mask = df["fetch_error"].notna()
    failed = df[failed_mask].copy()

    logger.info(f"Failed articles to retry: {len(failed):,}")

    if failed.empty:
        logger.info("No failed articles found.")
        return

    for idx, row in tqdm(failed.iterrows(), total=len(failed)):
        article_url = row["normalized_url"]
        timestamp = row["snapshot_timestamp"]
        source = row["source"]

        html_file, error = fetch_wayback_html(
            timestamp=timestamp,
            original_url=article_url,
            out_dir=html_dir / source,
            sleep_seconds=3.0,
        )

        if error:
            logger.error(f"Still failed | {article_url} | {error}")
            continue

        parsed = parse_article_html(html_file)

        df.at[idx, "fetch_error"] = None
        df.at[idx, "title"] = parsed["title"]
        df.at[idx, "text"] = parsed["text"]
        df.at[idx, "text_length"] = parsed["text_length"]

        logger.info(f"Recovered | {article_url} | text_length={parsed['text_length']}")

    df.to_parquet(articles_path, index=False)

    logger.info(f"Updated: {articles_path}")
    logger.info(f"Remaining errors: {df['fetch_error'].notna().sum():,}")


if __name__ == "__main__":
    main()