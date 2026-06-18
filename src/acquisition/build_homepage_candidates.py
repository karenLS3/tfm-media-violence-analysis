from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.acquisition.fetch_html import fetch_wayback_html
from src.extraction.extract_links import extract_internal_links_from_html
from src.utils.logging_config import setup_logger


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    logger = setup_logger("build_homepage_candidates", ROOT / "data" / "logs")

    cdx_path = ROOT / "data" / "raw_cdx" / "cdx_snapshots.parquet"
    cdx = pd.read_parquet(cdx_path)

    html_dir = ROOT / "data" / "raw_html" / "homepages"
    out_dir = ROOT / "data" / "candidates"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []

    for row in cdx.itertuples(index=False):
        logger.info(f"Downloading homepage snapshot: {row.source} | {row.timestamp}")

        html_path, error = fetch_wayback_html(
            timestamp=row.timestamp,
            original_url=row.original,
            out_dir=html_dir / row.source,
        )

        if error:
            logger.error(f"Fetch error | {row.timestamp} | {error}")
            continue

        links_df = extract_internal_links_from_html(
            html_path=html_path,
            base_url=row.original,
            source=row.source,
            country=row.country,
            snapshot_timestamp=row.timestamp,
        )

        logger.info(f"Extracted links: {len(links_df):,}")

        if not links_df.empty:
            frames.append(links_df)

    if not frames:
        logger.warning("No candidate links extracted.")
        return

    candidates = pd.concat(frames, ignore_index=True)

    candidates = candidates.drop_duplicates(
        subset=["source", "candidate_url"]
    )

    out_path = out_dir / "homepage_candidates.parquet"
    candidates.to_parquet(out_path, index=False)

    logger.info(f"Total candidate URLs: {len(candidates):,}")
    logger.info(f"Output: {out_path}")


if __name__ == "__main__":
    main()