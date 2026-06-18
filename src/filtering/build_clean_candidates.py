from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.filtering.filter_candidate_urls import filter_candidates
from src.utils.logging_config import setup_logger


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    logger = setup_logger("build_clean_candidates", ROOT / "data" / "logs")

    in_path = ROOT / "data" / "candidates" / "homepage_candidates.parquet"
    out_path = ROOT / "data" / "candidates" / "clean_candidates.parquet"

    df = pd.read_parquet(in_path)

    logger.info(f"Original candidates: {len(df):,}")

    clean_df = filter_candidates(df)

    logger.info(f"Clean candidates: {len(clean_df):,}")

    clean_df.to_parquet(out_path, index=False)

    logger.info(f"Saved: {out_path}")


if __name__ == "__main__":
    main()