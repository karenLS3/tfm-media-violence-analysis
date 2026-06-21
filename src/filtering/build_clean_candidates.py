from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.filtering.filter_candidate_urls import filter_candidates
from src.utils.logging_config import setup_logger
from src.utils.run_paths import get_run_paths


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter homepage candidates into clean article URL candidates."
    )

    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Identificador de corrida. Ejemplo: 2015_01_test. "
            "Si se omite, usa las carpetas legacy data/candidates, data/logs, etc."
        ),
    )

    return parser.parse_args()


def main(run_id: str | None = None) -> None:
    paths = get_run_paths(ROOT, run_id)

    logger = setup_logger("build_clean_candidates", paths.logs_dir)

    in_path = paths.candidates_dir / "homepage_candidates.parquet"
    out_path = paths.candidates_dir / "clean_candidates.parquet"
    report_dir = paths.reports_dir

    report_dir.mkdir(parents=True, exist_ok=True)
    paths.candidates_dir.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(
            f"No existe el archivo de candidatos esperado:\n"
            f"  {in_path}\n\n"
            f"Primero corre build_homepage_candidates con el mismo --run-id."
        )

    logger.info(f"Run ID: {run_id if run_id else 'legacy'}")
    logger.info(f"Reading homepage candidates: {in_path}")

    df = pd.read_parquet(in_path)

    logger.info(f"Original candidates: {len(df):,}")

    clean_df = filter_candidates(df)

    logger.info(f"Clean candidates: {len(clean_df):,}")

    clean_df.to_parquet(out_path, index=False)

    report_path = report_dir / "clean_candidates.csv"
    clean_df.to_csv(report_path, index=False, encoding="utf-8-sig")

    if not clean_df.empty and {"source", "country"}.issubset(clean_df.columns):
        summary = (
            clean_df.groupby(["source", "country"])
            .size()
            .reset_index(name="n_clean_candidates")
            .sort_values(["source", "country"])
        )
    else:
        summary = pd.DataFrame(
            {
                "metric": ["original_candidates", "clean_candidates"],
                "n": [len(df), len(clean_df)],
            }
        )

    summary_path = report_dir / "clean_candidates_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    logger.info(f"Saved parquet: {out_path}")
    logger.info(f"Saved report: {report_path}")
    logger.info(f"Saved summary: {summary_path}")
    logger.info("\nClean candidates summary:")
    logger.info("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    args = parse_args()
    main(run_id=args.run_id)