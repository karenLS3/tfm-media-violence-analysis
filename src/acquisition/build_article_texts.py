from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.acquisition.fetch_html import fetch_wayback_html
from src.extraction.parse_article import parse_article_html
from src.utils.logging_config import setup_logger
from src.utils.run_paths import get_run_paths


ROOT = Path(__file__).resolve().parents[2]


def safe_name(url: str) -> str:
    return hashlib.md5(url.encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download archived article HTML and extract article text."
    )

    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Identificador de corrida. Ejemplo: 2015_01_test. "
            "Si se omite, usa las carpetas legacy data/candidates, data/extracted_text, etc."
        ),
    )

    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=1.0,
        help="Pausa entre requests a Wayback Machine. Por defecto: 1.0 segundo.",
    )

    return parser.parse_args()


def main(run_id: str | None = None, sleep_seconds: float = 1.0) -> None:
    paths = get_run_paths(ROOT, run_id)

    logger = setup_logger("build_article_texts", paths.logs_dir)

    in_path = paths.candidates_dir / "clean_candidates.parquet"
    out_path = paths.extracted_text_dir / "articles_text.parquet"
    html_dir = paths.raw_html_dir / "articles"
    report_dir = paths.reports_dir

    paths.extracted_text_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(
            f"No existe el archivo de candidatos limpios esperado:\n"
            f"  {in_path}\n\n"
            f"Primero corre build_clean_candidates con el mismo --run-id."
        )

    logger.info(f"Run ID: {run_id if run_id else 'legacy'}")
    logger.info(f"Reading clean candidates: {in_path}")
    logger.info(f"HTML dir: {html_dir}")
    logger.info(f"Output path: {out_path}")

    candidates = pd.read_parquet(in_path)

    records = []

    for row in tqdm(candidates.itertuples(index=False), total=len(candidates)):
        row_dict = row._asdict()

        article_url = row_dict.get("normalized_url") or row_dict.get("candidate_url")
        timestamp = row_dict.get("snapshot_timestamp")
        source = row_dict.get("source", "unknown")

        if not article_url:
            logger.error(f"Missing article URL | row={row_dict}")
            records.append(
                {
                    **row_dict,
                    "fetch_error": "missing_article_url",
                    "title": "",
                    "text": "",
                    "text_length": 0,
                    "html_path": "",
                }
            )
            continue

        if not timestamp:
            logger.error(f"Missing snapshot timestamp | url={article_url}")
            records.append(
                {
                    **row_dict,
                    "fetch_error": "missing_snapshot_timestamp",
                    "title": "",
                    "text": "",
                    "text_length": 0,
                    "html_path": "",
                }
            )
            continue

        article_dir = html_dir / str(source)
        article_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading article: {source} | {timestamp} | {article_url}")

        html_file, error = fetch_wayback_html(
            timestamp=timestamp,
            original_url=article_url,
            out_dir=article_dir,
            sleep_seconds=sleep_seconds,
        )

        if error:
            logger.error(f"Fetch error | {source} | {timestamp} | {article_url} | {error}")

            records.append(
                {
                    **row_dict,
                    "fetch_error": error,
                    "title": "",
                    "text": "",
                    "text_length": 0,
                    "html_path": "",
                }
            )
            continue

        try:
            parsed = parse_article_html(html_file)

            records.append(
                {
                    **row_dict,
                    "fetch_error": None,
                    "html_path": str(html_file),
                    **parsed,
                }
            )

        except Exception as exc:
            logger.exception(f"Parse error | {source} | {timestamp} | {article_url}")

            records.append(
                {
                    **row_dict,
                    "fetch_error": f"parse_error: {type(exc).__name__}: {exc}",
                    "title": "",
                    "text": "",
                    "text_length": 0,
                    "html_path": str(html_file),
                }
            )

    df = pd.DataFrame(records)

    if "text_length" not in df.columns and "text" in df.columns:
        df["text_length"] = df["text"].fillna("").astype(str).str.len()

    df.to_parquet(out_path, index=False)

    report_path = report_dir / "articles_text.csv"
    df.to_csv(report_path, index=False, encoding="utf-8-sig")

    summary_rows = [
        {"metric": "clean_candidates", "n": len(candidates)},
        {"metric": "articles_processed", "n": len(df)},
        {"metric": "fetch_or_parse_errors", "n": int(df["fetch_error"].notna().sum())},
        {"metric": "articles_with_text", "n": int(df["text"].fillna("").astype(str).str.len().gt(0).sum())},
        {"metric": "articles_without_text", "n": int(df["text"].fillna("").astype(str).str.len().eq(0).sum())},
    ]

    summary = pd.DataFrame(summary_rows)

    summary_path = report_dir / "articles_text_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    if {"source", "country"}.issubset(df.columns):
        by_source = (
            df.assign(
                has_text=df["text"].fillna("").astype(str).str.len().gt(0),
                has_error=df["fetch_error"].notna(),
            )
            .groupby(["source", "country"])
            .agg(
                articles_processed=("source", "size"),
                articles_with_text=("has_text", "sum"),
                fetch_or_parse_errors=("has_error", "sum"),
                avg_text_length=("text_length", "mean"),
            )
            .reset_index()
            .sort_values(["source", "country"])
        )

        by_source_path = report_dir / "articles_text_by_source.csv"
        by_source.to_csv(by_source_path, index=False, encoding="utf-8-sig")

        logger.info(f"Saved by-source summary: {by_source_path}")
        logger.info("\nArticles by source:")
        logger.info("\n" + by_source.to_string(index=False))

    logger.info(f"Articles processed: {len(df):,}")
    logger.info(f"Saved parquet: {out_path}")
    logger.info(f"Saved report: {report_path}")
    logger.info(f"Saved summary: {summary_path}")
    logger.info("\nArticles text summary:")
    logger.info("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    args = parse_args()
    main(
        run_id=args.run_id,
        sleep_seconds=args.sleep_seconds,
    )