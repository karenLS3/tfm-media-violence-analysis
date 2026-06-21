from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.acquisition.fetch_html import fetch_wayback_html
from src.extraction.parse_article import parse_article_html
from src.utils.logging_config import setup_logger
from src.utils.run_paths import get_run_paths


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retry failed article HTML downloads and text extraction."
    )

    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Identificador de corrida. Ejemplo: 2015_01_test. "
            "Si se omite, usa las carpetas legacy data/extracted_text, data/raw_html, etc."
        ),
    )

    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=3.0,
        help="Pausa entre reintentos a Wayback Machine. Por defecto: 3.0 segundos.",
    )

    return parser.parse_args()


def main(run_id: str | None = None, sleep_seconds: float = 3.0) -> None:
    paths = get_run_paths(ROOT, run_id)

    logger = setup_logger("retry_failed_article_texts", paths.logs_dir)

    articles_path = paths.extracted_text_dir / "articles_text.parquet"
    html_dir = paths.raw_html_dir / "articles"
    report_dir = paths.reports_dir

    html_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    if not articles_path.exists():
        raise FileNotFoundError(
            f"No existe el archivo de artículos esperado:\n"
            f"  {articles_path}\n\n"
            f"Primero corre build_article_texts con el mismo --run-id."
        )

    logger.info(f"Run ID: {run_id if run_id else 'legacy'}")
    logger.info(f"Reading articles: {articles_path}")

    df = pd.read_parquet(articles_path)

    if "fetch_error" not in df.columns:
        logger.info("No existe columna fetch_error. Nada para reintentar.")
        return

    failed_mask = df["fetch_error"].notna()
    failed = df[failed_mask].copy()

    logger.info(f"Failed articles to retry: {len(failed):,}")

    if failed.empty:
        logger.info("No failed articles found.")
        return

    recovered = 0
    still_failed = 0
    parse_failed = 0

    for idx, row in tqdm(failed.iterrows(), total=len(failed)):
        row_dict = row.to_dict()

        article_url = row_dict.get("normalized_url") or row_dict.get("candidate_url")
        timestamp = row_dict.get("snapshot_timestamp")
        source = row_dict.get("source", "unknown")

        if not article_url:
            logger.error(f"Missing article URL on retry | index={idx}")
            df.at[idx, "fetch_error"] = "retry_missing_article_url"
            still_failed += 1
            continue

        if not timestamp:
            logger.error(f"Missing timestamp on retry | url={article_url}")
            df.at[idx, "fetch_error"] = "retry_missing_snapshot_timestamp"
            still_failed += 1
            continue

        logger.info(f"Retrying article: {source} | {timestamp} | {article_url}")

        html_file, error = fetch_wayback_html(
            timestamp=timestamp,
            original_url=article_url,
            out_dir=html_dir / str(source),
            sleep_seconds=sleep_seconds,
        )

        if error:
            logger.error(f"Still failed | {source} | {timestamp} | {article_url} | {error}")
            df.at[idx, "fetch_error"] = error
            still_failed += 1
            continue

        try:
            parsed = parse_article_html(html_file)

            df.at[idx, "fetch_error"] = None
            df.at[idx, "title"] = parsed.get("title", "")
            df.at[idx, "text"] = parsed.get("text", "")
            df.at[idx, "text_length"] = parsed.get(
                "text_length",
                len(str(parsed.get("text", ""))),
            )
            df.at[idx, "html_path"] = str(html_file)

            recovered += 1

            logger.info(
                f"Recovered | {source} | {article_url} | "
                f"text_length={df.at[idx, 'text_length']}"
            )

        except Exception as exc:
            logger.exception(f"Parse error on retry | {source} | {timestamp} | {article_url}")

            df.at[idx, "fetch_error"] = f"retry_parse_error: {type(exc).__name__}: {exc}"
            df.at[idx, "html_path"] = str(html_file)
            parse_failed += 1

    df.to_parquet(articles_path, index=False)

    report_path = report_dir / "articles_text_after_retry.csv"
    df.to_csv(report_path, index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(
        [
            {"metric": "failed_before_retry", "n": len(failed)},
            {"metric": "recovered", "n": recovered},
            {"metric": "still_failed", "n": still_failed},
            {"metric": "parse_failed", "n": parse_failed},
            {"metric": "remaining_errors", "n": int(df["fetch_error"].notna().sum())},
            {
                "metric": "articles_with_text",
                "n": int(df["text"].fillna("").astype(str).str.len().gt(0).sum()),
            },
            {
                "metric": "articles_without_text",
                "n": int(df["text"].fillna("").astype(str).str.len().eq(0).sum()),
            },
        ]
    )

    summary_path = report_dir / "articles_text_retry_summary.csv"
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
                remaining_errors=("has_error", "sum"),
                avg_text_length=("text_length", "mean"),
            )
            .reset_index()
            .sort_values(["source", "country"])
        )

        by_source_path = report_dir / "articles_text_after_retry_by_source.csv"
        by_source.to_csv(by_source_path, index=False, encoding="utf-8-sig")

        logger.info(f"Saved retry by-source summary: {by_source_path}")
        logger.info("\nAfter retry by source:")
        logger.info("\n" + by_source.to_string(index=False))

    logger.info(f"Updated parquet: {articles_path}")
    logger.info(f"Saved report: {report_path}")
    logger.info(f"Saved summary: {summary_path}")
    logger.info(f"Remaining errors: {df['fetch_error'].notna().sum():,}")
    logger.info("\nRetry summary:")
    logger.info("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    args = parse_args()
    main(
        run_id=args.run_id,
        sleep_seconds=args.sleep_seconds,
    )