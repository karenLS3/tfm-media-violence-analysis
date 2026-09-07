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
        description=(
            "Descarga archivos de artículos archivados y extrae "
            "el contenido y metadatos del artículo canónico."
        )
    )

    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Identificador de corrida. Ejemplo: 2015_01_test. "
            "Si se omite, usa las carpetas legacy."
        ),
    )

    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=1.0,
        help=(
            "Pausa entre requests a Wayback Machine. "
            "Por defecto: 1.0 segundo."
        ),
    )

    return parser.parse_args()

def first_nonempty(
    row: dict,
    columns: list[str],
) -> str:
    for column in columns:
        value = row.get(column)

        if value is None or pd.isna(value):
            continue

        value = str(value).strip()

        if value:
            return value

    return ""


def empty_parse_fields() -> dict:
    """
    Mantiene un esquema estable incluso cuando la descarga
    o el parseo falla.
    """
    return {
        "title": "",
        "title_source": "missing",
        "publication_date": "",
        "publication_date_source": "missing",
        "text": "",
        "text_length": 0,
    }


def main(
    run_id: str | None = None,
    sleep_seconds: float = 1.0,
) -> None:
    paths = get_run_paths(
        ROOT,
        run_id,
    )

    logger = setup_logger(
        "build_article_texts",
        paths.logs_dir,
    )

    in_path = (
        paths.candidates_dir
        / "clean_candidates.parquet"
    )

    out_path = (
        paths.extracted_text_dir
        / "articles_text.parquet"
    )

    html_dir = (
        paths.raw_html_dir
        / "articles"
    )

    report_dir = paths.reports_dir

    paths.extracted_text_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    html_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not in_path.exists():
        raise FileNotFoundError(
            "No existe el archivo de candidatos limpios esperado:\n"
            f"  {in_path}\n\n"
            "Primero ejecuta build_clean_candidates "
            "con el mismo --run-id."
        )

    logger.info(
        f"Run ID: {run_id if run_id else 'legacy'}"
    )
    logger.info(
        f"Reading clean candidates: {in_path}"
    )
    logger.info(
        f"HTML dir: {html_dir}"
    )
    logger.info(
        f"Output path: {out_path}"
    )

    candidates = pd.read_parquet(
        in_path
    )

    records: list[dict] = []

    for row in tqdm(
        candidates.itertuples(index=False),
        total=len(candidates),
    ):
        row_dict = row._asdict()

        article_url = first_nonempty(
            row_dict,
            ["normalized_url", "candidate_url"]
        )

        timestamp = first_nonempty(
            row_dict,
            ["snapshot_timestamp"]
        )

        source = first_nonempty(
            row_dict,
            ["source"]
        ) or "unknown"

        # -------------------------------------------------
        # Validación mínima
        # -------------------------------------------------

        if not article_url:
            logger.error(
                f"Missing article URL | row={row_dict}"
            )

            records.append(
                {
                    **row_dict,
                    "fetch_error": "missing_article_url",
                    "html_path": "",
                    **empty_parse_fields(),
                }
            )

            continue

        if not timestamp:
            logger.error(
                "Missing snapshot timestamp "
                f"| url={article_url}"
            )

            records.append(
                {
                    **row_dict,
                    "fetch_error": (
                        "missing_snapshot_timestamp"
                    ),
                    "html_path": "",
                    **empty_parse_fields(),
                }
            )

            continue

        # -------------------------------------------------
        # Descarga HTML
        # -------------------------------------------------

        article_dir = (
            html_dir
            / str(source)
        )

        article_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        logger.info(
            "Downloading article: "
            f"{source} | "
            f"{timestamp} | "
            f"{article_url}"
        )

        html_file, error = fetch_wayback_html(
            timestamp=timestamp,
            original_url=article_url,
            out_dir=article_dir,
            sleep_seconds=sleep_seconds,
        )

        if error:
            logger.error(
                "Fetch error | "
                f"{source} | "
                f"{timestamp} | "
                f"{article_url} | "
                f"{error}"
            )

            records.append(
                {
                    **row_dict,
                    "fetch_error": error,
                    "html_path": "",
                    **empty_parse_fields(),
                }
            )

            continue

        # -------------------------------------------------
        # Parseo canónico
        # -------------------------------------------------

        try:
            parsed = parse_article_html(
                html_path=html_file,
                source=source,
                snapshot_timestamp=timestamp,
            )

            records.append(
                {
                    **row_dict,
                    "fetch_error": None,
                    "html_path": str(
                        html_file
                    ),
                    **parsed,
                }
            )

        except Exception as exc:
            logger.exception(
                "Parse error | "
                f"{source} | "
                f"{timestamp} | "
                f"{article_url}"
            )

            records.append(
                {
                    **row_dict,
                    "fetch_error": (
                        "parse_error: "
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                    "html_path": str(
                        html_file
                    ),
                    **empty_parse_fields(),
                }
            )

    # -----------------------------------------------------
    # Dataset final de extracción
    # -----------------------------------------------------

    df = pd.DataFrame(
        records
    )

    expected_columns = {
        "fetch_error": None,
        "html_path": "",
        "title": "",
        "title_source": "missing",
        "publication_date": "",
        "publication_date_source": "missing",
        "text": "",
        "text_length": 0,
    }

    for column, default in expected_columns.items():
        if column not in df.columns:
            df[column] = default

    # Salvaguarda para mantener la columna incluso si por
    # alguna razón un parser antiguo no la devolviera.
    if (
        "text_length" not in df.columns
        and "text" in df.columns
    ):
        df["text_length"] = (
            df["text"]
            .fillna("")
            .astype(str)
            .str.len()
        )

    df.to_parquet(
        out_path,
        index=False,
    )

    # -----------------------------------------------------
    # Reporte tabular
    # -----------------------------------------------------

    report_path = (
        report_dir
        / "articles_text.csv"
    )

    df.to_csv(
        report_path,
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------
    # Resumen
    # -----------------------------------------------------

    has_text = (
        df["text"]
        .fillna("")
        .astype(str)
        .str.len()
        .gt(0)
    )

    has_error = (
        df["fetch_error"]
        .notna()
    )

    has_publication_date = (
        df["publication_date"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    )

    has_title = (
        df["title"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    )

    summary_rows = [
        {
            "metric": "clean_candidates",
            "n": len(candidates),
        },
        {
            "metric": "articles_processed",
            "n": len(df),
        },
        {
            "metric": "fetch_or_parse_errors",
            "n": int(
                has_error.sum()
            ),
        },
        {
            "metric": "articles_with_text",
            "n": int(
                has_text.sum()
            ),
        },
        {
            "metric": "articles_without_text",
            "n": int(
                (~has_text).sum()
            ),
        },
        {
            "metric": "articles_with_title",
            "n": int(
                has_title.sum()
            ),
        },
        {
            "metric": "articles_with_publication_date",
            "n": int(
                has_publication_date.sum()
            ),
        },
    ]

    summary = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        report_dir
        / "articles_text_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------
    # Resumen por fuente
    # -----------------------------------------------------

    if {
        "source",
        "country",
    }.issubset(df.columns):

        by_source = (
            df.assign(
                has_text=has_text,
                has_title=has_title,
                has_publication_date=(
                    has_publication_date
                ),
                has_error=has_error,
            )
            .groupby(
                [
                    "source",
                    "country",
                ],
                dropna=False,
            )
            .agg(
                articles_processed=(
                    "source",
                    "size",
                ),
                articles_with_text=(
                    "has_text",
                    "sum",
                ),
                articles_with_title=(
                    "has_title",
                    "sum",
                ),
                articles_with_publication_date=(
                    "has_publication_date",
                    "sum",
                ),
                fetch_or_parse_errors=(
                    "has_error",
                    "sum",
                ),
                avg_text_length=(
                    "text_length",
                    "mean",
                ),
            )
            .reset_index()
            .sort_values(
                [
                    "source",
                    "country",
                ]
            )
        )

        by_source_path = (
            report_dir
            / "articles_text_by_source.csv"
        )

        by_source.to_csv(
            by_source_path,
            index=False,
            encoding="utf-8-sig",
        )

        logger.info(
            "Saved by-source summary: "
            f"{by_source_path}"
        )

        logger.info(
            "\nArticles by source:"
        )

        logger.info(
            "\n"
            + by_source.to_string(
                index=False
            )
        )

    logger.info(
        f"Articles processed: {len(df):,}"
    )

    logger.info(
        f"Saved parquet: {out_path}"
    )

    logger.info(
        f"Saved report: {report_path}"
    )

    logger.info(
        f"Saved summary: {summary_path}"
    )

    logger.info(
        "\nArticles text summary:"
    )

    logger.info(
        "\n"
        + summary.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    args = parse_args()

    main(
        run_id=args.run_id,
        sleep_seconds=args.sleep_seconds,
    )