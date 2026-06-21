from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.acquisition.fetch_html import fetch_wayback_html
from src.extraction.extract_links import extract_internal_links_from_html
from src.utils.logging_config import setup_logger
from src.utils.run_paths import get_run_paths


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build homepage link candidates from archived homepage snapshots."
    )

    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Identificador de corrida. Ejemplo: 2015_01_test. "
            "Si se omite, usa las carpetas legacy data/raw_cdx, data/candidates, etc."
        ),
    )

    return parser.parse_args()


def write_diagnostics(
    diagnostics: list[dict],
    report_dir: Path,
    logger,
) -> None:
    diagnostics_df = pd.DataFrame(diagnostics)

    diagnostics_path = report_dir / "homepage_snapshot_diagnostics.csv"
    diagnostics_df.to_csv(diagnostics_path, index=False, encoding="utf-8-sig")

    logger.info(f"Homepage diagnostics: {diagnostics_path}")

    if diagnostics_df.empty:
        return

    summary = (
        diagnostics_df.assign(
            has_fetch_error=diagnostics_df["fetch_error"].notna(),
            has_links=diagnostics_df["links_extracted"].fillna(0).astype(int).gt(0),
        )
        .groupby(["source", "country"])
        .agg(
            snapshots_seen=("timestamp", "count"),
            fetch_errors=("has_fetch_error", "sum"),
            snapshots_with_links=("has_links", "sum"),
            total_links_extracted=("links_extracted", "sum"),
            avg_links_extracted=("links_extracted", "mean"),
        )
        .reset_index()
        .sort_values(["source", "country"])
    )

    summary_path = report_dir / "homepage_snapshot_diagnostics_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    logger.info(f"Homepage diagnostics summary: {summary_path}")
    logger.info("\nHomepage snapshot diagnostics summary:")
    logger.info("\n" + summary.to_string(index=False))


def main(run_id: str | None = None) -> None:
    paths = get_run_paths(ROOT, run_id)

    logger = setup_logger("build_homepage_candidates", paths.logs_dir)

    cdx_path = paths.raw_cdx_dir / "cdx_snapshots.parquet"

    if not cdx_path.exists():
        raise FileNotFoundError(
            f"No existe el archivo CDX esperado:\n"
            f"  {cdx_path}\n\n"
            f"Primero corre build_cdx_index con el mismo --run-id."
        )

    logger.info(f"Run ID: {run_id if run_id else 'legacy'}")
    logger.info(f"Reading CDX snapshots: {cdx_path}")

    cdx = pd.read_parquet(cdx_path)

    html_dir = paths.raw_html_dir / "homepages"
    out_dir = paths.candidates_dir
    report_dir = paths.reports_dir

    html_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    diagnostics: list[dict] = []
    frames: list[pd.DataFrame] = []

    for row in cdx.itertuples(index=False):
        logger.info(
            f"Downloading homepage snapshot: "
            f"{row.source} | {row.timestamp} | {row.original}"
        )

        html_path, error = fetch_wayback_html(
            timestamp=row.timestamp,
            original_url=row.original,
            out_dir=html_dir / row.source,
        )

        if error:
            logger.error(f"Fetch error | {row.source} | {row.timestamp} | {error}")

            diagnostics.append(
                {
                    "source": row.source,
                    "country": row.country,
                    "timestamp": row.timestamp,
                    "original": row.original,
                    "fetch_error": error,
                    "extract_error": None,
                    "html_path": "",
                    "html_size_bytes": 0,
                    "links_extracted": 0,
                }
            )

            continue

        html_size_bytes = 0

        try:
            html_size_bytes = Path(html_path).stat().st_size
        except Exception:
            html_size_bytes = 0

        try:
            links_df = extract_internal_links_from_html(
                html_path=html_path,
                base_url=row.original,
                source=row.source,
                country=row.country,
                snapshot_timestamp=row.timestamp,
            )

            diagnostics.append(
                {
                    "source": row.source,
                    "country": row.country,
                    "timestamp": row.timestamp,
                    "original": row.original,
                    "fetch_error": None,
                    "extract_error": None,
                    "html_path": str(html_path),
                    "html_size_bytes": html_size_bytes,
                    "links_extracted": len(links_df),
                }
            )

            logger.info(
                f"Extracted links: "
                f"{row.source} | {row.timestamp} | {len(links_df):,}"
            )

            if not links_df.empty:
                frames.append(links_df)

        except Exception as exc:
            logger.exception(
                f"Extract links error | {row.source} | {row.timestamp} | {row.original}"
            )

            diagnostics.append(
                {
                    "source": row.source,
                    "country": row.country,
                    "timestamp": row.timestamp,
                    "original": row.original,
                    "fetch_error": None,
                    "extract_error": f"{type(exc).__name__}: {exc}",
                    "html_path": str(html_path),
                    "html_size_bytes": html_size_bytes,
                    "links_extracted": 0,
                }
            )

            continue

    # Guardar diagnóstico SIEMPRE, incluso si no hubo candidatos.
    write_diagnostics(
        diagnostics=diagnostics,
        report_dir=report_dir,
        logger=logger,
    )

    if not frames:
        logger.warning("No candidate links extracted.")

        empty_candidates = pd.DataFrame(
            columns=[
                "candidate_url",
                "anchor_text",
                "source",
                "country",
                "snapshot_timestamp",
            ]
        )

        out_path = out_dir / "homepage_candidates.parquet"
        report_path = report_dir / "homepage_candidates.csv"
        summary_path = report_dir / "homepage_candidates_summary.csv"

        empty_candidates.to_parquet(out_path, index=False)
        empty_candidates.to_csv(report_path, index=False, encoding="utf-8-sig")

        empty_summary = pd.DataFrame(
            columns=["source", "country", "n_homepage_candidates"]
        )
        empty_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

        logger.info(f"Saved empty parquet: {out_path}")
        logger.info(f"Saved empty report: {report_path}")
        logger.info(f"Saved empty summary: {summary_path}")

        return

    candidates = pd.concat(frames, ignore_index=True)

    candidates = candidates.drop_duplicates(
        subset=["source", "candidate_url"]
    )

    out_path = out_dir / "homepage_candidates.parquet"
    candidates.to_parquet(out_path, index=False)

    report_path = report_dir / "homepage_candidates.csv"
    candidates.to_csv(report_path, index=False, encoding="utf-8-sig")

    summary = (
        candidates.groupby(["source", "country"])
        .size()
        .reset_index(name="n_homepage_candidates")
        .sort_values(["source", "country"])
    )

    summary_path = report_dir / "homepage_candidates_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    logger.info(f"Total candidate URLs: {len(candidates):,}")
    logger.info(f"Output parquet: {out_path}")
    logger.info(f"Output report: {report_path}")
    logger.info(f"Summary report: {summary_path}")
    logger.info("\nHomepage candidates summary:")
    logger.info("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    args = parse_args()
    main(run_id=args.run_id)