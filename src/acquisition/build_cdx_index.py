from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

from src.acquisition.cdx_client import query_cdx_simple
from src.acquisition.date_utils import iter_days
from src.utils.logging_config import setup_logger
from src.utils.run_paths import get_run_paths


ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_config_path(config: str) -> Path:
    """
    Permite usar:
      --config sources_test.yaml
    o:
      --config configs/sources_argentina_mexico.yml
    """
    candidate = Path(config)

    if candidate.exists():
        return candidate

    candidate = ROOT / "configs" / config

    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        f"No se encontró el config: {config}\n"
        f"Probé también: {candidate}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build CDX index from Wayback Machine snapshots."
    )

    parser.add_argument(
        "--config",
        default="sources_test.yaml",
        help=(
            "Archivo YAML de fuentes. Puede ser solo el nombre dentro de configs/, "
            "por ejemplo sources_test.yaml, o una ruta como configs/sources_argentina_mexico.yml."
        ),
    )

    parser.add_argument(
        "--from-date",
        default=None,
        help="Fecha inicial en formato YYYYMMDD. Si se omite, usa from_date del config.",
    )

    parser.add_argument(
        "--to-date",
        default=None,
        help="Fecha final en formato YYYYMMDD. Si se omite, usa to_date del config.",
    )

    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Identificador de corrida. Ejemplo: 2015_01_test. "
            "Si se omite, usa las carpetas legacy data/raw_cdx, data/logs, etc."
        ),
    )

    return parser.parse_args()


def main(
    config: str = "sources_test.yaml",
    from_date: str | None = None,
    to_date: str | None = None,
    run_id: str | None = None,
) -> None:
    paths = get_run_paths(ROOT, run_id)

    logger = setup_logger("build_cdx_index", paths.logs_dir)

    config_path = resolve_config_path(config)
    config_data = load_yaml(config_path)

    # Permite sobrescribir fechas desde CLI sin modificar el YAML.
    if from_date is not None:
        config_data["from_date"] = from_date

    if to_date is not None:
        config_data["to_date"] = to_date

    out_dir = paths.raw_cdx_dir
    report_dir = paths.reports_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Run ID: {run_id if run_id else 'legacy'}")
    logger.info(f"Usando config: {config_path}")
    logger.info(f"Rango: {config_data['from_date']} - {config_data['to_date']}")
    logger.info(f"Output CDX dir: {out_dir}")

    all_frames = []

    for source in config_data["sources"]:
        source_frames = []

        source_name = source["name"]
        domain = source["domain"]
        country = source["country"]

        logger.info(f"Fuente: {source_name} | {domain}")

        days = list(iter_days(config_data["from_date"], config_data["to_date"]))

        for day in tqdm(days, desc=f"CDX {source_name} por día"):
            try:
                df = query_cdx_simple(
                    url_pattern=domain,
                    from_date=day,
                    to_date=day,
                    limit=config_data.get("limit", 100),
                )

                if df.empty:
                    logger.info(f"{source_name} | {day}: 0 snapshots")
                    continue

                df["source"] = source_name
                df["country"] = country
                df["query_day"] = day
                df["year"] = df["timestamp"].astype(str).str[:4]

                source_frames.append(df)

                day_path = out_dir / f"cdx_{source_name}_{day}.parquet"
                df.to_parquet(day_path, index=False)

                logger.info(f"{source_name} | {day}: {len(df):,} snapshots")

            except Exception:
                logger.exception(f"Error CDX | {source_name} | {day}")

        if source_frames:
            source_df = pd.concat(source_frames, ignore_index=True)
            source_df = source_df.drop_duplicates(
                subset=["timestamp", "original", "digest", "source"]
            )

            source_path = out_dir / f"cdx_{source_name}.parquet"
            source_df.to_parquet(source_path, index=False)

            all_frames.append(source_df)

            logger.info(f"{source_name}: total {len(source_df):,} snapshots")
            logger.info(f"Archivo fuente: {source_path}")

    if not all_frames:
        logger.warning("No se obtuvieron snapshots.")
        return

    cdx = pd.concat(all_frames, ignore_index=True)

    cdx = cdx.drop_duplicates(
        subset=["timestamp", "original", "digest", "source"]
    )

    cdx_path = out_dir / "cdx_snapshots.parquet"
    cdx.to_parquet(cdx_path, index=False)

    summary = (
        cdx.groupby(["source", "country", "year"])
        .size()
        .reset_index(name="n_snapshots")
        .sort_values(["source", "year"])
    )

    summary_path = out_dir / "cdx_summary_by_source_year.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    # Copia del resumen en outputs/reports/<run_id>/ para auditoría rápida.
    report_summary_path = report_dir / "cdx_summary_by_source_year.csv"
    summary.to_csv(report_summary_path, index=False, encoding="utf-8-sig")

    logger.info(f"Total snapshots: {len(cdx):,}")
    logger.info(f"Archivo principal: {cdx_path}")
    logger.info(f"Resumen CDX: {summary_path}")
    logger.info(f"Resumen reporte: {report_summary_path}")


if __name__ == "__main__":
    args = parse_args()

    main(
        config=args.config,
        from_date=args.from_date,
        to_date=args.to_date,
        run_id=args.run_id,
    )