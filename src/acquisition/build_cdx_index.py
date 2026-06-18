from __future__ import annotations

from pathlib import Path
import yaml
import pandas as pd
from tqdm import tqdm

from src.acquisition.date_utils import iter_days
from src.utils.logging_config import setup_logger
from src.acquisition.cdx_client import query_cdx_simple


ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(config_name: str = "sources_test.yaml") -> None:
    logger = setup_logger("build_cdx_index", ROOT / "data" / "logs")

    config_path = ROOT / "configs" / config_name
    config = load_yaml(config_path)

    out_dir = ROOT / "data" / "raw_cdx"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Usando config: {config_path}")
    logger.info(f"Rango: {config['from_date']} - {config['to_date']}")

    all_frames = []

    for source in config["sources"]:
        source_frames = []

        logger.info(f"Fuente: {source['name']} | {source['domain']}")

        days = list(iter_days(config["from_date"], config["to_date"]))

        for day in tqdm(days, desc=f"CDX {source['name']} por día"):
            try:
                df = query_cdx_simple(
                    url_pattern=source["domain"],
                    from_date=day,
                    to_date=day,
                    limit=config.get("limit", 100),
                )

                if df.empty:
                    logger.info(f"{source['name']} | {day}: 0 snapshots")
                    continue

                df["source"] = source["name"]
                df["country"] = source["country"]
                df["query_day"] = day
                df["year"] = df["timestamp"].str[:4]

                source_frames.append(df)

                day_path = out_dir / f"cdx_{source['name']}_{day}.parquet"
                df.to_parquet(day_path, index=False)

                logger.info(f"{source['name']} | {day}: {len(df):,} snapshots")

            except Exception:
                logger.exception(f"Error CDX | {source['name']} | {day}")

        if source_frames:
            source_df = pd.concat(source_frames, ignore_index=True)
            source_path = out_dir / f"cdx_{source['name']}.parquet"
            source_df.to_parquet(source_path, index=False)

            all_frames.append(source_df)

            logger.info(f"{source['name']}: total {len(source_df):,} snapshots")
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
    summary.to_csv(summary_path, index=False)

    logger.info(f"Total snapshots: {len(cdx):,}")
    logger.info(f"Archivo principal: {cdx_path}")
    logger.info(f"Resumen: {summary_path}")


if __name__ == "__main__":
    main()