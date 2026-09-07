from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "analysis.yml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera tablas estructurales de Fase 1 desde el corpus canónico."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--snapshot", default=None, help="Ejemplo: 2015_2021")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def counts(df: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    return (
        df.groupby(group_columns, dropna=False)
        .agg(
            n_articles=("analysis_identity", "nunique"),
            total_words=("word_count", "sum"),
            median_words=("word_count", "median"),
        )
        .reset_index()
        .sort_values(group_columns)
    )


def save(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    config = load_config(resolve_path(args.config))
    snapshot = args.snapshot or str(config["study"]["snapshot_id"])
    snapshot_dir = resolve_path(config["output"]["root_dir"]) / snapshot
    input_path = snapshot_dir / "articles.parquet"
    tables_dir = snapshot_dir / "tables"

    if not input_path.exists():
        raise FileNotFoundError(
            f"No se encontró {input_path}. Ejecuta primero build_analysis_corpus."
        )

    tables_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(input_path)

    save(counts(df, ["analysis_year"]), tables_dir / "corpus_by_year.csv")
    save(
        counts(df, ["analysis_year", "country"]),
        tables_dir / "corpus_by_year_country.csv",
    )
    save(
        counts(df, ["analysis_year", "country", "source"]),
        tables_dir / "corpus_by_year_country_source.csv",
    )
    save(
        counts(df, ["country", "source"]),
        tables_dir / "corpus_by_country_source.csv",
    )
    save(
        counts(df, ["analysis_period", "country"]),
        tables_dir / "corpus_by_period_country.csv",
    )

    time_source = (
        df.groupby(["country", "source", "analysis_year_source"], dropna=False)
        .size()
        .reset_index(name="n_articles")
    )
    time_source["source_total"] = time_source.groupby(
        ["country", "source"], dropna=False
    )["n_articles"].transform("sum")
    time_source["percentage"] = (
        100 * time_source["n_articles"] / time_source["source_total"]
    )
    save(time_source, tables_dir / "analysis_time_source_by_media.csv")

    print("\nTablas estructurales de Fase 1 generadas.")
    print(f"Entrada: {input_path}")
    print(f"Salida: {tables_dir}")


if __name__ == "__main__":
    main()
