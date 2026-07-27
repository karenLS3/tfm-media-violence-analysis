from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construye los totales de noticias clasificadas "
            "por año y categoría."
        )
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2015,
        help="Primer año que se incluirá.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2025,
        help="Último año que se incluirá.",
    )
    return parser.parse_args()


def get_main_category(retrieval_bucket: str) -> str:
    """
    Agrupa las categorías detalladas en categorías principales.
    """
    bucket = str(retrieval_bucket or "")

    if bucket == "case_strong":
        return "case_strong"

    if bucket.startswith("case_review"):
        return "case_review"

    if bucket.startswith("topic"):
        return "topic"

    if bucket == "not_candidate":
        return "not_candidate"

    return "other"


def get_category_priority(retrieval_bucket: str) -> int:
    """
    Prioridad utilizada cuando una misma noticia aparece en varias
    corridas mensuales con clasificaciones diferentes.
    """
    main_category = get_main_category(retrieval_bucket)

    priorities = {
        "case_strong": 4,
        "case_review": 3,
        "topic": 2,
        "not_candidate": 1,
        "other": 0,
    }

    return priorities[main_category]


def first_nonempty(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.Series:
    """
    Devuelve el primer valor no vacío disponible entre varias columnas.
    """
    result = pd.Series("", index=df.index, dtype="string")

    for column in columns:
        if column not in df.columns:
            continue

        values = (
            df[column]
            .fillna("")
            .astype("string")
            .str.strip()
        )

        result = result.mask(result.eq(""), values)

    return result


def load_classified_runs() -> pd.DataFrame:
    """
    Lee todos los articles_classified.parquet existentes
    dentro de data/runs.
    """
    paths = sorted(
        (ROOT / "data" / "runs").glob(
            "*/processed/articles_classified.parquet"
        )
    )

    if not paths:
        raise FileNotFoundError(
            "No se encontraron archivos en:\n"
            "data/runs/*/processed/articles_classified.parquet"
        )

    frames: list[pd.DataFrame] = []

    for path in paths:
        frame = pd.read_parquet(path)

        # data/runs/<run_id>/processed/articles_classified.parquet
        frame["run_id"] = path.parents[1].name

        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def build_archive_year(df: pd.DataFrame) -> pd.Series:
    """
    Obtiene el año de la captura de Wayback Machine.

    Primero usa snapshot_timestamp. Si falta, intenta obtenerlo
    desde run_id, por ejemplo 2015_03 o 2015.
    """
    if "snapshot_timestamp" in df.columns:
        snapshot_year = (
            df["snapshot_timestamp"]
            .fillna("")
            .astype("string")
            .str.extract(r"^(\d{4})", expand=False)
        )
    else:
        snapshot_year = pd.Series(
            pd.NA,
            index=df.index,
            dtype="string",
        )

    run_year = (
        df["run_id"]
        .astype("string")
        .str.extract(r"^(\d{4})", expand=False)
    )

    year = snapshot_year.fillna(run_year)

    return pd.to_numeric(
        year,
        errors="coerce",
    ).astype("Int64")


def deduplicate_news(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina la misma noticia cuando aparece en varias corridas
    mensuales del mismo año.

    Se utiliza normalized_url y, si falta, candidate_url.
    """
    df = df.copy()

    df["_article_url"] = first_nonempty(
        df,
        ["normalized_url", "candidate_url"],
    )

    if "country" in df.columns:
        country = (
            df["country"]
            .fillna("")
            .astype("string")
            .str.strip()
        )
    else:
        country = pd.Series(
            "",
            index=df.index,
            dtype="string",
        )

    if "source" in df.columns:
        source = (
            df["source"]
            .fillna("")
            .astype("string")
            .str.strip()
        )
    else:
        source = pd.Series(
            "",
            index=df.index,
            dtype="string",
        )

    df["_article_key"] = (
        country
        + "|"
        + source
        + "|"
        + df["_article_url"]
    )

    # No combinar accidentalmente registros que no tengan URL.
    missing_url = df["_article_url"].eq("")

    df.loc[missing_url, "_article_key"] = (
        df.loc[missing_url, "run_id"].astype("string")
        + "|row|"
        + df.index[missing_url].astype("string")
    )

    df["_category_priority"] = (
        df["retrieval_bucket"]
        .map(get_category_priority)
    )

    if "text_length" not in df.columns:
        df["text_length"] = 0

    df["text_length"] = pd.to_numeric(
        df["text_length"],
        errors="coerce",
    ).fillna(0)

    # Si una noticia fue clasificada de forma diferente en dos meses,
    # se conserva la clasificación con evidencia más fuerte.
    # En caso de empate, se conserva la extracción con más texto.
    unique_news = (
        df.sort_values(
            [
                "archive_year",
                "_article_key",
                "_category_priority",
                "text_length",
            ],
            ascending=[True, True, False, False],
        )
        .drop_duplicates(
            subset=[
                "archive_year",
                "_article_key",
            ],
            keep="first",
        )
        .copy()
    )

    return unique_news


def main() -> None:
    args = parse_args()

    df = load_classified_runs()

    if "retrieval_bucket" not in df.columns:
        raise KeyError(
            "El dataset no contiene la columna retrieval_bucket."
        )

    df["archive_year"] = build_archive_year(df)

    df = df[
        df["archive_year"].between(
            args.start_year,
            args.end_year,
            inclusive="both",
        )
    ].copy()

    df["retrieval_bucket"] = (
        df["retrieval_bucket"]
        .fillna("unknown")
        .astype("string")
    )

    df["main_category"] = (
        df["retrieval_bucket"]
        .map(get_main_category)
    )

    rows_before_deduplication = len(df)

    unique_news = deduplicate_news(df)

    # ---------------------------------------------------------
    # Tabla detallada: una fila por año y retrieval_bucket
    # ---------------------------------------------------------

    detailed_summary = (
        unique_news
        .groupby(
            [
                "archive_year",
                "retrieval_bucket",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="n_unique_news")
        .sort_values(
            [
                "archive_year",
                "retrieval_bucket",
            ]
        )
    )

    # ---------------------------------------------------------
    # Tabla detallada en formato ancho
    # ---------------------------------------------------------

    detailed_wide = (
        detailed_summary
        .pivot(
            index="archive_year",
            columns="retrieval_bucket",
            values="n_unique_news",
        )
        .fillna(0)
        .astype(int)
        .reset_index()
    )

    detailed_wide.columns.name = None

    # ---------------------------------------------------------
    # Categorías principales
    # ---------------------------------------------------------

    main_summary = (
        unique_news
        .groupby(
            [
                "archive_year",
                "main_category",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="n_unique_news")
    )

    main_wide = (
        main_summary
        .pivot(
            index="archive_year",
            columns="main_category",
            values="n_unique_news",
        )
        .fillna(0)
        .astype(int)
        .reset_index()
    )

    main_wide.columns.name = None

    expected_columns = [
        "case_strong",
        "case_review",
        "topic",
        "not_candidate",
        "other",
    ]

    for column in expected_columns:
        if column not in main_wide.columns:
            main_wide[column] = 0

    main_wide["all_case_candidates"] = (
        main_wide["case_strong"]
        + main_wide["case_review"]
    )

    main_wide["total_unique_news"] = (
        main_wide[expected_columns].sum(axis=1)
    )

    main_wide = main_wide[
        [
            "archive_year",
            "case_strong",
            "case_review",
            "all_case_candidates",
            "topic",
            "not_candidate",
            "other",
            "total_unique_news",
        ]
    ].sort_values("archive_year")

    # ---------------------------------------------------------
    # Auditoría de duplicados
    # ---------------------------------------------------------

    audit = pd.DataFrame(
        [
            {
                "start_year": args.start_year,
                "end_year": args.end_year,
                "classified_rows_before_deduplication": (
                    rows_before_deduplication
                ),
                "unique_news_after_deduplication": len(unique_news),
                "duplicates_removed": (
                    rows_before_deduplication
                    - len(unique_news)
                ),
            }
        ]
    )

    # ---------------------------------------------------------
    # Guardar reportes
    # ---------------------------------------------------------

    output_dir = ROOT / "outputs" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    by_source = (
        unique_news
        .groupby(
            [
                "archive_year",
                "country",
                "source",
                "retrieval_bucket",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="n_unique_news")
        .sort_values(
            [
                "archive_year",
                "country",
                "source",
                "retrieval_bucket",
            ]
        )
    )

    by_source.to_csv(
        output_dir / "news_by_year_country_source_category.csv",
        index=False,
        encoding="utf-8-sig",
    )

    coverage = (
        unique_news
        .groupby(
            [
                "archive_year",
                "country",
                "source",
            ],
            dropna=False,
        )
        .agg(
            total_unique_news=("_article_key", "nunique"),
            case_strong=(
                "retrieval_bucket",
                lambda values: (values == "case_strong").sum(),
            ),
            case_review=(
                "retrieval_bucket",
                lambda values: values.str.startswith(
                    "case_review",
                    na=False,
                ).sum(),
            ),
            topic=(
                "retrieval_bucket",
                lambda values: values.str.startswith(
                    "topic",
                    na=False,
                ).sum(),
            ),
        )
        .reset_index()
    )

    coverage["case_candidate_rate"] = (
        (
            coverage["case_strong"]
            + coverage["case_review"]
        )
        / coverage["total_unique_news"]
    )

    coverage.to_csv(
        output_dir / "coverage_by_year_country_source.csv",
        index=False,
        encoding="utf-8-sig",
    )

    detailed_summary.to_csv(
        output_dir
        / "news_by_year_and_retrieval_bucket_long.csv",
        index=False,
        encoding="utf-8-sig",
    )

    detailed_wide.to_csv(
        output_dir
        / "news_by_year_and_retrieval_bucket_wide.csv",
        index=False,
        encoding="utf-8-sig",
    )

    main_wide.to_csv(
        output_dir
        / "news_by_year_and_main_category.csv",
        index=False,
        encoding="utf-8-sig",
    )

    audit.to_csv(
        output_dir
        / "news_by_year_deduplication_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nTotales de noticias únicas por año:\n")
    print(main_wide.to_string(index=False))

    print(
        "\nAuditoría de deduplicación:\n"
        f"- Filas clasificadas: "
        f"{rows_before_deduplication:,}\n"
        f"- Noticias únicas: "
        f"{len(unique_news):,}\n"
        f"- Duplicados eliminados: "
        f"{rows_before_deduplication - len(unique_news):,}"
    )

    print(
        f"\nReportes guardados en:\n{output_dir}"
    )


if __name__ == "__main__":
    main()