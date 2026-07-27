from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

from src.analysis.contextual_case_classifier import (
    build_prepared_contextual_lexicon,
    classify_contextual_case,
)
from src.filtering.article_classifier import load_lexicon


ROOT = Path(__file__).resolve().parents[2]

STATUS_RANK = {
    "candidate_explicit_case": 5,
    "candidate_contextual_high": 4,
    "review_contextual_medium": 3,
    "review_ambiguous_case": 2,
    "topic_explicit_not_case": 1,
    "not_selected": 0,
}

PRIORITY_RANK = {
    "P0_explicit_sample_only": 0,
    "P1_contextual_high": 1,
    "P2_contextual_medium": 2,
    "P3_ambiguous_case": 3,
    "P4_not_selected": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an article-level contextual gender-violence dataset "
            "without overwriting retrieval_bucket."
        )
    )
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument(
        "--max-excerpt-chars",
        type=int,
        default=600,
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def load_classified_runs() -> pd.DataFrame:
    paths = sorted(
        (ROOT / "data" / "runs").glob(
            "*/processed/articles_classified.parquet"
        )
    )

    if not paths:
        raise FileNotFoundError(
            "No files found at "
            "data/runs/*/processed/articles_classified.parquet"
        )

    frames: list[pd.DataFrame] = []

    for path in paths:
        frame = pd.read_parquet(path)
        frame["run_id"] = path.parents[1].name
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def first_nonempty(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.Series:
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


def build_archive_year(df: pd.DataFrame) -> pd.Series:
    if "snapshot_timestamp" in df.columns:
        timestamp_year = (
            df["snapshot_timestamp"]
            .fillna("")
            .astype("string")
            .str.extract(r"^(\d{4})", expand=False)
        )
    else:
        timestamp_year = pd.Series(
            pd.NA,
            index=df.index,
            dtype="string",
        )

    run_year = (
        df["run_id"]
        .astype("string")
        .str.extract(r"^(\d{4})", expand=False)
    )

    return pd.to_numeric(
        timestamp_year.fillna(run_year),
        errors="coerce",
    ).astype("Int64")


def build_snapshot_datetime(df: pd.DataFrame) -> pd.Series:
    if "snapshot_timestamp" not in df.columns:
        return pd.Series(pd.NaT, index=df.index)

    values = (
        df["snapshot_timestamp"]
        .fillna("")
        .astype("string")
        .str.extract(r"^(\d{14})", expand=False)
    )

    return pd.to_datetime(
        values,
        format="%Y%m%d%H%M%S",
        errors="coerce",
    )


def add_article_key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["_article_url"] = first_nonempty(
        df,
        [
            "normalized_url",
            "candidate_url",
            "original_url",
            "url",
        ],
    )

    country = first_nonempty(df, ["country"])
    source = first_nonempty(df, ["source"])

    df["_article_key"] = (
        country
        + "|"
        + source
        + "|"
        + df["_article_url"]
    )

    missing_url = df["_article_url"].eq("")

    df.loc[missing_url, "_article_key"] = (
        df.loc[missing_url, "run_id"].astype("string")
        + "|row|"
        + df.index[missing_url].astype("string")
    )

    return df


def enrich_contextual(
    df: pd.DataFrame,
    prepared_lexicon: dict,
) -> pd.DataFrame:
    results = []

    for row in tqdm(
        df.to_dict(orient="records"),
        total=len(df),
        desc="Contextual classification",
    ):
        results.append(
            classify_contextual_case(
                row=row,
                prepared_lexicon=prepared_lexicon,
            )
        )

    return pd.concat(
        [
            df.reset_index(drop=True),
            pd.DataFrame(results),
        ],
        axis=1,
    )


def deduplicate_articles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera un único registro por artículo y año de archivo, conservando
    las estadísticas de las capturas.

    El conjunto de datos a nivel de captura se guarda por separado y nunca
    se elimina.
    """
    df = df.copy()

    df["_status_rank"] = (
        df["provisional_case_status"]
        .map(STATUS_RANK)
        .fillna(-1)
    )

    if "text_length" not in df.columns:
        df["text_length"] = first_nonempty(
            df,
            ["text_clean", "text", "text_raw"],
        ).str.len()

    df["text_length"] = pd.to_numeric(
        df["text_length"],
        errors="coerce",
    ).fillna(0)

    snapshot_stats = (
        df.groupby(
            ["archive_year", "_article_key"],
            dropna=False,
        )
        .agg(
            snapshot_count=("_article_key", "size"),
            first_snapshot_datetime=(
                "snapshot_datetime",
                "min",
            ),
            last_snapshot_datetime=(
                "snapshot_datetime",
                "max",
            ),
        )
        .reset_index()
    )

    unique = (
        df.sort_values(
            [
                "archive_year",
                "_article_key",
                "_status_rank",
                "contextual_score",
                "text_length",
            ],
            ascending=[
                True,
                True,
                False,
                False,
                False,
            ],
        )
        .drop_duplicates(
            subset=["archive_year", "_article_key"],
            keep="first",
        )
        .merge(
            snapshot_stats,
            on=["archive_year", "_article_key"],
            how="left",
        )
        .copy()
    )

    return unique


def add_excerpt(
    df: pd.DataFrame,
    max_chars: int,
) -> pd.DataFrame:
    df = df.copy()

    body = first_nonempty(
        df,
        ["text_clean", "text", "text_raw"],
    )

    df["text_excerpt"] = (
        body
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.slice(0, max_chars)
    )

    return df


def existing_columns(
    df: pd.DataFrame,
    requested: list[str],
) -> list[str]:
    return [
        column
        for column in requested
        if column in df.columns
    ]


def save_summary(
    df: pd.DataFrame,
    group_columns: list[str],
    category_column: str,
    output_path: Path,
) -> pd.DataFrame:
    summary = (
        df.groupby(
            group_columns + [category_column],
            dropna=False,
        )
        .size()
        .reset_index(name="n_unique_news")
        .sort_values(group_columns + [category_column])
    )

    summary.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    return summary


def main() -> None:
    args = parse_args()

    retrieval_lexicon = load_lexicon(
        ROOT / "configs" / "lexicon_retrieval.yml"
    )
    contextual_lexicon = load_yaml(
        ROOT / "configs" / "lexicon_contextual.yml"
    )

    prepared_lexicon = build_prepared_contextual_lexicon(
        retrieval_lexicon=retrieval_lexicon,
        contextual_lexicon=contextual_lexicon,
    )

    snapshots = load_classified_runs()
    snapshots["archive_year"] = build_archive_year(snapshots)
    snapshots["snapshot_datetime"] = build_snapshot_datetime(
        snapshots
    )

    snapshots = snapshots[
        snapshots["archive_year"].between(
            args.start_year,
            args.end_year,
            inclusive="both",
        )
    ].copy()

    snapshots = add_article_key(snapshots)
    snapshots = enrich_contextual(
        snapshots,
        prepared_lexicon,
    )

    unique = deduplicate_articles(snapshots)
    unique = add_excerpt(
        unique,
        max_chars=args.max_excerpt_chars,
    )

    reports_dir = ROOT / "outputs" / "reports"
    tables_dir = ROOT / "outputs" / "tables"
    audits_dir = ROOT / "outputs" / "audits"

    for directory in [
        reports_dir,
        tables_dir,
        audits_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    # Se conservan ambos niveles:
    # 1) todas las capturas de Wayback;
    # 2) un único registro por artículo y año de archivo.
    snapshots.to_parquet(
        reports_dir / "articles_contextual_snapshots.parquet",
        index=False,
    )
    unique.to_parquet(
        reports_dir / "articles_contextual_unique.parquet",
        index=False,
    )

    save_summary(
        unique,
        group_columns=["archive_year"],
        category_column="gender_recognition_mode",
        output_path=(
            tables_dir / "gender_recognition_by_year.csv"
        ),
    )

    save_summary(
        unique,
        group_columns=[
            "archive_year",
            "country",
            "source",
        ],
        category_column="gender_recognition_mode",
        output_path=(
            tables_dir
            / "gender_recognition_by_year_country_source.csv"
        ),
    )

    save_summary(
        unique,
        group_columns=["archive_year"],
        category_column="provisional_case_status",
        output_path=(
            tables_dir / "contextual_case_status_by_year.csv"
        ),
    )

    coverage = (
        unique.groupby(
            ["archive_year", "country", "source"],
            dropna=False,
        )
        .agg(
            total_unique_news=("_article_key", "nunique"),
            explicit_case_candidates=(
                "provisional_case_status",
                lambda values: (
                    values == "candidate_explicit_case"
                ).sum(),
            ),
            contextual_high=(
                "provisional_case_status",
                lambda values: (
                    values == "candidate_contextual_high"
                ).sum(),
            ),
            contextual_medium=(
                "provisional_case_status",
                lambda values: (
                    values == "review_contextual_medium"
                ).sum(),
            ),
            ambiguous_cases=(
                "provisional_case_status",
                lambda values: (
                    values == "review_ambiguous_case"
                ).sum(),
            ),
        )
        .reset_index()
    )

    coverage["implicit_candidate_rate"] = (
        (
            coverage["contextual_high"]
            + coverage["contextual_medium"]
        )
        / coverage["total_unique_news"].replace(0, pd.NA)
    )

    coverage.to_csv(
        tables_dir
        / "contextual_coverage_by_year_country_source.csv",
        index=False,
        encoding="utf-8-sig",
    )

    report_columns = [
        "archive_year",
        "country",
        "source",
        "run_id",
        "snapshot_timestamp",
        "title_clean",
        "title",
        "anchor_text",
        "normalized_url",
        "candidate_url",
        "retrieval_bucket",
        "match_reasons_json",
        "explicit_label_type",
        "explicit_label_terms_json",
        "contextual_evidence_level",
        "contextual_reasons_json",
        "contextual_all_reasons_json",
        "violence_direction",
        "contextual_violence_types_json",
        "contextual_evidence_json",
        "gender_recognition_mode",
        "provisional_case_status",
        "review_priority",
        "snapshot_count",
        "first_snapshot_datetime",
        "last_snapshot_datetime",
        "text_excerpt",
    ]

    implicit = unique[
        unique["is_contextual_without_label"] == True
    ].copy()

    implicit[
        existing_columns(implicit, report_columns)
    ].to_csv(
        reports_dir
        / "contextual_without_explicit_label.csv",
        index=False,
        encoding="utf-8-sig",
    )

    review_queue = unique[
        unique["review_priority"].isin(
            [
                "P1_contextual_high",
                "P2_contextual_medium",
                "P3_ambiguous_case",
            ]
        )
    ].copy()

    review_queue["_priority_rank"] = (
        review_queue["review_priority"]
        .map(PRIORITY_RANK)
        .fillna(99)
    )

    review_queue = review_queue.sort_values(
        [
            "_priority_rank",
            "archive_year",
            "country",
            "source",
            "contextual_score",
        ],
        ascending=[True, True, True, True, False],
    )

    # Campos vacíos para una validación manual acotada y dirigida.
    review_queue["manual_case_label"] = ""
    review_queue["manual_gender_recognition"] = ""
    review_queue["manual_notes"] = ""

    review_output_columns = report_columns + [
        "manual_case_label",
        "manual_gender_recognition",
        "manual_notes",
    ]

    review_queue[
        existing_columns(review_queue, review_output_columns)
    ].to_csv(
        reports_dir / "contextual_review_queue.csv",
        index=False,
        encoding="utf-8-sig",
    )

    audit = pd.DataFrame(
        [
            {
                "start_year": args.start_year,
                "end_year": args.end_year,
                "snapshot_rows": len(snapshots),
                "unique_article_year_rows": len(unique),
                "duplicate_snapshots_collapsed_for_counts": (
                    len(snapshots) - len(unique)
                ),
                "candidate_explicit_case": (
                    unique["provisional_case_status"]
                    == "candidate_explicit_case"
                ).sum(),
                "candidate_contextual_high": (
                    unique["provisional_case_status"]
                    == "candidate_contextual_high"
                ).sum(),
                "review_contextual_medium": (
                    unique["provisional_case_status"]
                    == "review_contextual_medium"
                ).sum(),
                "review_ambiguous_case": (
                    unique["provisional_case_status"]
                    == "review_ambiguous_case"
                ).sum(),
                "topic_explicit_not_case": (
                    unique["provisional_case_status"]
                    == "topic_explicit_not_case"
                ).sum(),
                "not_selected": (
                    unique["provisional_case_status"]
                    == "not_selected"
                ).sum(),
                "reverse_direction_detected": (
                    unique["violence_direction"]
                    == "female_to_male"
                ).sum(),

                "possible_male_victim_detected": (
                    unique["violence_direction"]
                    == "possible_male_victim"
                ).sum(),

                "mixed_direction_detected": (
                    unique["violence_direction"]
                    == "mixed"
                ).sum(),
            }
        ]
    )

    audit.to_csv(
        audits_dir / "contextual_classification_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nContextual classification completed.")
    print(audit.to_string(index=False))
    print(
        "\nMain outputs:\n"
        f"- {reports_dir / 'articles_contextual_snapshots.parquet'}\n"
        f"- {reports_dir / 'articles_contextual_unique.parquet'}\n"
        f"- {reports_dir / 'contextual_without_explicit_label.csv'}\n"
        f"- {reports_dir / 'contextual_review_queue.csv'}\n"
        f"- {tables_dir / 'gender_recognition_by_year.csv'}\n"
        f"- {tables_dir / 'gender_recognition_by_year_country_source.csv'}"
    )


if __name__ == "__main__":
    main()
