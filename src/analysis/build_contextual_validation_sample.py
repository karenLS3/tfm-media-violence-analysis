from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TARGETS = {
    "explicit_case_control": 30,
    "explicit_topic_control": 20,
    "contextual_high": 60,
    "contextual_medium": 80,
    "ambiguous_case": 40,
    "hard_negative": 20,
}

MANUAL_COLUMNS = [
    "manual_case_label",
    "manual_gender_recognition",
    "manual_error_type",
    "manual_notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build contextual-classification audit tables and a "
            "stratified pilot validation sample."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "reports"
            / "articles_contextual_unique.parquet"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "validation",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--explicit-case-size",
        type=int,
        default=DEFAULT_TARGETS["explicit_case_control"],
    )
    parser.add_argument(
        "--explicit-topic-size",
        type=int,
        default=DEFAULT_TARGETS["explicit_topic_control"],
    )
    parser.add_argument(
        "--high-size",
        type=int,
        default=DEFAULT_TARGETS["contextual_high"],
    )
    parser.add_argument(
        "--medium-size",
        type=int,
        default=DEFAULT_TARGETS["contextual_medium"],
    )
    parser.add_argument(
        "--ambiguous-size",
        type=int,
        default=DEFAULT_TARGETS["ambiguous_case"],
    )
    parser.add_argument(
        "--hard-negative-size",
        type=int,
        default=DEFAULT_TARGETS["hard_negative"],
    )
    return parser.parse_args()


def existing_columns(
    df: pd.DataFrame,
    columns: list[str],
) -> list[str]:
    return [
        column
        for column in columns
        if column in df.columns
    ]


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


def add_text_excerpt(
    df: pd.DataFrame,
    max_chars: int = 800,
) -> pd.DataFrame:
    df = df.copy()

    if "text_excerpt" in df.columns:
        excerpt = (
            df["text_excerpt"]
            .fillna("")
            .astype("string")
        )
    else:
        excerpt = first_nonempty(
            df,
            ["text_clean", "text", "text_raw"],
        )

    df["text_excerpt"] = (
        excerpt
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.slice(0, max_chars)
    )

    return df


def balanced_round_robin_sample(
    df: pd.DataFrame,
    n: int,
    strata_columns: list[str],
    seed: int,
) -> pd.DataFrame:
    """
    Selecciona una muestra aproximadamente equilibrada por año, país y fuente.

    La muestra se equilibra de forma intencional para facilitar la detección
    de errores. Más adelante se incorporan ponderaciones poblacionales, de modo
    que las estimaciones agregadas también puedan reflejar la distribución
    original.
    """
    if n <= 0 or df.empty:
        return df.iloc[0:0].copy()

    if len(df) <= n:
        return df.copy()

    working = df.copy()
    available_strata = existing_columns(
        working,
        strata_columns,
    )

    if not available_strata:
        return working.sample(
            n=n,
            random_state=seed,
        )

    shuffled_groups: list[list[int]] = []

    grouped = working.groupby(
        available_strata,
        dropna=False,
        sort=True,
    )

    for group_number, (_, group) in enumerate(grouped):
        shuffled = group.sample(
            frac=1,
            random_state=seed + group_number,
        )
        shuffled_groups.append(shuffled.index.tolist())

    selected_indices: list[int] = []
    position = 0

    while len(selected_indices) < n:
        made_progress = False

        for group_indices in shuffled_groups:
            if position < len(group_indices):
                selected_indices.append(
                    group_indices[position]
                )
                made_progress = True

                if len(selected_indices) == n:
                    break

        if not made_progress:
            break

        position += 1

    return working.loc[selected_indices].copy()


def build_hard_negative_pool(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Selecciona registros negativos difíciles, en lugar de tomar una muestra
    puramente aleatoria dominada por noticias claramente no relacionadas.
    """
    pool = df[
        df["provisional_case_status"].eq("not_selected")
    ].copy()

    if "retrieval_bucket" in pool.columns:
        pool = pool[
            pool["retrieval_bucket"].eq("not_candidate")
        ].copy()

    signal_columns = existing_columns(
        pool,
        [
            "matched_terms_json",
            "match_reasons_json",
            "contextual_reasons_json",
            "title_clean",
            "title",
            "anchor_text",
        ],
    )

    if not signal_columns:
        return pool

    combined_signal = (
        pool[signal_columns]
        .fillna("")
        .astype("string")
        .agg(" ".join, axis=1)
        .str.lower()
    )

    hard_pattern = re.compile(
        r"female|mujer|mujeres|joven|jovenes|"
        r"victim|muerte|muert|asesin|homicid|"
        r"violencia|agres|desapare|pareja|expareja"
    )

    hard_mask = combined_signal.str.contains(
        hard_pattern,
        regex=True,
        na=False,
    )

    hard_pool = pool[hard_mask].copy()

    # Si las columnas del proyecto no brindan suficiente información para construir
    # un conjunto estricto, se usan como alternativa todas las filas no seleccionadas o no candidatas.
    return hard_pool if not hard_pool.empty else pool


def select_group(
    df: pd.DataFrame,
    validation_group: str,
    target_size: int,
    seed: int,
) -> pd.DataFrame:
    status_map = {
        "explicit_case_control": "candidate_explicit_case",
        "explicit_topic_control": "topic_explicit_not_case",
        "contextual_high": "candidate_contextual_high",
        "contextual_medium": "review_contextual_medium",
        "ambiguous_case": "review_ambiguous_case",
    }

    if validation_group == "hard_negative":
        pool = build_hard_negative_pool(df)
    else:
        pool = df[
            df["provisional_case_status"].eq(
                status_map[validation_group]
            )
        ].copy()

    sample = balanced_round_robin_sample(
        pool,
        n=target_size,
        strata_columns=[
            "archive_year",
            "country",
            "source",
        ],
        seed=seed,
    )

    sample["validation_group"] = validation_group
    return sample


def add_population_weights(
    sample: pd.DataFrame,
    population: pd.DataFrame,
) -> pd.DataFrame:
    sample = sample.copy()

    strata = existing_columns(
        sample,
        [
            "validation_group",
            "archive_year",
            "country",
            "source",
        ],
    )

    if not strata:
        sample["population_stratum_size"] = len(population)
        sample["sample_stratum_size"] = len(sample)
        sample["sample_weight"] = 1.0
        return sample

    population_frames = []

    for validation_group in sample[
        "validation_group"
    ].dropna().unique():
        if validation_group == "hard_negative":
            group_population = build_hard_negative_pool(
                population
            )
        else:
            group_status = {
                "explicit_case_control": (
                    "candidate_explicit_case"
                ),
                "explicit_topic_control": (
                    "topic_explicit_not_case"
                ),
                "contextual_high": (
                    "candidate_contextual_high"
                ),
                "contextual_medium": (
                    "review_contextual_medium"
                ),
                "ambiguous_case": (
                    "review_ambiguous_case"
                ),
            }[validation_group]

            group_population = population[
                population["provisional_case_status"]
                .eq(group_status)
            ].copy()

        group_population["validation_group"] = (
            validation_group
        )
        population_frames.append(group_population)

    weighted_population = pd.concat(
        population_frames,
        ignore_index=True,
    )

    population_counts = (
        weighted_population.groupby(
            strata,
            dropna=False,
        )
        .size()
        .reset_index(name="population_stratum_size")
    )

    sample_counts = (
        sample.groupby(
            strata,
            dropna=False,
        )
        .size()
        .reset_index(name="sample_stratum_size")
    )

    sample = (
        sample
        .merge(
            population_counts,
            on=strata,
            how="left",
        )
        .merge(
            sample_counts,
            on=strata,
            how="left",
        )
    )

    sample["sample_weight"] = (
        sample["population_stratum_size"]
        / sample["sample_stratum_size"]
    )

    return sample


def save_crosstabs(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    status_by_bucket = pd.crosstab(
        df["retrieval_bucket"],
        df["provisional_case_status"],
        dropna=False,
    ).reset_index()

    status_by_bucket.to_csv(
        output_dir
        / "provisional_status_by_retrieval_bucket.csv",
        index=False,
        encoding="utf-8-sig",
    )

    recognition_by_status = pd.crosstab(
        df["provisional_case_status"],
        df["gender_recognition_mode"],
        dropna=False,
    ).reset_index()

    recognition_by_status.to_csv(
        output_dir
        / "recognition_mode_by_provisional_status.csv",
        index=False,
        encoding="utf-8-sig",
    )

    contextual_level_by_bucket = pd.crosstab(
        df["retrieval_bucket"],
        df["contextual_evidence_level"],
        dropna=False,
    ).reset_index()

    contextual_level_by_bucket.to_csv(
        output_dir
        / "contextual_level_by_retrieval_bucket.csv",
        index=False,
        encoding="utf-8-sig",
    )

    by_source = (
        df.groupby(
            [
                "archive_year",
                "country",
                "source",
                "provisional_case_status",
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
                "provisional_case_status",
            ]
        )
    )

    by_source.to_csv(
        output_dir
        / "provisional_status_by_year_country_source.csv",
        index=False,
        encoding="utf-8-sig",
    )


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {args.input}"
        )

    df = pd.read_parquet(args.input)
    df = add_text_excerpt(df)

    required = {
        "retrieval_bucket",
        "provisional_case_status",
        "gender_recognition_mode",
        "contextual_evidence_level",
    }
    missing = sorted(required.difference(df.columns))

    if missing:
        raise KeyError(
            "The contextual dataset is missing columns: "
            + ", ".join(missing)
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_crosstabs(
        df=df,
        output_dir=args.output_dir,
    )

    targets = {
        "explicit_case_control": args.explicit_case_size,
        "explicit_topic_control": args.explicit_topic_size,
        "contextual_high": args.high_size,
        "contextual_medium": args.medium_size,
        "ambiguous_case": args.ambiguous_size,
        "hard_negative": args.hard_negative_size,
    }

    samples = []

    for group_number, (
        validation_group,
        target_size,
    ) in enumerate(targets.items()):
        group_sample = select_group(
            df=df,
            validation_group=validation_group,
            target_size=target_size,
            seed=args.seed + 1000 * group_number,
        )
        samples.append(group_sample)

    sample = pd.concat(
        samples,
        ignore_index=True,
    )

    sample = add_population_weights(
        sample=sample,
        population=df,
    )

    for column in MANUAL_COLUMNS:
        sample[column] = ""

    display_columns = [
        "validation_group",
        "archive_year",
        "country",
        "source",
        "run_id",
        "title_clean",
        "title",
        "anchor_text",
        "normalized_url",
        "candidate_url",
        "retrieval_bucket",
        "explicit_label_type",
        "contextual_evidence_level",
        "contextual_reasons_json",
        "contextual_evidence_json",
        "gender_recognition_mode",
        "provisional_case_status",
        "review_priority",
        "snapshot_count",
        "first_snapshot_datetime",
        "last_snapshot_datetime",
        "text_excerpt",
        "population_stratum_size",
        "sample_stratum_size",
        "sample_weight",
        *MANUAL_COLUMNS,
    ]

    sample = sample[
        existing_columns(sample, display_columns)
    ]

    sample.to_csv(
        args.output_dir
        / "contextual_validation_pilot.csv",
        index=False,
        encoding="utf-8-sig",
    )

    sample_distribution = (
        sample.groupby(
            [
                "validation_group",
                "archive_year",
                "country",
                "source",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="n_sampled")
    )

    sample_distribution.to_csv(
        args.output_dir
        / "contextual_validation_pilot_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nValidation pilot created.")
    print(
        sample["validation_group"]
        .value_counts()
        .rename_axis("validation_group")
        .reset_index(name="n")
        .to_string(index=False)
    )
    print(
        "\nMain file:\n"
        f"{args.output_dir / 'contextual_validation_pilot.csv'}"
    )
    print(
        "\nAudit cross-tabs saved in:\n"
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()
