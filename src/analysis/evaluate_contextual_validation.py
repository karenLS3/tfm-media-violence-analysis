from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

VALID_CASE_LABELS = {
    "relevant",
    "not_relevant",
    "uncertain",
}

VALID_RECOGNITION_LABELS = {
    "explicit",
    "contextual",
    "insufficient",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an annotated contextual validation pilot."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "validation"
            / "contextual_validation_pilot.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "validation",
    )
    return parser.parse_args()


def normalize_label(series: pd.Series) -> pd.Series:
    return (
        series
        .fillna("")
        .astype("string")
        .str.strip()
        .str.lower()
    )


def wilson_interval(
    successes: int,
    total: int,
    z: float = 1.96,
) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None

    proportion = successes / total
    denominator = 1 + (z**2 / total)
    centre = (
        proportion
        + z**2 / (2 * total)
    ) / denominator
    margin = (
        z
        * math.sqrt(
            (
                proportion * (1 - proportion)
                + z**2 / (4 * total)
            )
            / total
        )
        / denominator
    )

    return (
        max(0.0, centre - margin),
        min(1.0, centre + margin),
    )


def summarize_group(group: pd.DataFrame) -> pd.Series:
    weights = pd.to_numeric(
        group.get(
            "sample_weight",
            pd.Series(1.0, index=group.index),
        ),
        errors="coerce",
    ).fillna(1.0)

    relevant_mask = group["manual_case_label"].eq(
        "relevant"
    )
    not_relevant_mask = group[
        "manual_case_label"
    ].eq("not_relevant")
    uncertain_mask = group[
        "manual_case_label"
    ].eq("uncertain")

    decisive_mask = relevant_mask | not_relevant_mask

    n_relevant = int(relevant_mask.sum())
    n_not_relevant = int(not_relevant_mask.sum())
    n_uncertain = int(uncertain_mask.sum())
    n_decisive = n_relevant + n_not_relevant

    precision_decisive = (
        n_relevant / n_decisive
        if n_decisive
        else pd.NA
    )

    conservative_precision = (
        n_relevant / len(group)
        if len(group)
        else pd.NA
    )

    weighted_decisive_denominator = weights[
        decisive_mask
    ].sum()

    weighted_precision = (
        weights[relevant_mask].sum()
        / weighted_decisive_denominator
        if weighted_decisive_denominator
        else pd.NA
    )

    ci_low, ci_high = wilson_interval(
        successes=n_relevant,
        total=n_decisive,
    )

    return pd.Series(
        {
            "n_reviewed": len(group),
            "n_relevant": n_relevant,
            "n_not_relevant": n_not_relevant,
            "n_uncertain": n_uncertain,
            "precision_excluding_uncertain": (
                precision_decisive
            ),
            "weighted_precision_excluding_uncertain": (
                weighted_precision
            ),
            "conservative_precision": (
                conservative_precision
            ),
            "wilson_95_low": ci_low,
            "wilson_95_high": ci_high,
        }
    )


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Validation file not found: {args.input}"
        )

    df = pd.read_csv(args.input)

    required = {
        "validation_group",
        "manual_case_label",
        "manual_gender_recognition",
    }
    missing = sorted(required.difference(df.columns))

    if missing:
        raise KeyError(
            "Validation CSV is missing columns: "
            + ", ".join(missing)
        )

    df["manual_case_label"] = normalize_label(
        df["manual_case_label"]
    )
    df["manual_gender_recognition"] = normalize_label(
        df["manual_gender_recognition"]
    )

    reviewed = df[
        df["manual_case_label"].ne("")
    ].copy()

    invalid_case_labels = sorted(
        set(reviewed["manual_case_label"])
        .difference(VALID_CASE_LABELS)
    )

    if invalid_case_labels:
        raise ValueError(
            "Invalid manual_case_label values: "
            + ", ".join(invalid_case_labels)
        )

    recognition_reviewed = reviewed[
        reviewed["manual_gender_recognition"].ne("")
    ]

    invalid_recognition = sorted(
        set(
            recognition_reviewed[
                "manual_gender_recognition"
            ]
        ).difference(VALID_RECOGNITION_LABELS)
    )

    if invalid_recognition:
        raise ValueError(
            "Invalid manual_gender_recognition values: "
            + ", ".join(invalid_recognition)
        )

    if reviewed.empty:
        raise ValueError(
            "No annotated rows were found. Fill "
            "manual_case_label before evaluation."
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_rows = []

    for validation_group, group in reviewed.groupby(
        "validation_group",
        dropna=False,
    ):
        row = summarize_group(group).to_dict()
        row["validation_group"] = validation_group
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    preferred_columns = [
        "validation_group",
        "n_reviewed",
        "n_relevant",
        "n_not_relevant",
        "n_uncertain",
        "precision_excluding_uncertain",
        "weighted_precision_excluding_uncertain",
        "conservative_precision",
        "wilson_95_low",
        "wilson_95_high",
    ]

    summary = summary[
        [
            column
            for column in preferred_columns
            if column in summary.columns
        ]
    ]

    summary.to_csv(
        args.output_dir
        / "contextual_validation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    confusion = pd.crosstab(
        reviewed["validation_group"],
        reviewed["manual_case_label"],
        dropna=False,
    ).reset_index()

    confusion.to_csv(
        args.output_dir
        / "contextual_validation_confusion.csv",
        index=False,
        encoding="utf-8-sig",
    )

    recognition = pd.crosstab(
        recognition_reviewed["validation_group"],
        recognition_reviewed[
            "manual_gender_recognition"
        ],
        dropna=False,
    ).reset_index()

    recognition.to_csv(
        args.output_dir
        / "contextual_validation_recognition.csv",
        index=False,
        encoding="utf-8-sig",
    )

    errors = reviewed[
        reviewed["manual_case_label"].isin(
            ["not_relevant", "uncertain"]
        )
    ].copy()

    errors.to_csv(
        args.output_dir
        / "contextual_validation_errors.csv",
        index=False,
        encoding="utf-8-sig",
    )

    hard_negative = reviewed[
        reviewed["validation_group"].eq(
            "hard_negative"
        )
    ]

    if not hard_negative.empty:
        hard_negative_false_negative_rate = (
            hard_negative["manual_case_label"]
            .eq("relevant")
            .mean()
        )
    else:
        hard_negative_false_negative_rate = pd.NA

    overall = pd.DataFrame(
        [
            {
                "n_rows_in_sample": len(df),
                "n_reviewed": len(reviewed),
                "completion_rate": len(reviewed) / len(df),
                "hard_negative_relevant_rate": (
                    hard_negative_false_negative_rate
                ),
                "note": (
                    "The hard-negative relevant rate is not "
                    "overall recall; it is an audit of difficult "
                    "not-selected records."
                ),
            }
        ]
    )

    overall.to_csv(
        args.output_dir
        / "contextual_validation_overall.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nValidation summary:\n")
    print(summary.to_string(index=False))
    print(
        "\nOverall:\n"
        + overall.to_string(index=False)
    )
    print(
        "\nOutputs saved in:\n"
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()
