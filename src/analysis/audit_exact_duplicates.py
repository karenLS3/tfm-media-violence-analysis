from pathlib import Path

import pandas as pd


INPUT = Path(
    "outputs/final/case_articles_analysis.parquet"
)

OUTPUT = Path(
    "outputs/final/quality_repaired/"
    "exact_duplicate_groups.csv"
)


def main():

    df = pd.read_parquet(INPUT)

    if "analysis_text" not in df.columns:
        raise ValueError(
            "No existe la columna analysis_text."
        )

    # Normalización mínima exclusivamente para detectar
    # cuerpos textuales idénticos.
    normalized = (
        df["analysis_text"]
        .fillna("")
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    df = df.copy()

    df["_duplicate_text"] = normalized

    duplicated = df[
        df.duplicated(
            "_duplicate_text",
            keep=False
        )
        &
        df["_duplicate_text"].ne("")
    ].copy()

    # Crear identificador de grupo
    duplicated["duplicate_group"] = (
        duplicated
        .groupby("_duplicate_text")
        .ngroup()
        + 1
    )

    duplicated["duplicate_group_size"] = (
        duplicated
        .groupby("duplicate_group")[
            "duplicate_group"
        ]
        .transform("size")
    )

    columns = [
        c
        for c in [
            "duplicate_group",
            "duplicate_group_size",
            "country",
            "source",
            "newspaper",
            "year",
            "date",
            "published_at",
            "title",
            "candidate_url",
            "normalized_url",
            "analysis_text",
        ]
        if c in duplicated.columns
    ]

    result = (
        duplicated[columns]
        .sort_values(
            [
                "duplicate_group",
            ]
        )
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Filas duplicadas: {len(result):,}"
    )

    if not result.empty:
        print(
            "Grupos de duplicados:",
            result["duplicate_group"].nunique(),
        )

    print(
        f"Archivo: {OUTPUT}"
    )


if __name__ == "__main__":
    main()