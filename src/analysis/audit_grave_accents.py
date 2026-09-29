from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT
    / "outputs"
    / "analysis"
    / "2015_2025"
    / "articles.parquet"
)

OUTPUT = (
    ROOT
    / "outputs"
    / "analysis"
    / "2015_2025"
    / "quality"
    / "grave_accent_contexts.csv"
)

GRAVE_RE = re.compile(
    r"\b[\wÀ-ÿ]*[àèìòùÀÈÌÒÙ][\wÀ-ÿ]*\b"
)


def main() -> None:

    df = pd.read_parquet(INPUT)

    if "analysis_eligible" not in df.columns:
        raise ValueError(
            "El corpus canónico no contiene "
            "'analysis_eligible'."
        )

    df = df.loc[
        df["analysis_eligible"]
    ].copy()

    print(
        f"Documentos elegibles auditados: {len(df):,}"
    )

    records = []

    for idx, row in df.iterrows():

        text = str(
            row.get(
                "analysis_text",
                "",
            )
            or ""
        )

        for match in GRAVE_RE.finditer(text):

            token = match.group(0)

            start = max(
                0,
                match.start() - 100,
            )

            end = min(
                len(text),
                match.end() + 100,
            )

            context = (
                text[start:end]
                .replace("\n", " ")
            )

            records.append(
                {
                    "row_index": idx,
                    "country": row.get(
                        "country",
                        "",
                    ),
                    "source": row.get(
                        "source",
                        "",
                    ),
                    "title": row.get(
                        "title",
                        "",
                    ),
                    "candidate_url": row.get(
                        "candidate_url",
                        "",
                    ),
                    "token": token,
                    "context": context,
                }
            )

    result = pd.DataFrame(records)

    if result.empty:
        print(
            "No se encontraron palabras "
            "con acento grave."
        )
        return

    result["frequency"] = (
        result
        .groupby("token")["token"]
        .transform("size")
    )

    result = result.sort_values(
        [
            "frequency",
            "token",
            "source",
        ],
        ascending=[
            False,
            True,
            True,
        ],
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
        f"Ocurrencias: {len(result):,}"
    )

    print(
        f"Tokens distintos: "
        f"{result['token'].nunique():,}"
    )

    print(
        f"Artículos afectados: "
        f"{result['row_index'].nunique():,}"
    )

    print(
        f"\nArchivo: {OUTPUT}"
    )

    print("\nFrecuencias:\n")

    print(
        result[
            ["token", "frequency"]
        ]
        .drop_duplicates()
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()