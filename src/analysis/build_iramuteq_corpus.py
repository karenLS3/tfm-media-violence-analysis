from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd


def normalize_label(value: object) -> str:
    """Convert metadata values to safe IRaMuTeQ variable modalities."""
    if pd.isna(value):
        return "unknown"

    value = str(value).strip().lower()

    value = unicodedata.normalize("NFKD", value)
    value = "".join(
        char
        for char in value
        if not unicodedata.combining(char)
    )

    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = value.strip("_")

    return value or "unknown"


def clean_iramuteq_text(value: object) -> str:
    """
    Minimal preparation for IRaMuTeQ.

    The natural article text is preserved. We only remove characters that can
    interfere with IRaMuTeQ metadata syntax and normalize whitespace.
    """
    if pd.isna(value):
        return ""

    text = str(value)

    # Asterisks are structural characters in IRaMuTeQ.
    text = text.replace("*", " ")

    # Preserve accents and punctuation.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def build_country_corpus(
    df: pd.DataFrame,
    *,
    country: str,
    country_col: str,
    year_col: str,
    source_col: str,
    text_col: str,
) -> tuple[str, pd.DataFrame]:

    country_norm = normalize_label(country)

    working = df.copy()

    working["_country_norm"] = (
        working[country_col]
        .map(normalize_label)
    )

    working = working[
        working["_country_norm"] == country_norm
    ].copy()

    working[year_col] = pd.to_numeric(
        working[year_col],
        errors="coerce",
    )

    working = working[
        working[year_col].between(2015, 2025)
    ].copy()

    working["_iramuteq_text"] = (
        working[text_col]
        .map(clean_iramuteq_text)
    )

    working = working[
        working["_iramuteq_text"].str.len() > 0
    ].copy()

    working = working.sort_values(
        [year_col, source_col],
        kind="stable",
    ).reset_index(drop=True)

    blocks: list[str] = []

    audit_rows: list[dict] = []

    for idx, row in working.iterrows():
        article_id = idx + 1
        year = int(row[year_col])
        source = normalize_label(row[source_col])

        header = (
            f"**** "
            f"*id_{article_id:06d} "
            f"*year_{year} "
            f"*source_{source}"
        )

        blocks.append(
            f"{header}\n{row['_iramuteq_text']}"
        )

        audit_rows.append(
            {
                "iramuteq_id": f"id_{article_id:06d}",
                "year": year,
                "source": source,
            }
        )

    corpus = "\n\n".join(blocks)

    if corpus:
        corpus += "\n"

    audit = pd.DataFrame(audit_rows)

    return corpus, audit


def export_country(
    df: pd.DataFrame,
    *,
    country: str,
    output_dir: Path,
    country_col: str,
    year_col: str,
    source_col: str,
    text_col: str,
) -> None:

    country_slug = normalize_label(country)

    corpus, audit = build_country_corpus(
        df,
        country=country,
        country_col=country_col,
        year_col=year_col,
        source_col=source_col,
        text_col=text_col,
    )

    country_dir = output_dir / country_slug
    country_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = (
        country_dir
        / f"corpus_{country_slug}_2015_2025.txt"
    )

    audit_path = (
        country_dir
        / "corpus_audit.csv"
    )

    corpus_path.write_text(
        corpus,
        encoding="utf-8",
    )

    audit.to_csv(
        audit_path,
        index=False,
        encoding="utf-8",
    )

    print()
    print(f"{country.upper()}")
    print("-" * len(country))
    print(f"Textos exportados: {len(audit)}")

    if not audit.empty:
        print(
            audit.groupby("year")
            .size()
            .rename("n_articles")
            .to_string()
        )

    print(f"Corpus: {corpus_path}")
    print(f"Auditoría: {audit_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build independent IRaMuTeQ corpora "
            "for Argentina and Mexico."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Canonical analysis corpus (.parquet)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/analysis/2015_2025/iramuteq"
        ),
    )

    parser.add_argument(
        "--country-col",
        default="country",
    )

    parser.add_argument(
        "--year-col",
        default="archive_year",
    )

    parser.add_argument(
        "--source-col",
        default="source",
    )

    parser.add_argument(
        "--text-col",
        default="analysis_text",
    )

    return parser.parse_args()

def export_country_by_year(
    df: pd.DataFrame,
    *,
    country: str,
    output_dir: Path,
    country_col: str,
    year_col: str,
    source_col: str,
    text_col: str,
) -> pd.DataFrame:
    """
    Export one independent IRaMuTeQ corpus per year
    for a given country.
    """

    country_slug = normalize_label(country)

    working = df.copy()

    working["_country_norm"] = (
        working[country_col]
        .map(normalize_label)
    )

    working = working[
        working["_country_norm"] == country_slug
    ].copy()

    working[year_col] = pd.to_numeric(
        working[year_col],
        errors="coerce",
    )

    working = working[
        working[year_col].between(2015, 2025)
    ].copy()

    working["_iramuteq_text"] = (
        working[text_col]
        .map(clean_iramuteq_text)
    )

    working = working[
        working["_iramuteq_text"].str.len() > 0
    ].copy()

    country_dir = (
        output_dir
        / country_slug
        / "by_year"
    )

    country_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_rows: list[dict] = []

    for year in range(2015, 2026):

        subset = working[
            working[year_col] == year
        ].copy()

        subset = subset.sort_values(
            source_col,
            kind="stable",
        ).reset_index(drop=True)

        blocks: list[str] = []

        for idx, row in subset.iterrows():

            article_id = idx + 1

            source = normalize_label(
                row[source_col]
            )

            header = (
                "**** "
                f"*id_{article_id:06d} "
                f"*year_{year} "
                f"*source_{source}"
            )

            blocks.append(
                f"{header}\n"
                f"{row['_iramuteq_text']}"
            )

        corpus = "\n\n".join(blocks)

        if corpus:
            corpus += "\n"

        output_path = (
            country_dir
            / f"corpus_{country_slug}_{year}.txt"
        )

        output_path.write_text(
            corpus,
            encoding="utf-8",
        )

        manifest_rows.append(
            {
                "country": country_slug,
                "year": year,
                "n_articles": len(subset),
                "corpus_path": str(output_path),
            }
        )

    manifest = pd.DataFrame(
        manifest_rows
    )

    manifest_path = (
        country_dir
        / "yearly_manifest.csv"
    )

    manifest.to_csv(
        manifest_path,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        f"{country.upper()} — CORPUS POR AÑO"
    )
    print("-" * 50)

    print(
        manifest[
            ["year", "n_articles"]
        ].to_string(index=False)
    )

    print(
        f"Total: {manifest['n_articles'].sum()}"
    )

    print(
        f"Manifest: {manifest_path}"
    )

    return manifest


def main() -> None:
    args = parse_args()

    df = pd.read_parquet(args.input)

    required = {
        args.country_col,
        args.year_col,
        args.source_col,
        args.text_col,
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    export_country(
        df,
        country="argentina",
        output_dir=args.output_dir,
        country_col=args.country_col,
        year_col=args.year_col,
        source_col=args.source_col,
        text_col=args.text_col,
    )

    export_country(
        df,
        country="mexico",
        output_dir=args.output_dir,
        country_col=args.country_col,
        year_col=args.year_col,
        source_col=args.source_col,
        text_col=args.text_col,
    )

    export_country_by_year(
        df,
        country="argentina",
        output_dir=args.output_dir,
        country_col=args.country_col,
        year_col=args.year_col,
        source_col=args.source_col,
        text_col=args.text_col,
    )

    export_country_by_year(
        df,
        country="mexico",
        output_dir=args.output_dir,
        country_col=args.country_col,
        year_col=args.year_col,
        source_col=args.source_col,
        text_col=args.text_col,
    )

    print("OK: todos los documentos fueron exportados.")


if __name__ == "__main__":
    main()