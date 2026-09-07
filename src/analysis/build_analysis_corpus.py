from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from src.analysis.article_normalization import (
    canonicalize_url_key,
    classify_page_type,
    extract_date_from_url,
    parse_wayback_date,
    resolve_analysis_date,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "analysis.yml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construye el corpus analítico canónico de Fase 1 "
            "a partir de case_articles_main.parquet."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def series_or_empty(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series("", index=df.index, dtype="string")
    return df[column].fillna("").astype("string").str.strip()


def first_nonempty(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    result = pd.Series("", index=df.index, dtype="string")
    for column in columns:
        if column in df.columns:
            values = series_or_empty(df, column)
            result = result.mask(result.eq(""), values)
    return result


def normalize_whitespace(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).replace("\u00a0", " ")).strip()


def count_words(value: object) -> int:
    text = normalize_whitespace(value)
    if not text:
        return 0
    return len(re.findall(r"\b[\wáéíóúüñÁÉÍÓÚÜÑ]+\b", text, flags=re.UNICODE))


def ensure_phase0_article_key(df: pd.DataFrame) -> pd.Series:
    existing = series_or_empty(df, "article_key")
    url = first_nonempty(df, ["normalized_url", "candidate_url"])
    source = series_or_empty(df, "source")
    year = series_or_empty(df, "archive_year")
    title = first_nonempty(df, ["title_clean", "title", "title_raw"])
    fallback = "fallback|" + source + "|" + year + "|" + title
    generated = url.mask(url.eq(""), fallback)
    return existing.mask(existing.eq(""), generated)


def build_archive_year(df: pd.DataFrame, config: dict) -> pd.Series:
    column = config["dates"].get("archive_year_column", "archive_year")
    if column in df.columns:
        year = pd.to_numeric(df[column], errors="coerce").astype("Int64")
    else:
        year = pd.Series(pd.NA, index=df.index, dtype="Int64")

    snapshot_column = config["dates"].get(
        "snapshot_timestamp_column", "snapshot_timestamp"
    )
    if snapshot_column in df.columns:
        snapshot_year = (
            df[snapshot_column]
            .fillna("")
            .astype("string")
            .str.extract(r"^(\d{4})", expand=False)
        )
        snapshot_year = pd.to_numeric(snapshot_year, errors="coerce").astype("Int64")
        year = year.fillna(snapshot_year)
    return year


def first_publication_date(
    df: pd.DataFrame, candidate_columns: list[str]
) -> tuple[pd.Series, pd.Series]:
    dates = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    sources = pd.Series("", index=df.index, dtype="string")

    for column in candidate_columns:
        if column not in df.columns:
            continue
        try:
            parsed = pd.to_datetime(df[column], errors="coerce", utc=True)
            parsed = parsed.dt.tz_convert(None)
        except (TypeError, ValueError, AttributeError):
            parsed = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
        mask = dates.isna() & parsed.notna()
        dates.loc[mask] = parsed.loc[mask]
        sources.loc[mask] = column
    return dates, sources


def assign_period(year: object, periods: list[dict]) -> str:
    if year is None or pd.isna(year):
        return "unknown"
    year_int = int(year)
    for period in periods:
        if int(period["start_year"]) <= year_int <= int(period["end_year"]):
            return str(period["label"])
    return "outside_defined_periods"


def build_pre_dedup_dataframe(
    raw: pd.DataFrame,
    config: dict,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    df = raw.copy()
    df["article_key"] = ensure_phase0_article_key(df)
    df["article_url"] = first_nonempty(
        df, ["normalized_url", "candidate_url", "original_url", "url"]
    )
    df["canonical_url_key"] = df["article_url"].map(canonicalize_url_key).astype("string")

    df["analysis_identity"] = df["canonical_url_key"]
    missing = df["analysis_identity"].eq("")
    df.loc[missing, "analysis_identity"] = (
        "phase0|" + df.loc[missing, "article_key"].astype("string")
    )

    title = first_nonempty(df, list(config["text"]["title_columns"]))
    body = first_nonempty(df, list(config["text"]["body_columns"]))
    df["analysis_title"] = title.map(normalize_whitespace)
    df["analysis_body"] = body.map(normalize_whitespace)
    # Importante: anchor_text no entra en el corpus lingüístico.
    df["analysis_text"] = df["analysis_body"]

    df["title_word_count"] = df["analysis_title"].map(count_words)
    df["word_count"] = df["analysis_text"].map(count_words)
    df["char_count"] = df["analysis_text"].str.len()
    df["has_title"] = df["title_word_count"].gt(0)
    df["has_analysis_text"] = df["word_count"].gt(0)

    source_series = series_or_empty(df, "source")
    page_results = [
        classify_page_type(source, url)
        for source, url in zip(source_series, df["article_url"])
    ]
    df["page_type"] = [page_type for page_type, _ in page_results]
    df["page_type_reason"] = [reason for _, reason in page_results]
    excluded_types = set(
        config.get("page_filter", {}).get(
            "excluded_page_types", ["topic_page", "author_page", "section_page"]
        )
    )
    df["include_in_article_corpus"] = ~df["page_type"].isin(excluded_types)

    publication_date, publication_field = first_publication_date(
        df, list(config["dates"]["publication_date_columns"])
    )
    df["publication_date"] = publication_date
    df["publication_date_field"] = publication_field
    df["url_date"] = df["article_url"].map(extract_date_from_url)

    snapshot_column = config["dates"].get(
        "snapshot_timestamp_column", "snapshot_timestamp"
    )
    df["archive_date"] = (
        df[snapshot_column].map(parse_wayback_date)
        if snapshot_column in df.columns
        else pd.NaT
    )
    df["archive_year"] = build_archive_year(df, config)

    resolved = [
        resolve_analysis_date(pub, url_date, archive_year)
        for pub, url_date, archive_year in zip(
            df["publication_date"], df["url_date"], df["archive_year"]
        )
    ]
    df["analysis_date"] = [row[0] for row in resolved]
    df["analysis_year"] = pd.Series(
        [row[1] for row in resolved], index=df.index, dtype="Int64"
    )
    df["analysis_year_source"] = [row[2] for row in resolved]
    df["analysis_period"] = df["analysis_year"].map(
        lambda year: assign_period(year, config.get("periods", []))
    )

    return df[
        df["analysis_year"].between(start_year, end_year, inclusive="both")
    ].copy()


def build_page_type_audit(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(
            ["country", "source", "page_type", "page_type_reason"], dropna=False
        )
        .size()
        .reset_index(name="n_pages")
        .sort_values(["country", "source", "page_type"])
    )


def build_temporal_resolution_audit(df: pd.DataFrame) -> pd.DataFrame:
    table = (
        df.groupby(["country", "source", "analysis_year_source"], dropna=False)
        .size()
        .reset_index(name="n_articles")
    )
    table["source_total"] = table.groupby(
        ["country", "source"], dropna=False
    )["n_articles"].transform("sum")
    table["percentage"] = 100 * table["n_articles"] / table["source_total"]
    return table.sort_values(["country", "source", "analysis_year_source"])


def build_archive_vs_analysis_audit(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(
            [
                "country", "source", "archive_year", "analysis_year",
                "analysis_year_source",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="n_articles")
        .sort_values(["country", "source", "archive_year", "analysis_year"])
    )


def build_duplicate_audit(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for identity, group in df.groupby("analysis_identity", dropna=False):
        if len(group) <= 1:
            continue
        years = sorted({int(value) for value in group["archive_year"].dropna()})
        urls = sorted({str(value) for value in group["article_url"] if str(value).strip()})
        rows.append(
            {
                "analysis_identity": identity,
                "n_rows": len(group),
                "n_archive_years": len(years),
                "archive_years": ";".join(map(str, years)),
                "archive_first_year": min(years) if years else pd.NA,
                "archive_last_year": max(years) if years else pd.NA,
                "urls_seen": " | ".join(urls),
                "max_word_count": int(group["word_count"].max()),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "analysis_identity", "n_rows", "n_archive_years",
                "archive_years", "archive_first_year", "archive_last_year",
                "urls_seen", "max_word_count",
            ]
        )
    return pd.DataFrame(rows).sort_values(
        ["n_rows", "analysis_identity"], ascending=[False, True]
    ).reset_index(drop=True)


def add_archive_history(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for identity, group in df.groupby("analysis_identity", dropna=False):
        years = sorted({int(value) for value in group["archive_year"].dropna()})
        rows.append(
            {
                "analysis_identity": identity,
                "archive_first_year": min(years) if years else pd.NA,
                "archive_last_year": max(years) if years else pd.NA,
                "archive_year_count": len(years),
                "archive_years_seen": ";".join(map(str, years)),
                "snapshot_row_count": len(group),
            }
        )
    return df.merge(pd.DataFrame(rows), on="analysis_identity", how="left")


def deduplicate_articles(df: pd.DataFrame) -> pd.DataFrame:
    with_history = add_archive_history(df)
    return (
        with_history.sort_values(
            [
                "analysis_identity", "has_analysis_text", "word_count",
                "has_title", "char_count",
            ],
            ascending=[True, False, False, False, False],
        )
        .drop_duplicates(subset=["analysis_identity"], keep="first")
        .reset_index(drop=True)
    )


def build_title_availability(df: pd.DataFrame) -> pd.DataFrame:
    table = (
        df.groupby(["analysis_year", "country", "source"], dropna=False)
        .agg(
            n_articles=("analysis_identity", "nunique"),
            titles_available=("has_title", "sum"),
        )
        .reset_index()
    )
    table["title_availability_pct"] = (
        100 * table["titles_available"] / table["n_articles"]
    )
    return table.sort_values(["analysis_year", "country", "source"])


def build_text_quality(df: pd.DataFrame) -> pd.DataFrame:
    table = (
        df.groupby(["country", "source"], dropna=False)
        .agg(
            n_articles=("analysis_identity", "nunique"),
            articles_with_text=("has_analysis_text", "sum"),
            median_words=("word_count", "median"),
            mean_words=("word_count", "mean"),
            p05_words=("word_count", lambda s: s.quantile(0.05)),
            p95_words=("word_count", lambda s: s.quantile(0.95)),
        )
        .reset_index()
    )
    table["text_availability_pct"] = (
        100 * table["articles_with_text"] / table["n_articles"]
    )
    return table.sort_values(["country", "source"])


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    config = load_config(resolve_path(args.config))
    start_year = int(
        args.start_year if args.start_year is not None else config["study"]["start_year"]
    )
    end_year = int(
        args.end_year
        if args.end_year is not None
        else config["study"]["current_data_end_year"]
    )
    snapshot = f"{start_year}_{end_year}"

    input_path = resolve_path(config["corpus"]["main_path"])
    if not input_path.exists():
        raise FileNotFoundError(f"No se encontró el corpus principal: {input_path}")

    output_dir = resolve_path(config["output"]["root_dir"]) / snapshot
    tables_dir = output_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_parquet(input_path)
    pre = build_pre_dedup_dataframe(raw, config, start_year, end_year)


    page_type_audit = build_page_type_audit(pre)

    excluded_non_articles = pre[
        ~pre["include_in_article_corpus"]
    ].copy()

    included = pre[
        pre["include_in_article_corpus"]
    ].copy()

    duplicate_audit = build_duplicate_audit(included)

    final = deduplicate_articles(included)

    # Auditorías temporales del corpus REALMENTE analizado.
    temporal_audit = build_temporal_resolution_audit(final)
    archive_vs_analysis = build_archive_vs_analysis_audit(final)

    final["analysis_snapshot"] = snapshot

    final.to_parquet(output_dir / "articles.parquet", index=False)

    save_csv(page_type_audit, tables_dir / "page_type_audit.csv")
    audit_columns = [
        "article_key", "analysis_identity", "country", "source", "archive_year",
        "analysis_year", "analysis_year_source", "normalized_url", "candidate_url",
        "analysis_title", "word_count", "page_type", "page_type_reason",
    ]
    audit_columns = [c for c in audit_columns if c in excluded_non_articles.columns]
    save_csv(excluded_non_articles[audit_columns], tables_dir / "non_article_pages.csv")
    save_csv(temporal_audit, tables_dir / "temporal_resolution_by_source.csv")
    save_csv(archive_vs_analysis, tables_dir / "archive_vs_analysis_year.csv")
    save_csv(duplicate_audit, tables_dir / "duplicate_resolution_audit.csv")
    save_csv(
        build_title_availability(final),
        tables_dir / "title_availability_by_source_year.csv",
    )
    save_csv(build_text_quality(final), tables_dir / "text_quality_by_source.csv")

    summary = pd.DataFrame(
        [{
            "input_rows": len(raw),
            "rows_in_requested_analysis_range": len(pre),
            "non_article_rows_excluded": len(excluded_non_articles),
            "rows_before_deduplication": len(included),
            "duplicate_rows_removed": len(included) - len(final),
            "final_articles": len(final),
            "articles_with_text": int(final["has_analysis_text"].sum()),
            "articles_with_title": int(final["has_title"].sum()),
            "min_analysis_year": int(final["analysis_year"].min()) if not final.empty else pd.NA,
            "max_analysis_year": int(final["analysis_year"].max()) if not final.empty else pd.NA,
        }]
    )
    save_csv(summary, tables_dir / "analysis_corpus_summary.csv")

    manifest = {
        "snapshot": snapshot,
        "start_year": start_year,
        "end_year": end_year,
        "target_end_year": int(config["study"]["target_end_year"]),
        "input": str(input_path),
        "output": str(output_dir / "articles.parquet"),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_rows": int(len(raw)),
        "final_articles": int(len(final)),
        "non_article_rows_excluded": int(len(excluded_non_articles)),
        "duplicate_rows_removed": int(len(included) - len(final)),
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    print("\nCorpus analítico canónico creado.")
    print(summary.to_string(index=False))
    print(f"\nSalida: {output_dir}")


if __name__ == "__main__":
    main()
