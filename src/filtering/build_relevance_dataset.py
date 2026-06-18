from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.filtering.article_classifier import (
    load_lexicon,
    prepare_lexicon,
    classify_article,
)
from src.utils.logging_config import setup_logger


ROOT = Path(__file__).resolve().parents[2]


def sort_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort candidates for easier manual inspection.
    """
    if df.empty:
        return df

    sort_cols = []
    ascending = []

    for col in ["country", "source", "retrieval_bucket"]:
        if col in df.columns:
            sort_cols.append(col)
            ascending.append(True)

    for col in ["relevance_score", "case_score", "topic_score", "text_length"]:
        if col in df.columns:
            sort_cols.append(col)
            ascending.append(False)

    if not sort_cols:
        return df

    return df.sort_values(sort_cols, ascending=ascending)


def save_dataset(df: pd.DataFrame, parquet_path: Path, csv_path: Path | None = None) -> None:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)

    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")


def main() -> None:
    logger = setup_logger("build_relevance_dataset", ROOT / "data" / "logs")

    articles_path = ROOT / "data" / "extracted_text" / "articles_text.parquet"
    lexicon_path = ROOT / "configs" / "lexicon_retrieval.yml"

    out_dir = ROOT / "data" / "processed"
    report_dir = ROOT / "outputs" / "reports"

    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    classified_path = out_dir / "articles_classified.parquet"
    all_candidates_path = out_dir / "retrieval_candidates.parquet"
    case_strong_path = out_dir / "case_strong_candidates.parquet"
    case_review_path = out_dir / "case_review_candidates.parquet"
    case_all_path = out_dir / "case_candidates.parquet"
    topic_path = out_dir / "topic_candidates.parquet"
    topic_only_path = out_dir / "topic_only_candidates.parquet"

    logger.info(f"Reading articles: {articles_path}")
    df = pd.read_parquet(articles_path)

    logger.info(f"Reading lexicon: {lexicon_path}")
    lexicon = load_lexicon(lexicon_path)
    prepared_lexicon = prepare_lexicon(lexicon)

    rows = []

    for row in tqdm(df.to_dict(orient="records"), total=len(df)):
        result = classify_article(
            row=row,
            lexicon=lexicon,
            prepared_lexicon=prepared_lexicon,
        )
        rows.append({**row, **result})

    classified = pd.DataFrame(rows)

    # Main classified dataset.
    save_dataset(
        classified,
        parquet_path=classified_path,
        csv_path=report_dir / "articles_classified.csv",
    )

    # Candidate subsets.
    all_candidates = classified[classified["is_retrieval_candidate"] == True].copy()

    case_strong = classified[
        classified["retrieval_bucket"] == "case_strong"
    ].copy()

    case_review = classified[
        classified["retrieval_bucket"].astype(str).str.startswith("case_review")
    ].copy()

    case_all = classified[
        classified["is_case_candidate"] == True
    ].copy()

    topic_all = classified[
        classified["is_topic_candidate"] == True
    ].copy()

    topic_only = classified[
        (classified["is_topic_candidate"] == True)
        & (classified["is_case_candidate"] == False)
    ].copy()

    all_candidates = sort_candidates(all_candidates)
    case_strong = sort_candidates(case_strong)
    case_review = sort_candidates(case_review)
    case_all = sort_candidates(case_all)
    topic_all = sort_candidates(topic_all)
    topic_only = sort_candidates(topic_only)

    save_dataset(
        all_candidates,
        parquet_path=all_candidates_path,
        csv_path=report_dir / "retrieval_candidates.csv",
    )

    save_dataset(
        case_strong,
        parquet_path=case_strong_path,
        csv_path=report_dir / "case_strong_candidates.csv",
    )

    save_dataset(
        case_review,
        parquet_path=case_review_path,
        csv_path=report_dir / "case_review_candidates.csv",
    )

    save_dataset(
        case_all,
        parquet_path=case_all_path,
        csv_path=report_dir / "case_candidates.csv",
    )

    save_dataset(
        topic_all,
        parquet_path=topic_path,
        csv_path=report_dir / "topic_candidates.csv",
    )

    save_dataset(
        topic_only,
        parquet_path=topic_only_path,
        csv_path=report_dir / "topic_only_candidates.csv",
    )

    # Summaries.
    retrieval_bucket_summary = (
        classified["retrieval_bucket"]
        .value_counts(dropna=False)
        .reset_index()
    )
    retrieval_bucket_summary.columns = ["retrieval_bucket", "n"]

    candidate_label_summary = (
        classified["candidate_label"]
        .value_counts(dropna=False)
        .reset_index()
    )
    candidate_label_summary.columns = ["candidate_label", "n"]

    candidate_type_summary = (
        classified["candidate_type"]
        .value_counts(dropna=False)
        .reset_index()
    )
    candidate_type_summary.columns = ["candidate_type", "n"]

    retrieval_bucket_summary.to_csv(
        report_dir / "summary_retrieval_bucket.csv",
        index=False,
        encoding="utf-8-sig",
    )

    candidate_label_summary.to_csv(
        report_dir / "summary_candidate_label.csv",
        index=False,
        encoding="utf-8-sig",
    )

    candidate_type_summary.to_csv(
        report_dir / "summary_candidate_type.csv",
        index=False,
        encoding="utf-8-sig",
    )

    logger.info(f"Total articles: {len(classified):,}")
    logger.info(f"Retrieval candidates: {len(all_candidates):,}")
    logger.info(f"Case candidates: {len(case_all):,}")
    logger.info(f"Strong case candidates: {len(case_strong):,}")
    logger.info(f"Review case candidates: {len(case_review):,}")
    logger.info(f"Topic candidates: {len(topic_all):,}")
    logger.info(f"Topic-only candidates: {len(topic_only):,}")

    logger.info(f"Saved classified dataset: {classified_path}")
    logger.info(f"Saved all candidates: {all_candidates_path}")
    logger.info(f"Saved strong cases: {case_strong_path}")
    logger.info(f"Saved review cases: {case_review_path}")
    logger.info(f"Saved topic candidates: {topic_path}")

    logger.info("\nRetrieval bucket summary:")
    logger.info("\n" + retrieval_bucket_summary.to_string(index=False))

    logger.info("\nCandidate label summary:")
    logger.info("\n" + candidate_label_summary.to_string(index=False))

    logger.info("\nCandidate type summary:")
    logger.info("\n" + candidate_type_summary.to_string(index=False))


if __name__ == "__main__":
    main()