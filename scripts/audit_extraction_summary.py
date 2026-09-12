from pathlib import Path

import pandas as pd


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RUNS_DIR = ROOT / "data" / "runs"
REPORTS_DIR = ROOT / "outputs" / "reports"

OUTPUT_DIR = ROOT / "docs" / "evidence"
OUTPUT_FILE = OUTPUT_DIR / "extraction_summary.csv"


def main() -> None:
    # ========================================================
    # 1. Final extraction state
    # ========================================================

    article_files = sorted(
        RUNS_DIR.glob("*/extracted_text/articles_text.parquet")
    )

    if not article_files:
        raise FileNotFoundError(
            "No se encontraron archivos "
            "data/runs/*/extracted_text/articles_text.parquet"
        )

    articles_processed = 0
    articles_extracted_successfully = 0

    articles_with_fetch_error = 0
    articles_empty_without_fetch_error = 0
    articles_without_valid_extraction = 0

    for path in article_files:
        df = pd.read_parquet(path)

        required_columns = {
            "text",
            "fetch_error",
        }

        missing_columns = required_columns - set(df.columns)

        if missing_columns:
            raise ValueError(
                f"Faltan columnas en {path}: "
                f"{sorted(missing_columns)}"
            )

        # Text is considered valid only if it is non-empty
        text_empty = (
            df["text"]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
        )

        fetch_error = df["fetch_error"].notna()

        valid_extraction = (
            ~text_empty
            & ~fetch_error
        )

        # Final invalid extraction:
        # either a fetch error remains or no usable text exists
        invalid_extraction = (
            fetch_error
            | text_empty
        )

        articles_processed += len(df)

        articles_extracted_successfully += int(
            valid_extraction.sum()
        )

        articles_with_fetch_error += int(
            fetch_error.sum()
        )

        articles_empty_without_fetch_error += int(
            (
                text_empty
                & ~fetch_error
            ).sum()
        )

        articles_without_valid_extraction += int(
            invalid_extraction.sum()
        )

    # ========================================================
    # 2. Retry reports
    # ========================================================

    retry_files = sorted(
        REPORTS_DIR.glob(
            "*/articles_text_retry_summary.csv"
        )
    )

    articles_retried = 0
    articles_recovered_by_retry = 0

    for path in retry_files:
        df = pd.read_csv(path)

        required_columns = {
            "metric",
            "n",
        }

        missing_columns = required_columns - set(df.columns)

        if missing_columns:
            raise ValueError(
                f"Faltan columnas en {path}: "
                f"{sorted(missing_columns)}"
            )

        metrics = dict(
            zip(
                df["metric"],
                df["n"],
            )
        )

        articles_retried += int(
            metrics.get(
                "failed_before_retry",
                0,
            )
        )

        articles_recovered_by_retry += int(
            metrics.get(
                "recovered",
                0,
            )
        )

    # ========================================================
    # 3. Rates
    # ========================================================

    if articles_processed > 0:
        extraction_success_rate = (
            articles_extracted_successfully
            / articles_processed
            * 100
        )

        final_failure_rate = (
            articles_without_valid_extraction
            / articles_processed
            * 100
        )
    else:
        extraction_success_rate = 0.0
        final_failure_rate = 0.0

    if articles_retried > 0:
        retry_recovery_rate = (
            articles_recovered_by_retry
            / articles_retried
            * 100
        )
    else:
        retry_recovery_rate = 0.0

    # ========================================================
    # 4. Consistency checks
    # ========================================================

    if (
        articles_extracted_successfully
        + articles_without_valid_extraction
        != articles_processed
    ):
        raise ValueError(
            "Inconsistencia: artículos extraídos + "
            "artículos sin extracción válida != "
            "artículos procesados."
        )

    if (
        articles_with_fetch_error
        + articles_empty_without_fetch_error
        != articles_without_valid_extraction
    ):
        raise ValueError(
            "Inconsistencia: fetch_error + textos vacíos "
            "sin fetch_error != artículos sin extracción válida."
        )

    if (
        articles_recovered_by_retry
        > articles_retried
    ):
        raise ValueError(
            "Inconsistencia: hay más artículos recuperados "
            "que artículos reintentados."
        )

    # ========================================================
    # 5. Evidence summary
    #
    # Values are stored as strings so integer counts are not
    # written to CSV as 132.0, 428497.0, etc.
    # ========================================================

    summary = pd.DataFrame(
        [
            {
                "metric": "runs_analyzed",
                "value": str(len(article_files)),
                "description": (
                    "Number of final articles_text.parquet "
                    "files analyzed"
                ),
            },
            {
                "metric": "runs_with_retry_report",
                "value": str(len(retry_files)),
                "description": (
                    "Number of runs with a retry summary report"
                ),
            },
            {
                "metric": "articles_processed",
                "value": str(articles_processed),
                "description": (
                    "Total article records processed"
                ),
            },
            {
                "metric": "articles_extracted_successfully",
                "value": str(
                    articles_extracted_successfully
                ),
                "description": (
                    "Articles with non-empty text and "
                    "without fetch_error after retries"
                ),
            },
            {
                "metric": "articles_without_valid_extraction",
                "value": str(
                    articles_without_valid_extraction
                ),
                "description": (
                    "Articles without usable extracted text "
                    "after the extraction and retry process"
                ),
            },
            {
                "metric": "articles_with_fetch_error",
                "value": str(
                    articles_with_fetch_error
                ),
                "description": (
                    "Articles retaining fetch_error "
                    "after retries"
                ),
            },
            {
                "metric": "articles_empty_without_fetch_error",
                "value": str(
                    articles_empty_without_fetch_error
                ),
                "description": (
                    "Articles with empty text but "
                    "without fetch_error"
                ),
            },
            {
                "metric": "extraction_success_rate_pct",
                "value": f"{extraction_success_rate:.4f}",
                "description": (
                    "Percentage of processed articles "
                    "successfully extracted"
                ),
            },
            {
                "metric": "final_failure_rate_pct",
                "value": f"{final_failure_rate:.4f}",
                "description": (
                    "Percentage of processed articles "
                    "without usable extracted text"
                ),
            },
            {
                "metric": "articles_retried",
                "value": str(articles_retried),
                "description": (
                    "Articles submitted to the retry procedure"
                ),
            },
            {
                "metric": "articles_recovered_by_retry",
                "value": str(
                    articles_recovered_by_retry
                ),
                "description": (
                    "Articles successfully recovered "
                    "during retry"
                ),
            },
            {
                "metric": "retry_recovery_rate_pct",
                "value": f"{retry_recovery_rate:.4f}",
                "description": (
                    "Percentage of retried articles recovered"
                ),
            },
        ]
    )

    # ========================================================
    # 6. Save evidence
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    # ========================================================
    # 7. Console output
    # ========================================================

    print("\n=== EXTRACTION QUALITY AUDIT ===\n")

    print(
        f"Runs analyzed: "
        f"{len(article_files):,}"
    )

    print(
        f"Runs with retry report: "
        f"{len(retry_files):,}"
    )

    print(
        f"Articles processed: "
        f"{articles_processed:,}"
    )

    print(
        f"Successfully extracted: "
        f"{articles_extracted_successfully:,}"
    )

    print(
        f"Without valid extraction: "
        f"{articles_without_valid_extraction:,}"
    )

    print(
        f"  - With fetch_error: "
        f"{articles_with_fetch_error:,}"
    )

    print(
        f"  - Empty without fetch_error: "
        f"{articles_empty_without_fetch_error:,}"
    )

    print(
        f"Extraction success rate: "
        f"{extraction_success_rate:.2f}%"
    )

    print(
        f"Final failure rate: "
        f"{final_failure_rate:.2f}%"
    )

    print()
    print(
        f"Articles retried: "
        f"{articles_retried:,}"
    )

    print(
        f"Recovered by retry: "
        f"{articles_recovered_by_retry:,}"
    )

    print(
        f"Retry recovery rate: "
        f"{retry_recovery_rate:.2f}%"
    )

    print(
        "\nEvidence saved to:"
    )
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()