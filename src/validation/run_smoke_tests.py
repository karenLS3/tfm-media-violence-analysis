from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class SmokeWindow:
    run_id: str
    from_date: str
    to_date: str


def build_windows(suite: str) -> list[SmokeWindow]:
    if suite == "minimal":
        return [
            SmokeWindow("2015_01_test", "20150101", "20150103"),
        ]

    if suite == "distributed":
        return [
            SmokeWindow("2015_01_test", "20150101", "20150103"),
            SmokeWindow("2017_03_test", "20170307", "20170309"),
            SmokeWindow("2019_11_test", "20191124", "20191126"),
            SmokeWindow("2022_06_test", "20220615", "20220617"),
            SmokeWindow("2025_11_test", "20251124", "20251126"),
        ]

    if suite == "stratified":
        windows: list[SmokeWindow] = []

        for year in range(2015, 2026):
            windows.extend(
                [
                    SmokeWindow(f"{year}_01_test", f"{year}0101", f"{year}0103"),
                    SmokeWindow(f"{year}_03_test", f"{year}0307", f"{year}0309"),
                    SmokeWindow(f"{year}_06_test", f"{year}0615", f"{year}0617"),
                    SmokeWindow(f"{year}_11_test", f"{year}1124", f"{year}1126"),
                ]
            )

        return windows

    if suite == "annual":
        return [
            SmokeWindow("2015_01_01_test", "20150101", "20150101"),
            SmokeWindow("2016_03_08_test", "20160308", "20160308"),
            SmokeWindow("2017_06_15_test", "20170615", "20170615"),
            SmokeWindow("2018_11_25_test", "20181125", "20181125"),
            SmokeWindow("2019_01_01_test", "20190101", "20190101"),
            SmokeWindow("2020_03_08_test", "20200308", "20200308"),
            SmokeWindow("2021_06_15_test", "20210615", "20210615"),
            SmokeWindow("2022_11_25_test", "20221125", "20221125"),
            SmokeWindow("2023_01_01_test", "20230101", "20230101"),
            SmokeWindow("2024_03_08_test", "20240308", "20240308"),
            SmokeWindow("2025_11_25_test", "20251125", "20251125"),
        ]

    raise ValueError(
        f"Suite desconocida: {suite}. Usa: minimal, distributed, annual o stratified."
    )


def run_command(command: list[str], continue_on_error: bool) -> bool:
    print("\n" + "=" * 100)
    print("RUNNING:")
    print(" ".join(command))
    print("=" * 100)

    result = subprocess.run(command, cwd=ROOT)

    if result.returncode != 0:
        print(f"\nERROR: comando falló con código {result.returncode}")

        if not continue_on_error:
            raise SystemExit(result.returncode)

        return False

    return True


def pipeline_commands(
    window: SmokeWindow,
    config: str,
    sleep_seconds: float,
    skip_retry: bool,
) -> list[list[str]]:
    commands = [
        [
            sys.executable,
            "-m",
            "src.acquisition.build_cdx_index",
            "--config",
            config,
            "--from-date",
            window.from_date,
            "--to-date",
            window.to_date,
            "--run-id",
            window.run_id,
        ],
        [
            sys.executable,
            "-m",
            "src.acquisition.build_homepage_candidates",
            "--run-id",
            window.run_id,
        ],
        [
            sys.executable,
            "-m",
            "src.filtering.build_clean_candidates",
            "--run-id",
            window.run_id,
        ],
        [
            sys.executable,
            "-m",
            "src.acquisition.build_article_texts",
            "--run-id",
            window.run_id,
            "--sleep-seconds",
            str(sleep_seconds),
        ],
    ]

    if not skip_retry:
        commands.append(
            [
                sys.executable,
                "-m",
                "src.acquisition.retry_failed_article_texts",
                "--run-id",
                window.run_id,
                "--sleep-seconds",
                str(max(sleep_seconds, 3.0)),
            ]
        )

    commands.append(
        [
            sys.executable,
            "-m",
            "src.filtering.build_relevance_dataset",
            "--run-id",
            window.run_id,
        ]
    )

    return commands


def count_rows(path: Path) -> int:
    if not path.exists():
        return 0

    try:
        return len(pd.read_parquet(path))
    except Exception:
        return 0


def summarize_run(window: SmokeWindow) -> tuple[dict, pd.DataFrame]:
    run_base = ROOT / "data" / "runs" / window.run_id

    cdx_path = run_base / "raw_cdx" / "cdx_snapshots.parquet"
    homepage_path = run_base / "candidates" / "homepage_candidates.parquet"
    clean_path = run_base / "candidates" / "clean_candidates.parquet"
    articles_path = run_base / "extracted_text" / "articles_text.parquet"
    classified_path = run_base / "processed" / "articles_classified.parquet"

    summary = {
        "run_id": window.run_id,
        "from_date": window.from_date,
        "to_date": window.to_date,
        "cdx_snapshots": count_rows(cdx_path),
        "homepage_candidates": count_rows(homepage_path),
        "clean_candidates": count_rows(clean_path),
        "articles_text": count_rows(articles_path),
        "articles_classified": count_rows(classified_path),
        "fetch_or_parse_errors": None,
        "articles_with_text": None,
        "case_strong": None,
        "case_review_possible_violence": None,
        "topic_gender_violence": None,
        "not_candidate": None,
        "has_text_clean": None,
        "has_title_clean": None,
        "status": "missing_classified" if not classified_path.exists() else "ok",
    }

    by_source_rows = []

    if articles_path.exists():
        try:
            articles = pd.read_parquet(articles_path)

            if "fetch_error" in articles.columns:
                summary["fetch_or_parse_errors"] = int(
                    articles["fetch_error"].notna().sum()
                )

            if "text" in articles.columns:
                summary["articles_with_text"] = int(
                    articles["text"].fillna("").astype(str).str.len().gt(0).sum()
                )

        except Exception as exc:
            summary["status"] = f"articles_summary_error: {type(exc).__name__}"

    if classified_path.exists():
        try:
            classified = pd.read_parquet(classified_path)

            summary["has_text_clean"] = "text_clean" in classified.columns
            summary["has_title_clean"] = "title_clean" in classified.columns

            if "retrieval_bucket" in classified.columns:
                counts = classified["retrieval_bucket"].value_counts(dropna=False)

                summary["case_strong"] = int(counts.get("case_strong", 0))
                summary["case_review_possible_violence"] = int(
                    counts.get("case_review_possible_violence", 0)
                )
                summary["topic_gender_violence"] = int(
                    counts.get("topic_gender_violence", 0)
                )
                summary["not_candidate"] = int(counts.get("not_candidate", 0))

            if {"source", "country", "retrieval_bucket"}.issubset(classified.columns):
                table = (
                    classified.groupby(["source", "country", "retrieval_bucket"])
                    .size()
                    .reset_index(name="n")
                )

                table["run_id"] = window.run_id
                table["from_date"] = window.from_date
                table["to_date"] = window.to_date

                by_source_rows.append(table)

        except Exception as exc:
            summary["status"] = f"classified_summary_error: {type(exc).__name__}"

    if by_source_rows:
        by_source = pd.concat(by_source_rows, ignore_index=True)
    else:
        by_source = pd.DataFrame()

    return summary, by_source


def write_summary(
    windows: list[SmokeWindow],
    suite: str,
) -> None:
    out_dir = ROOT / "outputs" / "reports" / "smoke_tests"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    by_source_frames = []

    for window in windows:
        summary, by_source = summarize_run(window)
        summary_rows.append(summary)

        if not by_source.empty:
            by_source_frames.append(by_source)

    summary_df = pd.DataFrame(summary_rows)

    summary_path = out_dir / f"smoke_test_summary_{suite}.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print(f"SUMMARY WRITTEN: {summary_path}")
    print("=" * 100)
    print(summary_df.to_string(index=False))

    if by_source_frames:
        by_source_df = pd.concat(by_source_frames, ignore_index=True)
        by_source_path = out_dir / f"smoke_test_by_source_{suite}.csv"
        by_source_df.to_csv(by_source_path, index=False, encoding="utf-8-sig")

        print("\n" + "=" * 100)
        print(f"BY SOURCE WRITTEN: {by_source_path}")
        print("=" * 100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run smoke tests for the Wayback news mining pipeline."
    )

    parser.add_argument(
        "--suite",
        choices=["minimal", "distributed", "annual", "stratified"],
        default="minimal",
        help=(
            "minimal: una corrida. "
            "distributed: 5 ventanas entre 2015 y 2025. "
            "annual: 1 día por año entre 2015 y 2025. "
            "stratified: 44 ventanas, 4 por año entre 2015 y 2025."
        ),
    )

    parser.add_argument(
        "--config",
        default="configs/sources_argentina_mexico.yml",
        help="Config YAML de fuentes.",
    )

    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=1.0,
        help="Pausa entre descargas de artículos. Por defecto: 1 segundo.",
    )

    parser.add_argument(
        "--skip-retry",
        action="store_true",
        help="No ejecutar retry_failed_article_texts.",
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continúa con la siguiente corrida si una falla.",
    )

    parser.add_argument(
        "--only-summary",
        action="store_true",
        help="No ejecuta el pipeline; solo reconstruye el resumen desde data/runs/.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    windows = build_windows(args.suite)

    if args.only_summary:
        write_summary(windows, suite=args.suite)
        return

    print("\nSmoke test suite:", args.suite)
    print("Config:", args.config)
    print("Windows:")
    for window in windows:
        print(f"  - {window.run_id}: {window.from_date} -> {window.to_date}")

    for window in windows:
        print("\n" + "#" * 100)
        print(f"START RUN: {window.run_id} | {window.from_date} - {window.to_date}")
        print("#" * 100)

        commands = pipeline_commands(
            window=window,
            config=args.config,
            sleep_seconds=args.sleep_seconds,
            skip_retry=args.skip_retry,
        )

        run_ok = True

        for command in commands:
            ok = run_command(command, continue_on_error=args.continue_on_error)

            if not ok:
                run_ok = False
                print(f"Skipping remaining steps for run: {window.run_id}")
                break

        if run_ok:
            print(f"\nCOMPLETED RUN: {window.run_id}")
        else:
            print(f"\nFAILED RUN: {window.run_id}")

    write_summary(windows, suite=args.suite)


if __name__ == "__main__":
    main()
