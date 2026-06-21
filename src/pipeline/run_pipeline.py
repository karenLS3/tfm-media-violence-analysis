from __future__ import annotations

import argparse
import calendar
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PipelineWindow:
    run_id: str
    from_date: str
    to_date: str


def resolve_config_path(config: str) -> Path:
    raw = Path(config)

    candidates = [
        raw,
        ROOT / raw,
    ]

    if len(raw.parts) == 1:
        candidates.append(ROOT / "configs" / raw)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    tried = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        f"No se encontró el config: {config}\n"
        f"Rutas probadas:\n{tried}"
    )


def load_expected_sources(config: str) -> pd.DataFrame:
    config_path = resolve_config_path(config)

    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    rows = []

    for source in data.get("sources", []):
        rows.append(
            {
                "source": source.get("name", ""),
                "country": source.get("country", ""),
                "domain": source.get("domain", ""),
            }
        )

    return pd.DataFrame(rows)


def build_windows(
    mode: str,
    start_year: int,
    end_year: int,
    start_month: int | None = None,
    end_month: int | None = None,
) -> list[PipelineWindow]:
    windows: list[PipelineWindow] = []

    if mode == "yearly":
        for year in range(start_year, end_year + 1):
            windows.append(
                PipelineWindow(
                    run_id=f"{year}",
                    from_date=f"{year}0101",
                    to_date=f"{year}1231",
                )
            )

        return windows

    if mode == "monthly":
        for year in range(start_year, end_year + 1):
            first_month = start_month if year == start_year and start_month else 1
            last_month = end_month if year == end_year and end_month else 12

            for month in range(first_month, last_month + 1):
                last_day = calendar.monthrange(year, month)[1]

                windows.append(
                    PipelineWindow(
                        run_id=f"{year}_{month:02d}",
                        from_date=f"{year}{month:02d}01",
                        to_date=f"{year}{month:02d}{last_day:02d}",
                    )
                )

        return windows

    raise ValueError(f"Modo inválido: {mode}. Usa monthly o yearly.")


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


def stage_outputs(run_id: str) -> dict[str, Path]:
    run_base = ROOT / "data" / "runs" / run_id
    report_dir = ROOT / "outputs" / "reports" / run_id

    return {
        "cdx": run_base / "raw_cdx" / "cdx_snapshots.parquet",
        "homepage": run_base / "candidates" / "homepage_candidates.parquet",
        "clean": run_base / "candidates" / "clean_candidates.parquet",
        "articles": run_base / "extracted_text" / "articles_text.parquet",
        "retry": report_dir / "articles_text_retry_summary.csv",
        "classify": run_base / "processed" / "articles_classified.parquet",
    }


def should_skip_stage(stage: str, run_id: str, force: bool) -> bool:
    if force:
        return False

    output = stage_outputs(run_id)[stage]
    return output.exists()


def pipeline_commands(
    window: PipelineWindow,
    config: str,
    sleep_seconds: float,
    skip_retry: bool,
) -> list[tuple[str, list[str]]]:
    commands: list[tuple[str, list[str]]] = [
        (
            "cdx",
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
        ),
        (
            "homepage",
            [
                sys.executable,
                "-m",
                "src.acquisition.build_homepage_candidates",
                "--run-id",
                window.run_id,
            ],
        ),
        (
            "clean",
            [
                sys.executable,
                "-m",
                "src.filtering.build_clean_candidates",
                "--run-id",
                window.run_id,
            ],
        ),
        (
            "articles",
            [
                sys.executable,
                "-m",
                "src.acquisition.build_article_texts",
                "--run-id",
                window.run_id,
                "--sleep-seconds",
                str(sleep_seconds),
            ],
        ),
    ]

    if not skip_retry:
        commands.append(
            (
                "retry",
                [
                    sys.executable,
                    "-m",
                    "src.acquisition.retry_failed_article_texts",
                    "--run-id",
                    window.run_id,
                    "--sleep-seconds",
                    str(max(sleep_seconds, 3.0)),
                ],
            )
        )

    commands.append(
        (
            "classify",
            [
                sys.executable,
                "-m",
                "src.filtering.build_relevance_dataset",
                "--run-id",
                window.run_id,
            ],
        )
    )

    return commands


def read_parquet_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def count_rows(path: Path) -> int:
    df = read_parquet_safe(path)
    return len(df)


def source_counts(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    if df.empty or "source" not in df.columns:
        return pd.DataFrame(columns=["source", column_name])

    return (
        df.groupby("source")
        .size()
        .reset_index(name=column_name)
    )


def summarize_window(
    window: PipelineWindow,
    expected_sources: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    run_base = ROOT / "data" / "runs" / window.run_id
    report_dir = ROOT / "outputs" / "reports" / window.run_id

    cdx_path = run_base / "raw_cdx" / "cdx_snapshots.parquet"
    homepage_path = run_base / "candidates" / "homepage_candidates.parquet"
    clean_path = run_base / "candidates" / "clean_candidates.parquet"
    articles_path = run_base / "extracted_text" / "articles_text.parquet"
    classified_path = run_base / "processed" / "articles_classified.parquet"

    cdx = read_parquet_safe(cdx_path)
    homepage = read_parquet_safe(homepage_path)
    clean = read_parquet_safe(clean_path)
    articles = read_parquet_safe(articles_path)
    classified = read_parquet_safe(classified_path)

    cdx_sources = set(cdx["source"].dropna().astype(str)) if "source" in cdx.columns else set()
    homepage_sources = set(homepage["source"].dropna().astype(str)) if "source" in homepage.columns else set()
    clean_sources = set(clean["source"].dropna().astype(str)) if "source" in clean.columns else set()

    if not articles.empty and {"source", "text"}.issubset(articles.columns):
        article_sources_with_text = set(
            articles.loc[
                articles["text"].fillna("").astype(str).str.len().gt(0),
                "source",
            ]
            .dropna()
            .astype(str)
        )
    else:
        article_sources_with_text = set()

    if not classified.empty and "source" in classified.columns:
        classified_sources = set(classified["source"].dropna().astype(str))
    else:
        classified_sources = set()

    missing_homepage_sources = sorted(cdx_sources - homepage_sources)
    missing_clean_sources = sorted(homepage_sources - clean_sources)
    missing_text_sources = sorted(clean_sources - article_sources_with_text)
    missing_classified_sources = sorted(article_sources_with_text - classified_sources)

    fetch_or_parse_errors = None
    articles_with_text = None
    text_success_rate = None
    fetch_error_rate = None

    if not articles.empty:
        if "fetch_error" in articles.columns:
            fetch_or_parse_errors = int(articles["fetch_error"].notna().sum())
            fetch_error_rate = fetch_or_parse_errors / len(articles) if len(articles) else None

        if "text" in articles.columns:
            articles_with_text = int(
                articles["text"].fillna("").astype(str).str.len().gt(0).sum()
            )
            text_success_rate = articles_with_text / len(articles) if len(articles) else None

    has_text_clean = None
    has_title_clean = None
    case_strong = None
    case_review_possible_violence = None
    topic_gender_violence = None
    not_candidate = None

    if not classified.empty:
        has_text_clean = "text_clean" in classified.columns
        has_title_clean = "title_clean" in classified.columns

        if "retrieval_bucket" in classified.columns:
            counts = classified["retrieval_bucket"].value_counts(dropna=False)

            case_strong = int(counts.get("case_strong", 0))
            case_review_possible_violence = int(
                counts.get("case_review_possible_violence", 0)
            )
            topic_gender_violence = int(counts.get("topic_gender_violence", 0))
            not_candidate = int(counts.get("not_candidate", 0))

    if len(cdx) == 0:
        stage_issue = "no_cdx"
    elif len(homepage) == 0:
        stage_issue = "cdx_but_no_homepage_candidates"
    elif len(clean) == 0:
        stage_issue = "homepage_but_no_clean_candidates"
    elif len(articles) == 0:
        stage_issue = "clean_but_no_articles"
    elif classified.empty:
        stage_issue = "articles_but_no_classified"
    else:
        stage_issue = "ok"

    summary_row = {
        "run_id": window.run_id,
        "from_date": window.from_date,
        "to_date": window.to_date,
        "cdx_snapshots": len(cdx),
        "homepage_candidates": len(homepage),
        "clean_candidates": len(clean),
        "articles_text": len(articles),
        "articles_classified": len(classified),
        "sources_with_cdx": len(cdx_sources),
        "sources_with_homepage_candidates": len(homepage_sources),
        "sources_with_clean_candidates": len(clean_sources),
        "sources_with_text": len(article_sources_with_text),
        "sources_classified": len(classified_sources),
        "missing_homepage_sources": ",".join(missing_homepage_sources),
        "missing_clean_sources": ",".join(missing_clean_sources),
        "missing_text_sources": ",".join(missing_text_sources),
        "missing_classified_sources": ",".join(missing_classified_sources),
        "fetch_or_parse_errors": fetch_or_parse_errors,
        "fetch_error_rate": fetch_error_rate,
        "articles_with_text": articles_with_text,
        "text_success_rate": text_success_rate,
        "has_text_clean": has_text_clean,
        "has_title_clean": has_title_clean,
        "case_strong": case_strong,
        "case_review_possible_violence": case_review_possible_violence,
        "topic_gender_violence": topic_gender_violence,
        "not_candidate": not_candidate,
        "stage_issue": stage_issue,
    }

    by_source = expected_sources[["source", "country", "domain"]].copy()
    by_source["run_id"] = window.run_id
    by_source["from_date"] = window.from_date
    by_source["to_date"] = window.to_date

    for counts_df in [
        source_counts(cdx, "cdx_snapshots"),
        source_counts(homepage, "homepage_candidates"),
        source_counts(clean, "clean_candidates"),
    ]:
        by_source = by_source.merge(counts_df, on="source", how="left")

    if not articles.empty and "source" in articles.columns:
        articles_tmp = articles.copy()

        if "text" in articles_tmp.columns:
            articles_tmp["has_text"] = (
                articles_tmp["text"].fillna("").astype(str).str.len().gt(0)
            )
        else:
            articles_tmp["has_text"] = False

        if "fetch_error" in articles_tmp.columns:
            articles_tmp["has_error"] = articles_tmp["fetch_error"].notna()
        else:
            articles_tmp["has_error"] = False

        articles_by_source = (
            articles_tmp.groupby("source")
            .agg(
                articles_text=("source", "size"),
                articles_with_text=("has_text", "sum"),
                fetch_or_parse_errors=("has_error", "sum"),
            )
            .reset_index()
        )

        by_source = by_source.merge(articles_by_source, on="source", how="left")
    else:
        by_source["articles_text"] = 0
        by_source["articles_with_text"] = 0
        by_source["fetch_or_parse_errors"] = 0

    if not classified.empty and "source" in classified.columns:
        classified_by_source = (
            classified.groupby("source")
            .size()
            .reset_index(name="articles_classified")
        )

        by_source = by_source.merge(classified_by_source, on="source", how="left")
    else:
        by_source["articles_classified"] = 0

    diagnostics_summary_path = report_dir / "homepage_snapshot_diagnostics_summary.csv"
    diagnostics_summary = read_csv_safe(diagnostics_summary_path)

    if not diagnostics_summary.empty and "source" in diagnostics_summary.columns:
        keep_cols = [
            col
            for col in [
                "source",
                "snapshots_seen",
                "fetch_errors",
                "snapshots_with_links",
                "total_links_extracted",
                "avg_links_extracted",
            ]
            if col in diagnostics_summary.columns
        ]

        by_source = by_source.merge(
            diagnostics_summary[keep_cols],
            on="source",
            how="left",
        )

    numeric_cols = [
        "cdx_snapshots",
        "homepage_candidates",
        "clean_candidates",
        "articles_text",
        "articles_with_text",
        "fetch_or_parse_errors",
        "articles_classified",
        "snapshots_seen",
        "fetch_errors",
        "snapshots_with_links",
        "total_links_extracted",
        "avg_links_extracted",
    ]

    for col in numeric_cols:
        if col in by_source.columns:
            by_source[col] = by_source[col].fillna(0)

    def source_issue(row: pd.Series) -> str:
        if row.get("cdx_snapshots", 0) == 0:
            return "no_cdx"
        if row.get("homepage_candidates", 0) == 0:
            return "cdx_but_no_homepage_candidates"
        if row.get("clean_candidates", 0) == 0:
            return "homepage_but_no_clean_candidates"
        if row.get("articles_text", 0) == 0:
            return "clean_but_no_articles"
        if row.get("articles_with_text", 0) == 0:
            return "articles_without_text"
        if row.get("articles_classified", 0) == 0:
            return "articles_but_no_classified"
        return "ok"

    by_source["source_issue"] = by_source.apply(source_issue, axis=1)

    return summary_row, by_source


def make_human_report(
    summary: pd.DataFrame,
    by_source: pd.DataFrame,
    label: str,
) -> str:
    lines: list[str] = []

    total_windows = len(summary)
    ok_windows = int(summary["stage_issue"].eq("ok").sum()) if not summary.empty else 0
    issue_windows = total_windows - ok_windows

    total_cdx = int(summary["cdx_snapshots"].sum()) if "cdx_snapshots" in summary else 0
    total_homepage = int(summary["homepage_candidates"].sum()) if "homepage_candidates" in summary else 0
    total_clean = int(summary["clean_candidates"].sum()) if "clean_candidates" in summary else 0
    total_articles = int(summary["articles_text"].sum()) if "articles_text" in summary else 0
    total_with_text = int(summary["articles_with_text"].fillna(0).sum()) if "articles_with_text" in summary else 0
    total_classified = int(summary["articles_classified"].sum()) if "articles_classified" in summary else 0

    lines.append("=" * 80)
    lines.append(f"REPORTE FINAL DEL PIPELINE: {label}")
    lines.append("=" * 80)
    lines.append("")
    lines.append("RESUMEN GENERAL")
    lines.append(f"- Ventanas procesadas: {total_windows}")
    lines.append(f"- Ventanas OK: {ok_windows}")
    lines.append(f"- Ventanas con algo para revisar: {issue_windows}")
    lines.append(f"- CDX snapshots: {total_cdx:,}")
    lines.append(f"- Homepage candidates: {total_homepage:,}")
    lines.append(f"- Clean candidates: {total_clean:,}")
    lines.append(f"- Artículos procesados: {total_articles:,}")
    lines.append(f"- Artículos con texto: {total_with_text:,}")
    lines.append(f"- Artículos clasificados: {total_classified:,}")

    if total_articles:
        lines.append(f"- Tasa global con texto: {total_with_text / total_articles:.1%}")

    lines.append("")

    lines.append("DISTRIBUCIÓN DE PROBLEMAS POR VENTANA")
    if not summary.empty:
        issue_counts = summary["stage_issue"].value_counts(dropna=False)
        for issue, n in issue_counts.items():
            lines.append(f"- {issue}: {int(n)}")
    lines.append("")

    problem_windows = summary[summary["stage_issue"].ne("ok")].copy()

    lines.append("VENTANAS QUE REQUIEREN REVISIÓN")
    if problem_windows.empty:
        lines.append("- No hay ventanas con fallo de etapa.")
    else:
        cols = [
            "run_id",
            "stage_issue",
            "cdx_snapshots",
            "homepage_candidates",
            "clean_candidates",
            "articles_text",
            "articles_classified",
            "missing_homepage_sources",
            "missing_clean_sources",
            "missing_text_sources",
        ]

        cols = [col for col in cols if col in problem_windows.columns]
        lines.append(problem_windows[cols].head(30).to_string(index=False))

        if len(problem_windows) > 30:
            lines.append(f"... y {len(problem_windows) - 30} ventanas más. Ver CSV completo.")
    lines.append("")

    missing_homepage = summary[
        summary["missing_homepage_sources"].fillna("").astype(str).str.len().gt(0)
    ].copy()

    lines.append("FUENTES QUE APARECEN EN CDX PERO NO EN HOMEPAGE_CANDIDATES")
    if missing_homepage.empty:
        lines.append("- No se detectaron fuentes desaparecidas entre CDX y homepage_candidates.")
    else:
        cols = ["run_id", "missing_homepage_sources"]
        lines.append(missing_homepage[cols].head(40).to_string(index=False))

        if len(missing_homepage) > 40:
            lines.append(f"... y {len(missing_homepage) - 40} filas más. Ver CSV completo.")
    lines.append("")

    if "source_issue" in by_source.columns:
        source_problem = by_source[by_source["source_issue"].ne("ok")].copy()
    else:
        source_problem = pd.DataFrame()

    lines.append("PROBLEMAS POR FUENTE")
    if source_problem.empty:
        lines.append("- No se detectaron problemas por fuente.")
    else:
        cols = [
            "run_id",
            "source",
            "source_issue",
            "cdx_snapshots",
            "snapshots_seen",
            "fetch_errors",
            "snapshots_with_links",
            "homepage_candidates",
            "clean_candidates",
            "articles_text",
            "articles_with_text",
            "articles_classified",
        ]

        cols = [col for col in cols if col in source_problem.columns]
        lines.append(source_problem[cols].head(50).to_string(index=False))

        if len(source_problem) > 50:
            lines.append(f"... y {len(source_problem) - 50} filas más. Ver CSV completo.")
    lines.append("")

    high_fetch_error = summary[
        summary["fetch_error_rate"].fillna(0).gt(0.35)
    ].copy()

    lines.append("VENTANAS CON MUCHOS ERRORES DE DESCARGA/PARSE")
    if high_fetch_error.empty:
        lines.append("- No hay ventanas con fetch_error_rate > 35%.")
    else:
        cols = [
            "run_id",
            "fetch_or_parse_errors",
            "articles_text",
            "fetch_error_rate",
        ]
        lines.append(high_fetch_error[cols].head(30).to_string(index=False))
    lines.append("")

    low_text_success = summary[
        summary["text_success_rate"].fillna(1).lt(0.60)
    ].copy()

    lines.append("VENTANAS CON BAJA TASA DE TEXTO EXTRAÍDO")
    if low_text_success.empty:
        lines.append("- No hay ventanas con text_success_rate < 60%.")
    else:
        cols = [
            "run_id",
            "articles_with_text",
            "articles_text",
            "text_success_rate",
        ]
        lines.append(low_text_success[cols].head(30).to_string(index=False))
    lines.append("")

    classification_problem = summary[
        (summary["has_text_clean"].eq(False))
        | (summary["has_title_clean"].eq(False))
        | (summary["articles_classified"].fillna(0).eq(0))
    ].copy()

    lines.append("PROBLEMAS DE CLASIFICACIÓN / TEXTO LIMPIO")
    if classification_problem.empty:
        lines.append("- No se detectaron problemas obvios en text_clean/title_clean/classified.")
    else:
        cols = [
            "run_id",
            "articles_classified",
            "has_text_clean",
            "has_title_clean",
            "stage_issue",
        ]
        lines.append(classification_problem[cols].head(30).to_string(index=False))
    lines.append("")

    lines.append("CÓMO INTERPRETAR LOS PROBLEMAS")
    lines.append("- no_cdx: Wayback no devolvió snapshots para esa fuente/ventana.")
    lines.append("- cdx_but_no_homepage_candidates: revisar homepage_snapshot_diagnostics.csv.")
    lines.append("- homepage_but_no_clean_candidates: revisar filter_candidate_urls.py.")
    lines.append("- clean_but_no_articles: revisar descarga de artículos o URLs candidatas.")
    lines.append("- articles_without_text: revisar parse_article.py.")
    lines.append("- articles_but_no_classified: revisar build_relevance_dataset.py.")
    lines.append("")
    lines.append("ARCHIVOS CLAVE")
    lines.append(f"- outputs/reports/pipeline/pipeline_summary_{label}.csv")
    lines.append(f"- outputs/reports/pipeline/pipeline_by_source_{label}.csv")
    lines.append(f"- outputs/reports/pipeline/pipeline_problem_sources_{label}.csv")
    lines.append(f"- outputs/reports/pipeline/pipeline_review_{label}.txt")
    lines.append("")

    return "\n".join(lines)


def write_reports(
    windows: list[PipelineWindow],
    config: str,
    label: str,
) -> None:
    out_dir = ROOT / "outputs" / "reports" / "pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)

    expected_sources = load_expected_sources(config)

    summary_rows = []
    by_source_frames = []

    for window in windows:
        summary_row, by_source = summarize_window(
            window=window,
            expected_sources=expected_sources,
        )
        summary_rows.append(summary_row)
        by_source_frames.append(by_source)

    summary = pd.DataFrame(summary_rows)
    by_source_all = pd.concat(by_source_frames, ignore_index=True)

    summary_path = out_dir / f"pipeline_summary_{label}.csv"
    by_source_path = out_dir / f"pipeline_by_source_{label}.csv"
    problem_sources_path = out_dir / f"pipeline_problem_sources_{label}.csv"
    review_path = out_dir / f"pipeline_review_{label}.txt"

    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    by_source_all.to_csv(by_source_path, index=False, encoding="utf-8-sig")

    problem_sources = by_source_all[
        by_source_all["source_issue"].ne("ok")
    ].copy()

    problem_sources.to_csv(
        problem_sources_path,
        index=False,
        encoding="utf-8-sig",
    )

    review = make_human_report(
        summary=summary,
        by_source=by_source_all,
        label=label,
    )

    review_path.write_text(review, encoding="utf-8")

    print("\n" + review)
    print(f"\nReporte compacto guardado en: {review_path}")
    print(f"Resumen CSV guardado en: {summary_path}")
    print(f"Detalle por fuente guardado en: {by_source_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the final archived-news pipeline by month or by year."
    )

    parser.add_argument(
        "--mode",
        choices=["monthly", "yearly"],
        default="monthly",
        help="Modo de corrida final. Recomendado: monthly.",
    )

    parser.add_argument(
        "--start-year",
        type=int,
        default=2015,
        help="Año inicial.",
    )

    parser.add_argument(
        "--end-year",
        type=int,
        default=2025,
        help="Año final.",
    )

    parser.add_argument(
        "--start-month",
        type=int,
        default=None,
        help="Mes inicial opcional, solo útil para modo monthly.",
    )

    parser.add_argument(
        "--end-month",
        type=int,
        default=None,
        help="Mes final opcional, solo útil para modo monthly.",
    )

    parser.add_argument(
        "--config",
        default="configs/sources_argentina_mexico_full.yml",
        help="Config YAML final.",
    )

    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=2.0,
        help="Pausa entre descargas de artículos.",
    )

    parser.add_argument(
        "--skip-retry",
        action="store_true",
        help="No ejecutar retry_failed_article_texts.",
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continúa con la siguiente ventana si una etapa falla.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocesa aunque ya existan outputs.",
    )

    parser.add_argument(
        "--only-summary",
        action="store_true",
        help="No ejecuta el pipeline; solo reconstruye los reportes finales.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    windows = build_windows(
        mode=args.mode,
        start_year=args.start_year,
        end_year=args.end_year,
        start_month=args.start_month,
        end_month=args.end_month,
    )

    label = f"{args.mode}_{args.start_year}_{args.end_year}"

    if args.start_month or args.end_month:
        label += f"_{args.start_month or 1:02d}_{args.end_month or 12:02d}"

    print("\nFinal pipeline")
    print(f"Mode: {args.mode}")
    print(f"Years: {args.start_year}-{args.end_year}")
    print(f"Months: {args.start_month or 1}-{args.end_month or 12}")
    print(f"Config: {args.config}")
    print(f"Windows: {len(windows)}")

    if args.only_summary:
        write_reports(
            windows=windows,
            config=args.config,
            label=label,
        )
        return

    for window in windows:
        print("\n" + "#" * 100)
        print(f"START WINDOW: {window.run_id} | {window.from_date} - {window.to_date}")
        print("#" * 100)

        commands = pipeline_commands(
            window=window,
            config=args.config,
            sleep_seconds=args.sleep_seconds,
            skip_retry=args.skip_retry,
        )

        for stage, command in commands:
            if should_skip_stage(stage, window.run_id, force=args.force):
                print(f"SKIP {stage}: output already exists for {window.run_id}")
                continue

            ok = run_command(
                command=command,
                continue_on_error=args.continue_on_error,
            )

            if not ok:
                print(f"FAILED STAGE: {stage} | run_id={window.run_id}")
                break

    write_reports(
        windows=windows,
        config=args.config,
        label=label,
    )


if __name__ == "__main__":
    main()