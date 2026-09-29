from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import hashlib
import re

import pandas as pd

from src.utils.text_quality import (
    contains_html,
    contains_mojibake,
    contains_url,
)


# =====================================================================
# RUTAS
# =====================================================================

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT = (
    ROOT
    / "outputs"
    / "analysis"
    / "2015_2025"
    / "articles.parquet"
)

DEFAULT_OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "analysis"
    / "2015_2025"
    / "quality"
)


# =====================================================================
# COLUMNAS TEXTUALES POSIBLES
# =====================================================================

TEXT_COLUMN_CANDIDATES = [
    "analysis_text",
    "text_clean",
    "text",
    "article_text",
    "full_text",
    "clean_text",
    "body",
    "content",
]


# =====================================================================
# UTILIDADES
# =====================================================================

def detect_text_column(
    df: pd.DataFrame,
    requested: str | None = None,
) -> str:
    """
    Determina qué columna contiene el cuerpo textual del artículo.

    Si se especifica --text-col, utiliza esa columna.
    En caso contrario, intenta reconocer nombres habituales.
    """

    if requested is not None:
        if requested not in df.columns:
            raise ValueError(
                f"La columna solicitada '{requested}' no existe.\n"
                f"Columnas disponibles:\n{list(df.columns)}"
            )

        return requested

    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in df.columns:
            return candidate

    raise ValueError(
        "No se pudo identificar automáticamente la columna de texto.\n\n"
        "Columnas disponibles:\n"
        f"{list(df.columns)}\n\n"
        "Ejecuta nuevamente indicando explícitamente:\n"
        "python -m src.analysis.audit_corpus_quality "
        "--text-col NOMBRE_COLUMNA"
    )


def normalized_text_hash(text: str) -> str:
    """
    Hash de una representación sencilla normalizada del texto.
    Se utiliza para detectar duplicados exactos.
    """

    normalized = re.sub(
        r"\s+",
        " ",
        str(text).lower(),
    ).strip()

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()

def extract_suspicious_tokens(
    df: pd.DataFrame,
    text_col: str,
) -> pd.DataFrame:
    """
    Extrae tokens que contienen caracteres típicos de mojibake
    para conocer qué errores de codificación son más frecuentes.
    """

    counter = Counter()

    pattern = re.compile(
        r"\S*(?:Ã|Â|â|�)\S*"
    )

    for text in df[text_col].fillna("").astype(str):

        tokens = pattern.findall(text)

        for token in tokens:
            token = token.strip(
                ".,;:!?()[]{}\"'“”‘’"
            )

            if token:
                counter[token] += 1

    records = [
        {
            "token": token,
            "frequency": frequency,
        }
        for token, frequency in counter.most_common()
    ]

    return pd.DataFrame(records)


# =====================================================================
# FRAGMENTOS REPETIDOS
# =====================================================================

def find_repeated_fragments(
    df: pd.DataFrame,
    text_col: str,
) -> pd.DataFrame:
    """
    Busca líneas que aparecen repetidamente dentro de artículos
    del mismo medio.

    Esto ayuda a descubrir footers, navegación, copyright,
    elementos del CMS, etc.
    """

    newspaper_col = None

    for candidate in [
        "newspaper",
        "source",
        "media",
        "domain",
    ]:
        if candidate in df.columns:
            newspaper_col = candidate
            break

    if newspaper_col is None:
        print(
            "\nNo existe una columna de medio reconocible. "
            "Se omite auditoría de fragmentos por medio."
        )
        return pd.DataFrame()

    records = []

    for newspaper, subset in df.groupby(
        newspaper_col,
        dropna=False,
    ):
        counter = Counter()

        for text in subset[text_col].fillna("").astype(str):

            # Cada fragmento cuenta como máximo una vez
            # dentro del mismo artículo.
            lines = {
                re.sub(r"\s+", " ", line.strip().lower())
                for line in text.splitlines()
                if 20 <= len(line.strip()) <= 300
            }

            counter.update(lines)

        n_articles = len(subset)

        for fragment, count in counter.items():

            prevalence = (
                count / n_articles
                if n_articles
                else 0
            )

            # Repetición suficientemente frecuente para
            # ser interesante como posible boilerplate.
            if (
                count >= 10
                and prevalence >= 0.05
            ):
                records.append(
                    {
                        "newspaper": newspaper,
                        "fragment": fragment,
                        "articles": count,
                        "total_articles_medium": n_articles,
                        "prevalence": prevalence,
                    }
                )

    if not records:
        return pd.DataFrame()

    repeated = pd.DataFrame(records)

    repeated = repeated.sort_values(
        [
            "newspaper",
            "prevalence",
            "articles",
        ],
        ascending=[
            True,
            False,
            False,
        ],
    )

    return repeated




# =====================================================================
# AUDITORÍA PRINCIPAL
# =====================================================================

def audit_corpus(
    input_path: Path,
    output_dir: Path,
    requested_text_col: str | None,
) -> None:

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("AUDITORÍA DE CALIDAD DEL CORPUS")
    print("=" * 70)

    print(f"\nCorpus: {input_path}")

    df = pd.read_parquet(input_path)

    df = df.loc[
        df["analysis_eligible"]
    ].copy()

    print(f"Artículos: {len(df):,}")
    print(f"Columnas: {len(df.columns)}")

    # -----------------------------------------------------------------
    # 1. IDENTIFICAR COLUMNA DE TEXTO
    # -----------------------------------------------------------------

    text_col = detect_text_column(
        df,
        requested=requested_text_col,
    )

    print(
        f"\nColumna textual utilizada: "
        f"{text_col}"
    )

    # -----------------------------------------------------------------
    # 2. CONSERVAR TEXTO ORIGINAL
    # -----------------------------------------------------------------

    df["audit_original_text"] = (
        df[text_col]
        .fillna("")
        .astype(str)
    )

    # -----------------------------------------------------------------
    # 3. INDICADORES DE PROBLEMAS
    # -----------------------------------------------------------------

    df["audit_has_mojibake"] = (
        df["audit_original_text"]
        .map(contains_mojibake)
    )

    df["audit_has_html"] = (
        df["audit_original_text"]
        .map(contains_html)
    )

    df["audit_has_url"] = (
        df["audit_original_text"]
        .map(contains_url)
    )

    df["audit_original_length"] = (
        df["audit_original_text"]
        .str.len()
    )

    # -----------------------------------------------------------------
    # 6. TEXTOS SOSPECHOSOS
    # -----------------------------------------------------------------

    df["audit_suspicious"] = (
        df["audit_has_mojibake"]
        | df["audit_has_html"]
        | df["audit_has_url"]
        | (df["audit_original_length"] < 200)
    )

    # -----------------------------------------------------------------
    # 7. DUPLICADOS EXACTOS
    # -----------------------------------------------------------------

    df["audit_text_hash"] = (
        df["audit_original_text"]
        .map(normalized_text_hash)
    )

    duplicate_mask = (
        df.duplicated(
            "audit_text_hash",
            keep=False,
        )
        & (df["audit_original_length"] > 0)
    )

    df["audit_exact_duplicate"] = duplicate_mask

    # -----------------------------------------------------------------
    # 8. EXPORTAR ARTÍCULOS SOSPECHOSOS
    # -----------------------------------------------------------------

    metadata_candidates = [
        "article_id",
        "country",
        "newspaper",
        "source",
        "year",
        "date",
        "published_at",
        "candidate_url",
        "normalized_url",
        "title",
    ]

    metadata_columns = [
        c
        for c in metadata_candidates
        if c in df.columns
    ]

    audit_columns = [
        "audit_has_mojibake",
        "audit_has_html",
        "audit_has_url",
        "audit_original_length",
        "audit_exact_duplicate",
        "audit_original_text",
    ]

    suspicious_columns = (
        metadata_columns
        + audit_columns
    )

    suspicious = df.loc[
        df["audit_suspicious"],
        suspicious_columns,
    ]

    suspicious.to_csv(
        output_dir / "suspicious_articles.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------------------
    # 9. EXPORTAR CASOS DE MOJIBAKE
    # -----------------------------------------------------------------

    encoding_problems = df.loc[
        df["audit_has_mojibake"],
        suspicious_columns,
    ]

    encoding_problems.to_csv(
        output_dir / "encoding_problems.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------------------
    # 10. EXPORTAR CASOS HTML
    # -----------------------------------------------------------------

    html_problems = df.loc[
        df["audit_has_html"],
        suspicious_columns,
    ]

    html_problems.to_csv(
        output_dir / "html_problems.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------------------
    # 11. EXPORTAR CASOS CON URLS
    # -----------------------------------------------------------------

    url_problems = df.loc[
        df["audit_has_url"],
        suspicious_columns,
    ]

    url_problems.to_csv(
        output_dir / "url_problems.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------------------
    # 12. DUPLICADOS
    # -----------------------------------------------------------------

    duplicate_columns = (
        metadata_columns
        + [
            "audit_text_hash",
            "audit_original_length",
            "audit_original_text",
        ]
    )

    duplicates = (
        df.loc[
            duplicate_mask,
            duplicate_columns,
        ]
        .sort_values("audit_text_hash")
    )

    duplicates.to_csv(
        output_dir / "exact_duplicates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------------------
    # 13. FRAGMENTOS REPETIDOS
    # -----------------------------------------------------------------

    repeated = find_repeated_fragments(
        df,
        text_col="audit_original_text",
    )

    if not repeated.empty:
        repeated.to_csv(
            output_dir / "repeated_fragments.csv",
            index=False,
            encoding="utf-8-sig",
        )

    # -----------------------------------------------------------------
    # 13B. TOKENS DE MOJIBAKE MÁS FRECUENTES
    # -----------------------------------------------------------------

    suspicious_tokens = extract_suspicious_tokens(
        df,
        "audit_original_text",
    )

    suspicious_tokens.to_csv(
        output_dir / "mojibake_tokens.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------------------
    # 14. RESUMEN GLOBAL
    # -----------------------------------------------------------------

    summary = pd.DataFrame(
        [
            {
                "metric": "articles",
                "value": len(df),
            },
            {
                "metric": "articles_with_mojibake",
                "value": int(
                    df["audit_has_mojibake"].sum()
                ),
            },
            {
                "metric": "articles_with_html",
                "value": int(
                    df["audit_has_html"].sum()
                ),
            },
            {
                "metric": "articles_with_urls",
                "value": int(
                    df["audit_has_url"].sum()
                ),
            },
            {
                "metric": "suspicious_articles",
                "value": int(
                    df["audit_suspicious"].sum()
                ),
            },
            {
                "metric": "exact_duplicate_rows",
                "value": int(
                    df["audit_exact_duplicate"].sum()
                ),
            },
            {
                "metric": "mean_original_length",
                "value": (
                    df["audit_original_length"]
                    .mean()
                ),
            },
        ]
    )

    summary.to_csv(
        output_dir / "corpus_quality_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------------------
    # 15. AUDITORÍA POR MEDIO
    # -----------------------------------------------------------------

    newspaper_col = None

    for candidate in [
        "newspaper",
        "source",
        "media",
        "domain",
    ]:
        if candidate in df.columns:
            newspaper_col = candidate
            break

    country_col = (
        "country"
        if "country" in df.columns
        else None
    )

    group_columns = [
        c
        for c in [
            country_col,
            newspaper_col,
        ]
        if c is not None
    ]

    if group_columns:

        by_source = (
            df
            .groupby(
                group_columns,
                dropna=False,
            )
            .agg(
                articles=(
                    "audit_original_text",
                    "size",
                ),
                mojibake=(
                    "audit_has_mojibake",
                    "sum",
                ),
                html=(
                    "audit_has_html",
                    "sum",
                ),
                url=(
                    "audit_has_url",
                    "sum",
                ),
                suspicious=(
                    "audit_suspicious",
                    "sum",
                ),
                exact_duplicates=(
                    "audit_exact_duplicate",
                    "sum",
                ),
            )
            .reset_index()
        )

        by_source["pct_mojibake"] = (
            100
            * by_source["mojibake"]
            / by_source["articles"]
        )

        by_source["pct_html"] = (
            100
            * by_source["html"]
            / by_source["articles"]
        )

        by_source["pct_url"] = (
            100
            * by_source["url"]
            / by_source["articles"]
        )

        by_source["pct_suspicious"] = (
            100
            * by_source["suspicious"]
            / by_source["articles"]
        )

        by_source.to_csv(
            output_dir / "quality_by_source.csv",
            index=False,
            encoding="utf-8-sig",
        )

    # -----------------------------------------------------------------
    # 16. GUARDAR DATAFRAME AUDITADO
    # -----------------------------------------------------------------

    audited_path = (
        output_dir
        / "articles_audited.parquet"
    )

    df.to_parquet(
        audited_path,
        index=False,
    )

    # -----------------------------------------------------------------
    # RESULTADOS
    # -----------------------------------------------------------------

    print("\n" + "=" * 70)
    print("RESULTADOS")
    print("=" * 70)

    print(
        f"\nMojibake: "
        f"{df['audit_has_mojibake'].sum():,}"
    )

    print(
        f"HTML residual: "
        f"{df['audit_has_html'].sum():,}"
    )

    print(
        f"URLs residuales: "
        f"{df['audit_has_url'].sum():,}"
    )

    print(
        f"Artículos sospechosos: "
        f"{df['audit_suspicious'].sum():,}"
    )

    print(
        f"Filas involucradas en duplicados exactos: "
        f"{df['audit_exact_duplicate'].sum():,}"
    )

    print(
        "\nArchivos generados en:"
        f"\n{output_dir.resolve()}"
    )

# =====================================================================
# CLI
# =====================================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Audita problemas de calidad textual "
            "del corpus final."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=(
            "Corpus parquet a auditar."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directorio donde guardar "
            "los resultados de auditoría."
        ),
    )

    parser.add_argument(
        "--text-col",
        type=str,
        default=None,
        help=(
            "Nombre explícito de la columna "
            "que contiene el cuerpo textual."
        ),
    )

    args = parser.parse_args()

    audit_corpus(
        input_path=args.input,
        output_dir=args.output_dir,
        requested_text_col=args.text_col,
    )


if __name__ == "__main__":
    main()