from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

from src.utils.text_quality import contains_mojibake


# =====================================================================
# RUTAS
# =====================================================================

INPUT = Path(
    "outputs/final/case_articles_analysis.parquet"
)

OUTPUT = Path(
    "outputs/final/case_articles_phase1.parquet"
)

QUALITY_DIR = Path(
    "outputs/final/quality_phase1"
)

EXCLUDED_OUTPUT = (
    QUALITY_DIR / "excluded_non_article_records.csv"
)

DEDUP_OUTPUT = (
    QUALITY_DIR / "removed_exact_duplicates.csv"
)


# =====================================================================
# DETECCIÓN DE PÁGINAS QUE NO SON ARTÍCULOS
# =====================================================================

def classify_non_article(
    source: str,
    url: str,
    title: str,
    text: str,
) -> str | None:

    source = str(source or "").lower().strip()
    url = str(url or "").lower().strip()
    title = str(title or "").strip()
    text_lower = str(text or "").lower()

    # -------------------------------------------------------------
    # MILENIO
    # -------------------------------------------------------------

    if source == "milenio":

        # Página temática / listado.
        if "/temas/" in url:
            return "topic_listing_page"

        # Página principal de columnista:
        # /opinion/enrique-acevedo
        #
        # No confundir con una columna concreta:
        # /opinion/autor/seccion/titulo
        if re.search(
            r"milenio\.com/opinion/[^/]+/?$",
            url,
        ):
            return "author_listing_page"

        # Video sin cuerpo periodístico extraído.
        if (
            "/videos/" in url
            and
            "utilizamos cookies para darte la mejor experiencia"
            in text_lower
        ):
            return "video_cookie_page_without_article_text"

    # -------------------------------------------------------------
    # CLARÍN
    # -------------------------------------------------------------

    if source == "clarin_arg":

        if (
            "preferís suscribirte por teléfono"
            in text_lower
            and
            "preguntas frecuentes"
            in text_lower
        ):
            return "subscription_page_without_article_text"

    # -------------------------------------------------------------
    # LA NACIÓN
    # -------------------------------------------------------------

    if source == "lanacion":

        if "/tema/" in url:
            return "topic_listing_page"

    return None


# =====================================================================
# NORMALIZACIÓN PARA DEDUPLICACIÓN
# =====================================================================

def build_dedup_text(
    text: str,
) -> str:

    text = str(text or "").lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:

    print("=" * 72)
    print("CONSOLIDACIÓN DEL CORPUS ANALÍTICO")
    print("=" * 72)

    QUALITY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not INPUT.exists():
        raise FileNotFoundError(
            f"No existe el corpus de entrada: {INPUT}"
        )

    df = pd.read_parquet(INPUT).copy()

    print(
        f"\nRegistros de entrada: {len(df):,}"
    )

    # -------------------------------------------------------------
    # 1. Validar columnas
    # -------------------------------------------------------------

    required = [
        "country",
        "source",
        "candidate_url",
        "analysis_text",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Faltan columnas requeridas: {missing}"
        )

    # -------------------------------------------------------------
    # 2. Control de corrupción textual
    # -------------------------------------------------------------

    mojibake_mask = (
        df["analysis_text"]
        .fillna("")
        .map(contains_mojibake)
    )

    if mojibake_mask.any():

        print(
            df.loc[
                mojibake_mask,
                [
                    "source",
                    "candidate_url",
                ]
            ]
            .head(20)
            .to_string(index=False)
        )

        raise RuntimeError(
            "Todavía existe mojibake en analysis_text."
        )

    # Formas con acento grave espurio ya verificadas.
    known_bad_grave = re.compile(
        r"\b(?:"
        r"mà|ahà|allà|iranà|conocà|otomà|rubà|"
        r"nohemà|haità|pedà|vivà|israelà|"
        r"sobrevivà|noemà|anahà|alà|analà|anaà|"
        r"cundapà|marà|decidà|esquà|fungà|perdà|"
        r"prometà|referà|salà|sentà|volvà|ademà|"
        r"asà|aquà|creà|nacà|escogà|neftalà|"
        r"ocurriò|vìa|travès|comisiòn|bazaldùa|"
        r"mèxico|investigaciòn|desapariciòn|secretarìa"
        r")\b",
        flags=re.IGNORECASE,
    )

    grave_bad_mask = (
        df["analysis_text"]
        .fillna("")
        .str.contains(
            known_bad_grave,
            regex=True,
            na=False,
        )
    )

    if grave_bad_mask.any():

        print(
            "\nTodavía quedan formas corruptas:\n"
        )

        print(
            df.loc[
                grave_bad_mask,
                [
                    "source",
                    "candidate_url",
                    "analysis_text",
                ]
            ]
            .head(20)
            .to_string(
                index=False,
                max_colwidth=250,
            )
        )

        raise RuntimeError(
            "Quedan formas conocidas con "
            "acentos graves espurios."
        )

    # -------------------------------------------------------------
    # 3. Detectar páginas no documentales
    # -------------------------------------------------------------

    if "title" in df.columns:
        titles = df["title"].fillna("")
    else:
        titles = pd.Series(
            "",
            index=df.index,
        )

    df["analysis_exclusion_reason"] = [
        classify_non_article(
            source=source,
            url=url,
            title=title,
            text=text,
        )
        for source, url, title, text in zip(
            df["source"].fillna(""),
            df["candidate_url"].fillna(""),
            titles,
            df["analysis_text"].fillna(""),
        )
    ]

    excluded = (
        df.loc[
            df["analysis_exclusion_reason"].notna()
        ]
        .copy()
    )

    valid = (
        df.loc[
            df["analysis_exclusion_reason"].isna()
        ]
        .copy()
    )

    exclusion_columns = [
        column
        for column in [
            "country",
            "source",
            "title",
            "candidate_url",
            "normalized_url",
            "analysis_exclusion_reason",
            "analysis_text",
        ]
        if column in excluded.columns
    ]

    excluded[
        exclusion_columns
    ].to_csv(
        EXCLUDED_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\nRegistros no documentales excluidos: "
        f"{len(excluded):,}"
    )

    if not excluded.empty:

        print("\nMotivos:")

        print(
            excluded[
                "analysis_exclusion_reason"
            ]
            .value_counts()
            .to_string()
        )

    # -------------------------------------------------------------
    # 4. Clave de deduplicación
    #
    # Se deduplica únicamente dentro del mismo país y medio.
    # -------------------------------------------------------------

    valid["_dedup_text"] = (
        valid["analysis_text"]
        .map(build_dedup_text)
    )

    dedup_columns = [
        "country",
        "source",
        "_dedup_text",
    ]

    valid["duplicate_count"] = (
        valid
        .groupby(
            dedup_columns,
            dropna=False,
        )["_dedup_text"]
        .transform("size")
    )

    valid["duplicate_urls"] = (
        valid
        .groupby(
            dedup_columns,
            dropna=False,
        )["candidate_url"]
        .transform(
            lambda values: " || ".join(
                dict.fromkeys(
                    str(value)
                    for value in values
                    if pd.notna(value)
                )
            )
        )
    )

    # -------------------------------------------------------------
    # 5. Registrar las filas duplicadas que se eliminan
    # -------------------------------------------------------------

    duplicate_removed_mask = (
        valid.duplicated(
            subset=dedup_columns,
            keep="first",
        )
    )

    duplicate_removed = (
        valid.loc[
            duplicate_removed_mask
        ]
        .copy()
    )

    duplicate_output_columns = [
        column
        for column in [
            "country",
            "source",
            "title",
            "candidate_url",
            "normalized_url",
            "duplicate_count",
            "duplicate_urls",
            "analysis_text",
        ]
        if column in duplicate_removed.columns
    ]

    duplicate_removed[
        duplicate_output_columns
    ].to_csv(
        DEDUP_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------
    # 6. Corpus final
    # -------------------------------------------------------------

    final = (
        valid.loc[
            ~duplicate_removed_mask
        ]
        .copy()
    )

    final = final.drop(
        columns=[
            "_dedup_text",
            "analysis_exclusion_reason",
        ],
        errors="ignore",
    )

    # -------------------------------------------------------------
    # 7. Validación posterior
    # -------------------------------------------------------------

    final_check = pd.DataFrame(
        {
            "country": final["country"],
            "source": final["source"],
            "_text": (
                final["analysis_text"]
                .map(build_dedup_text)
            ),
        }
    )

    remaining_duplicates = (
        final_check
        .duplicated(
            subset=[
                "country",
                "source",
                "_text",
            ],
            keep=False,
        )
        .sum()
    )

    if remaining_duplicates:
        raise RuntimeError(
            "Todavía quedan duplicados exactos "
            "dentro del mismo medio."
        )

    if final["analysis_text"].isna().any():
        raise RuntimeError(
            "Existen analysis_text nulos."
        )

    if (
        final["analysis_text"]
        .str.strip()
        .eq("")
        .any()
    ):
        raise RuntimeError(
            "Existen analysis_text vacíos."
        )

    # -------------------------------------------------------------
    # 8. Guardar
    # -------------------------------------------------------------

    final.to_parquet(
        OUTPUT,
        index=False,
    )

    print(
        f"\nDuplicados exactos eliminados: "
        f"{len(duplicate_removed):,}"
    )

    print(
        f"Corpus final Fase 1: "
        f"{len(final):,}"
    )

    print("\nArchivos generados:")

    print(
        f"- {OUTPUT}"
    )

    print(
        f"- {EXCLUDED_OUTPUT}"
    )

    print(
        f"- {DEDUP_OUTPUT}"
    )


if __name__ == "__main__":
    main()