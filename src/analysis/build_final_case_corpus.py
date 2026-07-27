from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "outputs" / "reports" / "articles_contextual_unique.parquet"
OUTPUT_DIR = ROOT / "outputs" / "final"
DECISIONS_PATH = (
    ROOT
    / "data"
    / "annotations"
    / "female_aggressor_scope_decisions.csv"
)

STRICT_STATUSES = {
    "candidate_explicit_case",
    "candidate_contextual_high",
}

SENSITIVITY_STATUSES = {
    "review_contextual_medium",
}

AMBIGUOUS_STATUSES = {
    "review_ambiguous_case",
}

VALID_MANUAL_DECISIONS = {
    "",
    "include",
    "exclude",
    "review",
}


def normalize_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFD", text)
    text = "".join(
        char
        for char in text
        if unicodedata.category(char) != "Mn"
    )
    text = re.sub(r"\s+", " ", text.lower())
    return text.strip()


def first_value(
    row: pd.Series,
    columns: list[str],
) -> str:
    for column in columns:
        if column not in row.index:
            continue

        value = row.get(column)

        if pd.notna(value) and str(value).strip():
            return str(value)

    return ""


def build_analysis_text(row: pd.Series) -> str:
    title = first_value(
        row,
        ["title_clean", "title", "title_raw"],
    )
    anchor = first_value(row, ["anchor_text"])
    body = first_value(
        row,
        ["text_clean", "text", "text_raw", "text_excerpt"],
    )

    return normalize_text(
        "\n".join([title, anchor, body[:4000]])
    )


# Estas expresiones solo generan alertas.
# Nunca excluyen automáticamente un artículo.
FEMALE_SUBJECT = (
    r"(?:la|una)\s+"
    r"(?:mujer|esposa|novia|exnovia|expareja|amante|"
    r"madre|hija|tia|hermana|inquilina)"
)

VIOLENT_VERB = (
    r"(?:asesino|mato|golpeo|apunalo|acuchillo|"
    r"estrangulo|baleo|descuartizo|enveneno|"
    r"drogo|asfixio|arrojo|lanzo)"
)

AUXILIARY = (
    r"(?:(?:habia|habria|presuntamente)\s+)?"
    r"(?:(?:lo|la)\s+)?"
)

MALE_TARGET = (
    r"(?:a\s+)?(?:su\s+)?"
    r"(?:esposo|marido|novio|exnovio|pareja|hombre|hijo)"
)

FEMALE_TARGET = (
    r"(?:a\s+)?(?:la|una|su)\s+"
    r"(?:mujer|joven|adolescente|chica|menor|"
    r"amante|novia|esposa|exnovia|madre|hija)"
)


CLEAR_FEMALE_TO_MALE_PATTERNS = [
    re.compile(
        rf"\b{FEMALE_SUBJECT}\s+"
        rf"{AUXILIARY}{VIOLENT_VERB}\s+"
        rf"{MALE_TARGET}\b"
    ),
    re.compile(
        r"\b(?:ella|la mujer|la esposa|la novia|"
        r"la madre|la hija)\s+"
        r"(?:(?:habia|habria)\s+)?"
        r"(?:lo\s+)?"
        rf"{VIOLENT_VERB}\b"
    ),
    re.compile(
        rf"\b{VIOLENT_VERB}\s+a\s+su\s+"
        r"(?:esposo|marido|novio|pareja|hijo)\b"
    ),
]


CLEAR_FEMALE_TO_FEMALE_PATTERNS = [
    re.compile(
        rf"\b{FEMALE_SUBJECT}\s+"
        rf"{AUXILIARY}{VIOLENT_VERB}\s+"
        rf"{FEMALE_TARGET}\b"
    ),
    re.compile(
        r"\b(?:ella|la mujer|la esposa|la novia|"
        r"la madre|la hija|la tia)\s+"
        r"(?:(?:habia|habria)\s+)?"
        r"(?:la\s+)?"
        rf"{VIOLENT_VERB}\b"
        r".{0,70}\b"
        r"(?:mujer|joven|adolescente|chica|amante|"
        r"madre|hija|sobrina)\b"
    ),
]


POSSIBLE_FEMALE_AGGRESSOR_PATTERNS = [
    re.compile(
        r"\b(?:detuvieron|arrestaron|apresaron|"
        r"imputaron|acusaron)\s+a\s+"
        r"(?:la|una)\s+"
        r"(?:mujer|esposa|novia|exnovia|expareja|"
        r"madre|hija|tia|hermana)\b"
    ),
    re.compile(
        r"\b(?:la|una)\s+"
        r"(?:mujer|esposa|novia|exnovia|expareja|"
        r"madre|hija|tia|hermana)\b"
        r".{0,90}\b"
        r"(?:detenida|arrestada|apresada|imputada|"
        r"acusada|sospechosa)\b"
    ),
    re.compile(
        r"\bse sospecha que (?:ella|la mujer|la esposa|"
        r"la madre|la hija)\b"
        r".{0,100}\b"
        r"(?:cometio|participo en|mando a cometer)\s+"
        r"(?:el crimen|el homicidio|el asesinato)\b"
    ),
]


def matches_any(
    text: str,
    patterns: list[re.Pattern[str]],
) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def classify_scope_alert(
    row: pd.Series,
) -> pd.Series:
    """
    Devuelve solo una alerta automática.

    La exclusión final depende exclusivamente de
    manual_scope_decision == "exclude".
    """
    direction = str(
        row.get("violence_direction", "") or ""
    ).strip()

    text = build_analysis_text(row)

    if direction == "female_to_male":
        return pd.Series(
            {
                "auto_scope_status": "alert",
                "auto_scope_reason": (
                    "female_aggressor_male_victim"
                ),
            }
        )

    if matches_any(
        text,
        CLEAR_FEMALE_TO_MALE_PATTERNS,
    ):
        return pd.Series(
            {
                "auto_scope_status": "alert",
                "auto_scope_reason": (
                    "possible_female_aggressor_male_victim"
                ),
            }
        )

    if matches_any(
        text,
        CLEAR_FEMALE_TO_FEMALE_PATTERNS,
    ):
        return pd.Series(
            {
                "auto_scope_status": "alert",
                "auto_scope_reason": (
                    "possible_female_aggressor_female_victim"
                ),
            }
        )

    if matches_any(
        text,
        POSSIBLE_FEMALE_AGGRESSOR_PATTERNS,
    ):
        return pd.Series(
            {
                "auto_scope_status": "alert",
                "auto_scope_reason": (
                    "possible_female_aggressor"
                ),
            }
        )

    return pd.Series(
        {
            "auto_scope_status": "no_alert",
            "auto_scope_reason": "",
        }
    )


def series_or_empty(
    df: pd.DataFrame,
    column: str,
) -> pd.Series:
    if column in df.columns:
        return (
            df[column]
            .fillna("")
            .astype("string")
            .str.strip()
        )

    return pd.Series(
        "",
        index=df.index,
        dtype="string",
    )


def build_article_key(df: pd.DataFrame) -> pd.Series:
    """
    Construye una clave estable para unir decisiones manuales.
    """
    normalized_url = series_or_empty(
        df,
        "normalized_url",
    )
    candidate_url = series_or_empty(
        df,
        "candidate_url",
    )
    source = series_or_empty(df, "source")
    year = series_or_empty(df, "archive_year")

    if "title_clean" in df.columns:
        title = series_or_empty(df, "title_clean")
    else:
        title = series_or_empty(df, "title")

    article_key = normalized_url.mask(
        normalized_url.eq(""),
        candidate_url,
    )

    fallback = (
        "fallback|"
        + source
        + "|"
        + year
        + "|"
        + title
    )

    return article_key.mask(
        article_key.eq(""),
        fallback,
    )


def load_manual_decisions(
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    candidates = candidates.copy()
    candidates["article_key"] = build_article_key(
        candidates
    )

    if not DECISIONS_PATH.exists():
        candidates["manual_scope_decision"] = ""
        return candidates

    decisions = pd.read_csv(
        DECISIONS_PATH,
        dtype="string",
    )

    if "manual_scope_decision" not in decisions.columns:
        raise KeyError(
            "El archivo de decisiones no contiene "
            "'manual_scope_decision'."
        )

    if "article_key" not in decisions.columns:
        decisions["article_key"] = build_article_key(
            decisions
        )

    decisions["manual_scope_decision"] = (
        decisions["manual_scope_decision"]
        .fillna("")
        .astype("string")
        .str.strip()
        .str.lower()
    )

    invalid = sorted(
        set(decisions["manual_scope_decision"])
        - VALID_MANUAL_DECISIONS
    )

    if invalid:
        raise ValueError(
            "Valores inválidos en manual_scope_decision: "
            + ", ".join(invalid)
            + ". Permitidos: include, exclude, review o vacío."
        )

    decisions = (
        decisions[
            [
                "article_key",
                "manual_scope_decision",
            ]
        ]
        .drop_duplicates(
            subset=["article_key"],
            keep="last",
        )
    )

    candidates = candidates.merge(
        decisions,
        on="article_key",
        how="left",
    )

    candidates["manual_scope_decision"] = (
        candidates["manual_scope_decision"]
        .fillna("")
        .astype("string")
        .str.strip()
        .str.lower()
    )

    return candidates


def export_pending_decisions(
    candidates: pd.DataFrame,
) -> None:
    pending = candidates[
        candidates["auto_scope_status"].eq("alert")
        & candidates["manual_scope_decision"].eq("")
    ].copy()

    columns = [
        "article_key",
        "normalized_url",
        "candidate_url",
        "archive_year",
        "country",
        "source",
        "provisional_case_status",
        "gender_recognition_mode",
        "violence_direction",
        "auto_scope_status",
        "auto_scope_reason",
        "title_clean",
        "title",
        "text_excerpt",
        "manual_scope_decision",
    ]

    existing = [
        column
        for column in columns
        if column in pending.columns
    ]

    pending[existing].to_csv(
        OUTPUT_DIR
        / "female_aggressor_scope_decisions_pending.csv",
        index=False,
        encoding="utf-8-sig",
    )


def save_parquet(
    df: pd.DataFrame,
    filename: str,
) -> None:
    df.to_parquet(
        OUTPUT_DIR / filename,
        index=False,
    )


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró: {INPUT_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_parquet(INPUT_PATH)

    if "provisional_case_status" not in df.columns:
        raise KeyError(
            "Falta la columna provisional_case_status."
        )

    candidate_statuses = (
        STRICT_STATUSES
        | SENSITIVITY_STATUSES
        | AMBIGUOUS_STATUSES
    )

    candidates = df[
        df["provisional_case_status"].isin(
            candidate_statuses
        )
    ].copy()

    alerts = candidates.apply(
        classify_scope_alert,
        axis=1,
    )

    candidates = pd.concat(
        [
            candidates.reset_index(drop=True),
            alerts.reset_index(drop=True),
        ],
        axis=1,
    )

    candidates = load_manual_decisions(candidates)
    export_pending_decisions(candidates)

    # Solo una decisión manual "exclude" elimina el artículo.
    not_manually_excluded = (
        ~candidates["manual_scope_decision"].eq(
            "exclude"
        )
    )

    main_corpus = candidates[
        candidates["provisional_case_status"].isin(
            STRICT_STATUSES
        )
        & not_manually_excluded
    ].copy()

    sensitivity_corpus = candidates[
        candidates["provisional_case_status"].isin(
            SENSITIVITY_STATUSES
        )
        & not_manually_excluded
    ].copy()

    ambiguous_corpus = candidates[
        candidates["provisional_case_status"].isin(
            AMBIGUOUS_STATUSES
        )
        & not_manually_excluded
    ].copy()

    excluded = candidates[
        candidates["manual_scope_decision"].eq(
            "exclude"
        )
    ].copy()

    manually_included = candidates[
        candidates["manual_scope_decision"].eq(
            "include"
        )
    ].copy()

    pending_review = candidates[
        (
            candidates["auto_scope_status"].eq(
                "alert"
            )
            & candidates[
                "manual_scope_decision"
            ].eq("")
        )
        | candidates[
            "manual_scope_decision"
        ].eq("review")
    ].copy()

    save_parquet(
        main_corpus,
        "case_articles_main.parquet",
    )
    save_parquet(
        sensitivity_corpus,
        "case_articles_sensitivity_medium.parquet",
    )
    save_parquet(
        ambiguous_corpus,
        "case_articles_ambiguous.parquet",
    )
    save_parquet(
        excluded,
        "excluded_by_manual_scope_decision.parquet",
    )
    save_parquet(
        manually_included,
        "manually_included_scope_articles.parquet",
    )
    save_parquet(
        pending_review,
        "scope_alerts_pending_review.parquet",
    )

    summary = pd.DataFrame(
        [
            {
                "all_contextual_candidates": len(candidates),
                "main_case_articles": len(main_corpus),
                "sensitivity_medium_articles": (
                    len(sensitivity_corpus)
                ),
                "ambiguous_articles": (
                    len(ambiguous_corpus)
                ),
                "manual_exclusions": len(excluded),
                "manual_inclusions": (
                    len(manually_included)
                ),
                "pending_scope_alerts": (
                    len(pending_review)
                ),
            }
        ]
    )

    summary.to_csv(
        OUTPUT_DIR
        / "final_case_corpus_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    main_distribution = (
        main_corpus.groupby(
            [
                "provisional_case_status",
                "gender_recognition_mode",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="n_articles")
        .sort_values(
            [
                "provisional_case_status",
                "gender_recognition_mode",
            ]
        )
    )

    main_distribution.to_csv(
        OUTPUT_DIR
        / "main_corpus_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if excluded.empty:
        exclusion_summary = pd.DataFrame(
            columns=[
                "provisional_case_status",
                "auto_scope_reason",
                "n_articles",
            ]
        )
    else:
        exclusion_summary = (
            excluded.groupby(
                [
                    "provisional_case_status",
                    "auto_scope_reason",
                ],
                dropna=False,
            )
            .size()
            .reset_index(name="n_articles")
            .sort_values(
                "n_articles",
                ascending=False,
            )
        )

    exclusion_summary.to_csv(
        OUTPUT_DIR
        / "manual_scope_exclusion_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nCorpus final creado:\n")
    print(summary.to_string(index=False))

    print("\nDistribución del corpus principal:\n")
    print(main_distribution.to_string(index=False))

    print(
        "\nArchivos principales:\n"
        f"- {OUTPUT_DIR / 'case_articles_main.parquet'}\n"
        f"- {OUTPUT_DIR / 'case_articles_sensitivity_medium.parquet'}\n"
        f"- {OUTPUT_DIR / 'case_articles_ambiguous.parquet'}\n"
        f"- {OUTPUT_DIR / 'excluded_by_manual_scope_decision.parquet'}\n"
        f"- {OUTPUT_DIR / 'scope_alerts_pending_review.parquet'}\n"
        f"- {OUTPUT_DIR / 'female_aggressor_scope_decisions_pending.csv'}\n"
        f"- {OUTPUT_DIR / 'final_case_corpus_summary.csv'}"
    )


if __name__ == "__main__":
    main()
