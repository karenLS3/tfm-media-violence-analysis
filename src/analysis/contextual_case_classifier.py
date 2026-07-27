from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any

from src.filtering.article_classifier import (
    find_matches,
    find_matches_by_window,
    group_has_prefix,
    has_male_victim_pattern,
    normalize_text,
    prepare_lexicon,
    terms_for_prefix,
)


FEMALE_GROUPS = [
    "female_reference_terms.strong",
    "female_reference_terms.gendered_phrases",
]

FEMALE_ROLE_GROUPS = [
    "female_reference_terms.roles",
]

VIOLENT_DEATH_GROUPS = [
    "indirect_terms.victim_death_phrases.violent_or_suspicious",
]

AMBIGUOUS_DEATH_GROUPS = [
    "indirect_terms.victim_death_phrases.ambiguous_or_accidental",
]

DISAPPEARANCE_GROUPS = [
    "indirect_terms.disappearance_phrases",
]

INJURY_OR_METHOD_GROUPS = [
    "indirect_terms.injury_or_method_terms",
]

GENERIC_CRIME_GROUPS = [
    "indirect_terms.generic_crime_terms",
]

GENERAL_VIOLENCE_ACT_GROUPS = [
    "violence_types_and_modalities.acts",
]

SEXUAL_VIOLENCE_GROUPS = [
    "sexual_violence_terms",
    "violence_types_and_modalities.sexual",
]

INTIMATE_PARTNER_GROUPS = [
    "relationship_terms.intimate_partner",
]

FAMILY_OR_CLOSE_GROUPS = [
    "relationship_terms.family_or_close_relation",
]

MALE_AGGRESSOR_GROUPS = [
    "aggressor_terms.male_aggressor",
]

PRIOR_VIOLENCE_GROUPS = [
    "prior_violence_or_control_terms",
]

ACCIDENT_GROUPS = [
    "accident_or_disaster_terms.strong",
]

DIRECT_GROUPS = [
    "direct_terms.high_precision",
]


EXPLICIT_GENDER_LABEL_PHRASES = {
    "violencia de genero",
    "violencia basada en genero",
    "violencia por razones de genero",
    "violencia machista",
    "violencia contra la mujer",
    "violencia contra las mujeres",
    "violencia hacia la mujer",
    "violencia hacia las mujeres",
    "violencia feminicida",
    "crimen de genero",
    "asesinato por razones de genero",
    "mujeres asesinadas por razones de genero",
}

GENDERED_DESCRIPTION_PHRASES = {
    "muerte violenta de una mujer",
    "muerte violenta de mujeres",
    "asesinatos de mujeres",
}

LEVEL_RANK = {
    "none": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
}


FEMALE_NOUN = (
    r"(?:mujer|joven|adolescente|nina|menor|victima|"
    r"esposa|novia|exnovia|cooperante)"
)

MALE_RELATION_OR_PERSON = (
    r"(?:hombre|agresor|sospechoso|acusado|detenido|"
    r"esposo|marido|novio|exnovio|pareja|expareja|"
    r"padre|padrastro|hermano|vecino|hijo)"
)

VIOLENT_VERB = (
    r"(?:asesino|mato|golpeo|apu[nn]alo|estrangulo|"
    r"baleo|secuestro|amenazo|drogo|enveneno|descuartizo)"
)

SEXUAL_VERB = (
    r"(?:violo|abuso|agredio\s+sexualmente|"
    r"sometio\s+sexualmente)"
)


VIOLENT_EVENT_TEXT_PATTERNS = [
    r"\b(?:fue\s+)?asesinad[ao]s?\b",
    r"\basesinaron\b",
    r"\bla\s+(?:asesinaron|mataron)\b",
    r"\b(?:fue\s+)?hallad[ao]s?\s+(?:muert[ao]s?|sin vida)\b",
    r"\b(?:fue\s+)?encontrad[ao]s?\s+(?:muert[ao]s?|sin vida)\b",
    r"\baparecio\s+muert[ao]s?\b",
    r"\bestrangulad[ao]s?\b",
    r"\bapun?alad[ao]s?\b",
    r"\bbalead[ao]s?\b",
    r"\bheridas?\s+de\s+arma\s+blanca\b",
    r"\bsignos?\s+de\s+violencia(?:\s+sexual)?\b",
    r"\babuso\s+sexual\s+seguido\s+de\s+muerte\b",
]

AMBIGUOUS_DEATH_TEXT_PATTERNS = [
    r"\bmurio\b",
    r"\bfallecio\b",
    r"\bmuert[ao]s?\b",
    r"\bsin vida\b",
]

DISAPPEARANCE_TEXT_PATTERNS = [
    r"\bdesaparecid[ao]s?\b",
    r"\bhabia desaparecido\b",
    r"\bestaba desaparecid[ao]\b",
    r"\bdenuncia de desaparicion\b",
]

SEXUAL_VIOLENCE_TEXT_PATTERNS = [
    r"\bviolacion\b",
    r"\bviolaciones\b",
    r"\babuso sexual\b",
    r"\bagresion sexual\b",
    r"\besclava sexual\b",
    r"\besclavas sexuales\b",
    r"\bexplotacion sexual\b",
    r"\btrata de mujeres\b",
    r"\bviolencia sexual\b",
    r"\bviolo\b",
    r"\babus[oa]\s+de\b",
    # Sintaxis de alegación. Esto evita que los usos genéricos de 'abuso'
    # se consideren violencia sexual salvo que exista una acusación o denuncia explícita.
    r"\b(?:acusa|denuncia)\s+(?:de\s+)?"
    r"(?:abuso|violacion|agresion sexual)\s+a(?:l)?\b",
    r"\b(?:denuncio|acuso)\b.{0,65}\b"
    r"(?:abuso|violacion|agresion sexual)\b",
]

# Construcciones claras en las que la víctima es una mujer.
FEMALE_VICTIM_DIRECTION_PATTERNS = [
    rf"\b(?:la|una|esta)\s+{FEMALE_NOUN}\b"
    rf".{{0,55}}\b(?:fue|habia sido|resulto)?\s*"
    rf"(?:asesinada|muerta|golpeada|apun?alada|estrangulada|"
    rf"baleada|secuestrada|violada|abusada)\b",

    rf"\b(?:el|un|su)\s+{MALE_RELATION_OR_PERSON}\b"
    rf".{{0,90}}\b(?:la\s+)?"
    rf"(?:asesino|mato|golpeo|apun?alo|estrangulo|baleo|"
    rf"secuestro|amenazo|violo)\b",

    rf"\b(?:asesino|mato|golpeo|apun?alo|estrangulo|baleo|"
    rf"secuestro|amenazo|violo)\b"
    rf".{{0,70}}\b(?:a\s+)?(?:la|una|su)\s+{FEMALE_NOUN}\b",

    r"\b(?:la|ella)\s+(?:asesino|mato|golpeo|violo|"
    r"apun?alo|estrangulo|secuestro|amenazo)\b",

    r"\babuso\s+de\s+ella\b",
]

# Dirección inversa clara: mujer como agresora y hombre como víctima.
FEMALE_AGGRESSOR_MALE_VICTIM_PATTERNS = [
    rf"\b(?:la|una)\s+(?:mujer|esposa|novia|exnovia|expareja)\b"
    rf".{{0,150}}\b{VIOLENT_VERB}\b"
    rf".{{0,60}}\b(?:a\s+)?(?:su\s+)?"
    rf"(?:esposo|marido|novio|exnovio|pareja|hombre)\b",

    # Formas verbales perfectas o compuestas frecuentes en textos periodísticos:
    # 'la mujer habia drogado a su esposo',
    # 'la esposa lo habia apunalado'.
    r"\b(?:la|una)\s+(?:mujer|esposa|novia|exnovia|expareja)\b"
    r".{0,120}\b(?:habia|ha|haya|habria|fue)?\s*"
    r"(?:drogado|apunalado|asesinado|matado|golpeado|estrangulado|"
    r"baleado|envenenado|descuartizado)\b"
    r".{0,80}\b(?:a\s+)?(?:su\s+)?"
    r"(?:esposo|marido|novio|exnovio|pareja|hombre)\b",

    rf"\b(?:ella|la mujer|la esposa|la novia)\b"
    rf".{{0,150}}\b(?:lo\s+)?{VIOLENT_VERB}\b",

    r"\b(?:ella|la mujer|la esposa|la novia|la expareja)\b"
    r".{0,150}\b(?:lo\s+)?(?:habia|ha|haya|habria)?\s*"
    r"(?:drogado|apunalado|asesinado|matado|golpeado|estrangulado|"
    r"baleado|envenenado|descuartizado)\b",

    rf"\b(?:descuartizo|asesino|mato|apun?alo|"
    rf"enveneno|drogo)\s+a\s+su\s+"
    rf"(?:esposo|marido|novio|pareja)\b",
]

# Violencia sexual directa contra una víctima mujer.
DIRECT_SEXUAL_FEMALE_VICTIM_PATTERNS = [
    rf"\b{SEXUAL_VERB}\b"
    rf".{{0,90}}\b(?:a|de)\s+(?:la|una)\s+{FEMALE_NOUN}\b",

    rf"\b(?:la|una)\s+{FEMALE_NOUN}\b"
    rf".{{0,60}}\b(?:fue\s+)?"
    rf"(?:violada|abusada|agredida sexualmente)\b",

    r"\babuso\s+de\s+ella\b",
    r"\bla\s+violo\b",
    r"\bla\s+convirtio\s+en\s+esclava sexual\b",
    r"\bla\s+mantuvo\s+cautiva\b.{0,90}\b(?:violo|abuso)\b",
]

# Alegación probable: útil para revisión, pero menos segura que la sintaxis directa.
PROBABLE_SEXUAL_FEMALE_VICTIM_PATTERNS = [
    rf"\b(?:la\s+)?{FEMALE_NOUN}\s+que\s+"
    rf"(?:acusa|denuncia)\s+(?:de\s+)?"
    rf"(?:abuso|violacion|agresion sexual)\s+a(?:l)?\b",

    rf"\b(?:la|una)\s+{FEMALE_NOUN}\b"
    rf".{{0,75}}\b(?:denuncio|acuso)\b"
    rf".{{0,65}}\b(?:abuso|violacion|agresion sexual)\b",
]


def _matches_any_text_pattern(
    text: str,
    patterns: list[str],
) -> bool:
    normalized = normalize_text(text)
    return any(
        re.search(pattern, normalized)
        for pattern in patterns
    )


def deep_merge(base: dict, extra: dict) -> dict:
    result = deepcopy(base)

    for key, value in extra.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)

    return result


def build_prepared_contextual_lexicon(
    retrieval_lexicon: dict,
    contextual_lexicon: dict,
) -> dict:
    merged = deep_merge(retrieval_lexicon, contextual_lexicon)
    return prepare_lexicon(merged)


def safe_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def build_analysis_text(row: dict) -> str:
    title = (
        row.get("title_clean")
        or row.get("title")
        or row.get("title_raw")
        or ""
    )
    body = (
        row.get("text_clean")
        or row.get("text")
        or row.get("text_raw")
        or ""
    )
    anchor = row.get("anchor_text") or ""

    return "\n".join(
        [
            str(anchor),
            str(title),
            str(body)[:2500],
        ]
    )


def detect_explicit_label(
    matches_global: dict[str, list[str]],
) -> tuple[str, list[str]]:
    direct_terms = terms_for_prefix(matches_global, DIRECT_GROUPS)
    normalized_terms = {
        normalize_text(term): term
        for term in direct_terms
    }

    femicide_terms = [
        original
        for normalized, original in normalized_terms.items()
        if "femicid" in normalized or "feminicid" in normalized
    ]
    if femicide_terms:
        return "femicide_feminicide", sorted(set(femicide_terms))

    gender_label_terms = [
        original
        for normalized, original in normalized_terms.items()
        if normalized in EXPLICIT_GENDER_LABEL_PHRASES
    ]
    if gender_label_terms:
        return "gender_violence", sorted(set(gender_label_terms))

    description_terms = [
        original
        for normalized, original in normalized_terms.items()
        if normalized in GENDERED_DESCRIPTION_PHRASES
    ]
    if description_terms:
        return "gendered_description", sorted(set(description_terms))

    return "none", []


def _has(matches: dict[str, list[str]], groups: list[str]) -> bool:
    return group_has_prefix(matches, groups)


def _upgrade_level(current: str, proposed: str) -> str:
    if LEVEL_RANK[proposed] > LEVEL_RANK[current]:
        return proposed
    return current


def detect_violence_direction(
    window_text: str,
) -> tuple[str, list[str]]:
    reverse = _matches_any_text_pattern(
        window_text,
        FEMALE_AGGRESSOR_MALE_VICTIM_PATTERNS,
    )
    direct_female_victim = _matches_any_text_pattern(
        window_text,
        FEMALE_VICTIM_DIRECTION_PATTERNS,
    )
    direct_sexual = _matches_any_text_pattern(
        window_text,
        DIRECT_SEXUAL_FEMALE_VICTIM_PATTERNS,
    )
    probable_sexual = _matches_any_text_pattern(
        window_text,
        PROBABLE_SEXUAL_FEMALE_VICTIM_PATTERNS,
    )

    if reverse and (direct_female_victim or direct_sexual):
        return "mixed_or_conflicting", [
            "reverse_direction_pattern",
            "female_victim_pattern",
        ]

    if reverse:
        return "female_to_male", [
            "female_aggressor_male_victim_pattern"
        ]

    if direct_sexual:
        return "male_to_female", [
            "direct_sexual_violence_against_female"
        ]

    if direct_female_victim:
        return "female_victim_explicit", [
            "female_victim_grammatical_pattern"
        ]

    if probable_sexual:
        return "probable_male_to_female", [
            "female_sexual_violence_allegation_pattern"
        ]

    return "unknown", []


def detect_window_violence_types(
    *,
    has_violent_death: bool,
    has_disappearance: bool,
    has_injury: bool,
    has_prior_violence: bool,
    has_sexual_violence: bool,
) -> list[str]:
    types: list[str] = []

    if has_violent_death:
        types.append("lethal_or_suspicious_death")
    if has_sexual_violence:
        types.append("sexual_violence")
    if has_disappearance:
        types.append("disappearance")
    if has_injury:
        types.append("physical_violence_or_method")
    if has_prior_violence:
        types.append("prior_violence_or_control")

    return sorted(set(types))


def _article_value_from_decisive_evidence(
    evidence: list[dict[str, Any]],
    field: str,
    empty_value: str,
) -> str:
    values = {
        str(item.get(field))
        for item in evidence
        if item.get(field)
        and str(item.get(field)) != empty_value
    }

    if not values:
        return empty_value

    if len(values) == 1:
        return next(iter(values))

    return "mixed"


def classify_contextual_windows(
    matches_by_window: list[dict[str, Any]],
    max_evidence: int = 10,
) -> dict[str, Any]:
    """
    Clasificación contextual con tres salvaguardas:

    1. La dirección importa: no se promueven casos de mujer agresora y hombre víctima.
    2. Los términos familiares no promueven un artículo por sí solos.
    3. La violencia sexual clara contra una víctima mujer puede ser relevante incluso
       sin una relación de pareja íntima.
    """
    best_level = "none"
    all_evidence: list[dict[str, Any]] = []

    for item in matches_by_window:
        matches = item.get("matches", {})
        window_text = item.get("window_text", "")

        has_female = _has(matches, FEMALE_GROUPS)
        has_female_role = _has(matches, FEMALE_ROLE_GROUPS)
        has_violent_death = _has(matches, VIOLENT_DEATH_GROUPS)
        has_ambiguous_death = _has(matches, AMBIGUOUS_DEATH_GROUPS)
        has_disappearance = _has(matches, DISAPPEARANCE_GROUPS)
        has_injury = _has(matches, INJURY_OR_METHOD_GROUPS)
        has_generic_crime = _has(matches, GENERIC_CRIME_GROUPS)
        has_general_violence = _has(
            matches,
            GENERAL_VIOLENCE_ACT_GROUPS,
        )
        has_sexual_violence = (
            _has(matches, SEXUAL_VIOLENCE_GROUPS)
            or _matches_any_text_pattern(
                window_text,
                SEXUAL_VIOLENCE_TEXT_PATTERNS,
            )
        )
        has_intimate_partner = _has(
            matches,
            INTIMATE_PARTNER_GROUPS,
        )
        has_family_relation = _has(
            matches,
            FAMILY_OR_CLOSE_GROUPS,
        )
        has_male_aggressor = _has(
            matches,
            MALE_AGGRESSOR_GROUPS,
        )
        has_prior_violence = _has(
            matches,
            PRIOR_VIOLENCE_GROUPS,
        )
        has_accident = _has(matches, ACCIDENT_GROUPS)

        has_violent_event_text = _matches_any_text_pattern(
            window_text,
            VIOLENT_EVENT_TEXT_PATTERNS,
        )
        has_ambiguous_death_text = _matches_any_text_pattern(
            window_text,
            AMBIGUOUS_DEATH_TEXT_PATTERNS,
        )
        has_disappearance_text = _matches_any_text_pattern(
            window_text,
            DISAPPEARANCE_TEXT_PATTERNS,
        )

        has_violent_death = (
            has_violent_death
            or has_violent_event_text
        )
        has_ambiguous_death = (
            has_ambiguous_death
            or (
                has_ambiguous_death_text
                and not has_violent_event_text
            )
        )
        has_disappearance = (
            has_disappearance
            or has_disappearance_text
        )

        direction, direction_reasons = (
            detect_violence_direction(window_text)
        )

        # No se descarta un posible artículo mixto solo porque aparezca "hombre".
        # El detector explícito de dirección inversa aporta más información.
        if (
            has_male_victim_pattern(window_text)
            and direction == "unknown"
        ):
            direction = "possible_male_victim"
            direction_reasons = [
                "male_victim_pattern_without_clear_direction"
            ]

        strong_event = (
            has_violent_death
            or has_injury
            or has_general_violence
            or has_sexual_violence
        )

        effective_female = has_female or (
            has_female_role
            and (
                strong_event
                or has_disappearance
                or has_prior_violence
            )
        )

        if not effective_female:
            continue

        violence_types = detect_window_violence_types(
            has_violent_death=has_violent_death,
            has_disappearance=has_disappearance,
            has_injury=has_injury,
            has_prior_violence=has_prior_violence,
            has_sexual_violence=has_sexual_violence,
        )

        window_level = "none"
        window_reasons: list[str] = []

        # La dirección inversa explícita constituye una señal de exclusión.
        if direction == "female_to_male":
            window_reasons.append(
                "reverse_direction_female_aggressor_male_victim"
            )

        # Violencia sexual clara contra una víctima mujer.
        elif (
            has_sexual_violence
            and direction in {
                "male_to_female",
                "female_victim_explicit",
            }
        ):
            window_level = "high"
            window_reasons.append(
                "sexual_violence_against_female_with_clear_direction"
            )

        # La sintaxis de alegación es relevante, pero se conserva para revisión.
        elif (
            has_sexual_violence
            and direction == "probable_male_to_female"
        ):
            window_level = "medium"
            window_reasons.append(
                "sexual_violence_against_female_probable_direction"
            )

        # La relación de pareja íntima o violencia previa solo se consideran altas cuando
        # la mujer es gramaticalmente la víctima y no solo cuando es mencionada.
        elif (
            (strong_event or has_disappearance)
            and direction in {
                "male_to_female",
                "female_victim_explicit",
            }
            and (
                has_intimate_partner
                or has_prior_violence
            )
        ):
            window_level = "high"

            if has_intimate_partner:
                window_reasons.append(
                    "female_victim_event_plus_intimate_partner"
                )
            if has_prior_violence:
                window_reasons.append(
                    "female_victim_event_plus_prior_violence_or_control"
                )

        # Las denuncias previas o el control pueden vincular dos oraciones, incluso
        # cuando el agresor no se menciona explícitamente.
        elif (
            (strong_event or has_disappearance)
            and has_prior_violence
            and direction not in {
                "female_to_male",
                "possible_male_victim",
            }
        ):
            window_level = "high"
            window_reasons.append(
                "female_event_plus_prior_violence_or_control"
            )

        # Un término de agresor masculino solo funciona como evidencia de apoyo
        # cuando la sintaxis también identifica a la mujer como víctima.
        elif (
            strong_event
            and has_male_aggressor
            and direction in {
                "male_to_female",
                "female_victim_explicit",
                "probable_male_to_female",
            }
        ):
            window_level = "medium"
            window_reasons.append(
                "female_victim_event_plus_supported_male_aggressor"
            )

        elif (
            has_generic_crime
            and has_intimate_partner
            and direction in {
                "male_to_female",
                "female_victim_explicit",
            }
        ):
            window_level = "medium"
            window_reasons.append(
                "female_victim_generic_crime_plus_intimate_partner"
            )

        elif (
            has_disappearance
            and (
                has_intimate_partner
                or has_prior_violence
            )
            and direction != "female_to_male"
        ):
            window_level = "medium"
            window_reasons.append(
                "female_disappearance_plus_contextual_relation"
            )

        # Deliberadamente los términos de relación familiar o cercana NO son suficientes.
        elif (
            strong_event
            or has_disappearance
            or has_ambiguous_death
        ):
            window_level = "low"
            window_reasons.append(
                "female_event_without_sufficient_gender_context"
            )

            if has_intimate_partner:
                window_reasons.append(
                    "intimate_partner_term_without_clear_direction"
                )
            if has_family_relation:
                window_reasons.append(
                    "family_or_close_relation_not_used_as_decisive_evidence"
                )
            if has_male_aggressor:
                window_reasons.append(
                    "male_aggressor_term_without_clear_direction"
                )

        if has_accident and window_level != "none":
            window_reasons.append(
                "possible_accident_or_disaster_context"
            )

            if not has_prior_violence:
                window_level = "low"

        evidence_item = {
            "level": window_level,
            "reasons": sorted(set(window_reasons)),
            "direction": direction,
            "direction_reasons": direction_reasons,
            "violence_types": violence_types,
            "window_text": window_text[:500],
            "matches": matches,
        }
        all_evidence.append(evidence_item)

        best_level = _upgrade_level(
            best_level,
            window_level,
        )

    all_evidence.sort(
        key=lambda item: LEVEL_RANK[item["level"]],
        reverse=True,
    )

    decisive_evidence = [
        item
        for item in all_evidence
        if item["level"] == best_level
    ]

    # Cuando no existe un nivel positivo, se conserva la evidencia de exclusión para auditoría.
    if best_level == "none":
        decisive_evidence = [
            item
            for item in all_evidence
            if item["reasons"]
        ]

    decisive_reasons = sorted(
        {
            reason
            for item in decisive_evidence
            for reason in item["reasons"]
        }
    )

    all_reasons = sorted(
        {
            reason
            for item in all_evidence
            for reason in item["reasons"]
        }
    )

    decisive_types = sorted(
        {
            violence_type
            for item in decisive_evidence
            for violence_type in item["violence_types"]
        }
    )

    article_direction = _article_value_from_decisive_evidence(
        decisive_evidence,
        field="direction",
        empty_value="unknown",
    )

    return {
        "level": best_level,
        "decisive_reasons": decisive_reasons,
        "all_reasons": all_reasons,
        "evidence": all_evidence[:max_evidence],
        "direction": article_direction,
        "violence_types": decisive_types,
    }


def derive_outputs(
    retrieval_bucket: str,
    explicit_label_type: str,
    contextual_level: str,
) -> dict[str, Any]:
    is_retrieval_case = str(retrieval_bucket).startswith("case_")
    has_explicit_gender_label = explicit_label_type in {
        "femicide_feminicide",
        "gender_violence",
    }

    if explicit_label_type == "femicide_feminicide":
        recognition_mode = "explicit_femicide_feminicide"
    elif explicit_label_type == "gender_violence":
        recognition_mode = "explicit_gender_violence"
    elif contextual_level in {"high", "medium"}:
        recognition_mode = "contextual_without_explicit_label"
    elif explicit_label_type == "gendered_description":
        recognition_mode = "gendered_description_without_explicit_label"
    elif is_retrieval_case:
        recognition_mode = "ambiguous_case_without_explicit_label"
    else:
        recognition_mode = "no_case_evidence"

    if has_explicit_gender_label and (
        is_retrieval_case or contextual_level in {"high", "medium"}
    ):
        provisional_status = "candidate_explicit_case"
        review_priority = "P0_explicit_sample_only"

    elif has_explicit_gender_label:
        provisional_status = "topic_explicit_not_case"
        review_priority = "P4_not_selected"

    elif contextual_level == "high":
        provisional_status = "candidate_contextual_high"
        review_priority = "P1_contextual_high"

    elif contextual_level == "medium":
        provisional_status = "review_contextual_medium"
        review_priority = "P2_contextual_medium"

    elif is_retrieval_case:
        provisional_status = "review_ambiguous_case"
        review_priority = "P3_ambiguous_case"

    else:
        provisional_status = "not_selected"
        review_priority = "P4_not_selected"

    return {
        "has_explicit_gender_label": has_explicit_gender_label,
        "gender_recognition_mode": recognition_mode,
        "provisional_case_status": provisional_status,
        "review_priority": review_priority,
        "is_contextual_without_label": (
            not has_explicit_gender_label
            and contextual_level in {"high", "medium"}
        ),
        "analysis_case_candidate": provisional_status in {
            "candidate_explicit_case",
            "candidate_contextual_high",
            "review_contextual_medium",
            "review_ambiguous_case",
        },
    }


def classify_contextual_case(
    row: dict,
    prepared_lexicon: dict,
) -> dict[str, Any]:
    analysis_text = build_analysis_text(row)

    matches_global = find_matches(
        analysis_text,
        prepared_lexicon,
    )
    matches_by_window = find_matches_by_window(
        analysis_text,
        prepared_lexicon,
    )

    explicit_label_type, explicit_terms = detect_explicit_label(
        matches_global
    )

    contextual = classify_contextual_windows(
        matches_by_window
    )

    contextual_level = contextual["level"]

    derived = derive_outputs(
        retrieval_bucket=str(row.get("retrieval_bucket", "")),
        explicit_label_type=explicit_label_type,
        contextual_level=contextual_level,
    )

    return {
        "explicit_label_type": explicit_label_type,
        "explicit_label_terms_json": safe_json_dumps(explicit_terms),
        "contextual_evidence_level": contextual_level,
        "contextual_score": LEVEL_RANK[contextual_level],
        # Solo los motivos que determinaron el nivel final del artículo.
        "contextual_reasons_json": safe_json_dumps(
            contextual["decisive_reasons"]
        ),
        # Todos los motivos permanecen disponibles para depuración.
        "contextual_all_reasons_json": safe_json_dumps(
            contextual["all_reasons"]
        ),
        "contextual_evidence_json": safe_json_dumps(
            contextual["evidence"]
        ),
        "violence_direction": contextual["direction"],
        "contextual_violence_types_json": safe_json_dumps(
            contextual["violence_types"]
        ),
        **derived,
    }