from src.analysis.contextual_case_classifier import (
    build_prepared_contextual_lexicon,
    classify_contextual_case,
)


BASE_LEXICON = {
    "female_reference_terms": {
        "strong": ["mujer", "mujeres"],
        "gendered_phrases": ["la joven", "una joven"],
        "roles": [
            "esposa",
            "novia",
            "madre",
            "cooperante",
        ],
    },
    "direct_terms": {
        "high_precision": [
            "femicidio",
            "feminicidio",
            "violencia de género",
            "muerte violenta de una mujer",
        ]
    },
    "indirect_terms": {
        "victim_death_phrases": {
            "violent_or_suspicious": [
                "mujer asesinada",
                "joven hallada muerta",
            ],
            "ambiguous_or_accidental": [
                "mujer muerta",
            ],
        },
        "disappearance_phrases": [
            "mujer desaparecida",
        ],
        "injury_or_method_terms": [
            "apuñalada",
            "arma blanca",
            "estrangulada",
        ],
        "generic_crime_terms": [
            "asesinato",
            "homicidio",
            "detenido",
        ],
    },
    "violence_types_and_modalities": {
        "acts": [
            "amenazas",
            "golpeada",
            "abuso",
        ]
    },
    "sexual_violence_terms": [
        "violó",
        "violación",
        "abuso sexual",
        "esclava sexual",
    ],
    "relationship_terms": {
        "intimate_partner": [
            "pareja",
            "expareja",
            "esposo",
            "marido",
        ],
        "family_or_close_relation": [
            "padre",
            "hijo",
            "vecino",
        ],
    },
    "aggressor_terms": {
        "male_aggressor": [
            "agresor",
            "sospechoso",
            "detenido",
            "hombre",
        ]
    },
    "accident_or_disaster_terms": {
        "strong": [
            "accidente",
            "choque",
            "bombardeo",
        ]
    },
}

CONTEXTUAL_LEXICON = {
    "prior_violence_or_control_terms": {
        "previous_reports": [
            "lo había denunciado",
            "antecedentes de violencia",
        ],
        "separation_or_rejection": [
            "no aceptaba la separación",
        ],
    }
}


def prepared_lexicon():
    return build_prepared_contextual_lexicon(
        BASE_LEXICON,
        CONTEXTUAL_LEXICON,
    )


def test_explicit_feminicide_case():
    row = {
        "title": "Investigan un feminicidio",
        "text": "La mujer fue asesinada.",
        "retrieval_bucket": "case_strong",
    }

    result = classify_contextual_case(
        row,
        prepared_lexicon(),
    )

    assert result["explicit_label_type"] == "femicide_feminicide"
    assert result["has_explicit_gender_label"] is True
    assert result["provisional_case_status"] == "candidate_explicit_case"


def test_contextual_high_without_explicit_label():
    row = {
        "title": "Detuvieron a la expareja",
        "text": (
            "La mujer fue asesinada por su expareja. "
            "La víctima lo había denunciado."
        ),
        "retrieval_bucket": "case_strong",
    }

    result = classify_contextual_case(
        row,
        prepared_lexicon(),
    )

    assert result["has_explicit_gender_label"] is False
    assert result["contextual_evidence_level"] == "high"
    assert (
        result["gender_recognition_mode"]
        == "contextual_without_explicit_label"
    )
    assert (
        result["provisional_case_status"]
        == "candidate_contextual_high"
    )


def test_accident_is_not_promoted_to_contextual_high():
    row = {
        "title": "Una mujer murió en un accidente",
        "text": "La mujer murió durante un choque en la ruta.",
        "retrieval_bucket": "case_review_female_death",
    }

    result = classify_contextual_case(
        row,
        prepared_lexicon(),
    )

    assert result["contextual_evidence_level"] in {"none", "low"}
    assert result["provisional_case_status"] == "review_ambiguous_case"


def test_ambiguous_death_remains_for_review():
    row = {
        "title": "Una joven fue hallada muerta",
        "text": "La joven fue hallada muerta en un terreno.",
        "retrieval_bucket": "case_strong",
    }

    result = classify_contextual_case(
        row,
        prepared_lexicon(),
    )

    assert result["has_explicit_gender_label"] is False
    assert result["contextual_evidence_level"] == "low"
    assert result["provisional_case_status"] == "review_ambiguous_case"


def test_female_aggressor_male_victim_is_not_promoted():
    row = {
        "title": "Descuartizó a su marido",
        "text": (
            "La mujer había drogado a su esposo "
            "y lo había apuñalado con un arma blanca."
        ),
        "retrieval_bucket": "case_review_possible_violence",
    }

    result = classify_contextual_case(
        row,
        prepared_lexicon(),
    )

    assert result["violence_direction"] == "female_to_male"
    assert result["contextual_evidence_level"] == "none"
    assert result["is_contextual_without_label"] is False


def test_family_term_alone_is_not_decisive():
    row = {
        "title": "Declaraciones sobre un escándalo",
        "text": (
            "La mujer habló de un abuso ocurrido años atrás. "
            "El hijo de la reina asistió a otra reunión."
        ),
        "retrieval_bucket": "not_candidate",
    }

    result = classify_contextual_case(
        row,
        prepared_lexicon(),
    )

    assert result["contextual_evidence_level"] in {"none", "low"}
    assert result["provisional_case_status"] == "not_selected"


def test_sexual_violence_allegation_is_contextual_medium():
    row = {
        "title": "Una mujer acusa de abuso al hijo de la reina",
        "text": (
            "La mujer que acusa de abuso al hijo de la reina "
            "presentó una denuncia ante la justicia."
        ),
        "retrieval_bucket": "not_candidate",
    }

    result = classify_contextual_case(
        row,
        prepared_lexicon(),
    )

    assert result["contextual_evidence_level"] == "medium"
    assert result["violence_direction"] == "probable_male_to_female"
    assert result["provisional_case_status"] == "review_contextual_medium"


def test_direct_sexual_violence_against_woman_is_high():
    row = {
        "title": "El calvario de una cooperante",
        "text": (
            "El líder violó en varias ocasiones a la cooperante. "
            "Después la mantuvo cautiva como esclava sexual."
        ),
        "retrieval_bucket": "not_candidate",
    }

    result = classify_contextual_case(
        row,
        prepared_lexicon(),
    )

    assert result["contextual_evidence_level"] == "high"
    assert result["violence_direction"] == "male_to_female"
    assert result["provisional_case_status"] == "candidate_contextual_high"
