import pandas as pd

from src.analysis.build_final_case_corpus import (
    classify_scope_alert,
)


def test_explicit_femicide_male_aggressor_has_no_scope_alert():
    row = pd.Series(
        {
            "title": "Investigan un femicidio",
            "text": (
                "El hombre mató a su esposa. "
                "La Justicia investiga el hecho como femicidio."
            ),
            "explicit_label_type": "femicide_feminicide",
            "violence_direction": "female_victim_explicit",
        }
    )

    result = classify_scope_alert(row)

    assert result["auto_scope_status"] == "no_alert"


def test_male_kills_partner_is_not_female_aggressor_alert():
    row = pd.Series(
        {
            "title": "Femicidio: mató a su pareja",
            "text": (
                "El acusado mató a su pareja y fue detenido."
            ),
            "explicit_label_type": "femicide_feminicide",
            "violence_direction": "unknown",
        }
    )

    result = classify_scope_alert(row)

    assert result["auto_scope_status"] == "no_alert"


def test_female_aggressor_still_requires_review():
    row = pd.Series(
        {
            "title": "Una mujer mató a su marido",
            "text": (
                "La mujer mató a su marido durante una pelea."
            ),
            "explicit_label_type": "none",
            "violence_direction": "female_to_male",
        }
    )

    result = classify_scope_alert(row)

    assert result["auto_scope_status"] == "alert"
    assert (
        result["auto_scope_reason"]
        == "female_aggressor_male_victim"
    )


def test_possible_self_defense_still_requires_review():
    row = pd.Series(
        {
            "title": "Una mujer mató a su esposo",
            "text": (
                "La mujer mató a su esposo cuando él la atacaba. "
                "Declaró que actuó para defenderse."
            ),
            "explicit_label_type": "none",
            "violence_direction": "female_to_male",
        }
    )

    result = classify_scope_alert(row)

    assert result["auto_scope_status"] == "alert"