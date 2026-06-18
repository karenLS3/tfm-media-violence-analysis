from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import yaml


# Only these YAML roots are used as retrieval terms.
# metadata, thresholds and classification_rules are intentionally ignored.
RETRIEVAL_ROOTS = {
    "female_reference_terms",
    "direct_terms",
    "violence_types_and_modalities",
    "indirect_terms",
    "relationship_terms",
    "aggressor_terms",
    "prior_violence_or_control_terms",
    "structural_context_terms",
    "help_and_prevention_terms",
    "sexual_violence_terms",
    "digital_or_cyber_violence_terms",
    "accident_or_disaster_terms",
}


def normalize_text(text: str) -> str:
    """
    Normalize text for robust matching:
    - lowercase
    - remove accents
    - normalize spaces

    This means that a lexicon term such as "violencia de género"
    also matches "violencia de genero".
    """
    text = text or ""
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_lexicon(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def flatten_lexicon(d: dict, prefix: str = "") -> dict[str, list[str]]:
    """
    Flatten nested YAML groups.

    Example:
        female_reference_terms:
          strong:
            - mujer

    becomes:
        female_reference_terms.strong -> ["mujer"]

    Only roots listed in RETRIEVAL_ROOTS are used.
    """
    flat: dict[str, list[str]] = {}

    for key, value in d.items():
        if not prefix and key not in RETRIEVAL_ROOTS:
            continue

        new_key = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict):
            flat.update(flatten_lexicon(value, new_key))

        elif isinstance(value, list):
            terms = []
            seen_normalized = set()

            for item in value:
                if not isinstance(item, str):
                    continue

                norm = normalize_text(item)

                if not norm:
                    continue

                # Avoid duplicate variants such as "violación" and "violacion".
                if norm in seen_normalized:
                    continue

                seen_normalized.add(norm)
                terms.append(item)

            if terms:
                flat[new_key] = terms

    return flat


def compile_term_pattern(term: str) -> re.Pattern:
    """
    Compile a lexicon term as a regex pattern.

    The pattern is accent-insensitive because both the term and the text
    are normalized before matching.
    """
    norm = normalize_text(term)
    escaped = re.escape(norm)
    escaped = escaped.replace(r"\ ", r"\s+")

    return re.compile(rf"(?<!\w){escaped}(?!\w)", flags=re.IGNORECASE)


def prepare_lexicon(lexicon: dict) -> dict[str, list[tuple[str, re.Pattern]]]:
    """
    Precompile all lexicon patterns once.
    This makes classification faster.
    """
    flat = flatten_lexicon(lexicon)
    prepared: dict[str, list[tuple[str, re.Pattern]]] = {}

    for group, terms in flat.items():
        prepared[group] = [(term, compile_term_pattern(term)) for term in terms]

    return prepared


def is_boilerplate_window(text: str) -> bool:
    """
    Detect windows that probably come from navigation,
    rankings, menus, sidebars or footer content.

    This is important for archived newspaper pages because Wayback HTML
    often contains "most read", menus, related links, and other article titles.
    """
    norm = normalize_text(text)

    boilerplate_patterns = [
        "tops clarin",
        "leidas comentadas",
        "mi cuenta",
        "cerrar sesion",
        "buscar clarin",
        "noticias deportes extrashow",
        "servicios",
        "newsletter",
        "mas leidas",
        "ultimas noticias",
        "lo mas leido",
        "mas comentadas",
        "relacionadas",
        "notas relacionadas",
    ]

    return any(pattern in norm for pattern in boilerplate_patterns)


def find_matches(
    text: str,
    prepared_lexicon: dict[str, list[tuple[str, re.Pattern]]],
) -> dict[str, list[str]]:
    """
    Find all lexicon matches in a text, grouped by lexicon category.
    """
    norm_text = normalize_text(text)
    matches: dict[str, list[str]] = {}

    for group, term_patterns in prepared_lexicon.items():
        found = []

        for original_term, pattern in term_patterns:
            if pattern.search(norm_text):
                found.append(original_term)

        if found:
            matches[group] = sorted(set(found))

    return matches


def split_sentences(text: str) -> list[str]:
    """
    Simple sentence splitter.

    This is intentionally simple. The goal is not perfect linguistic parsing,
    but avoiding global co-occurrence false positives.
    """
    text = text or ""
    parts = re.split(r"(?<=[\.\?\!])\s+|\n+", text)
    return [p.strip() for p in parts if p and p.strip()]


def build_sentence_windows(sentences: list[str], window_size: int = 2) -> list[str]:
    """
    Build windows of one and two consecutive sentences.

    This captures cases such as:
        "La mujer había denunciado a su expareja.
         Al día siguiente apareció muerta."
    """
    windows = []

    for i, sentence in enumerate(sentences):
        windows.append(sentence)

        if window_size >= 2 and i + 1 < len(sentences):
            windows.append(sentence + " " + sentences[i + 1])

    return windows


def find_matches_by_window(
    text: str,
    prepared_lexicon: dict[str, list[tuple[str, re.Pattern]]],
) -> list[dict[str, Any]]:
    """
    Find lexicon matches in sentence windows.

    Boilerplate windows are skipped so rankings/menus do not generate
    false positives.
    """
    sentences = split_sentences(text)
    windows = build_sentence_windows(sentences, window_size=2)

    results = []

    for i, window_text in enumerate(windows):
        if is_boilerplate_window(window_text):
            continue

        matches = find_matches(window_text, prepared_lexicon)

        if matches:
            results.append(
                {
                    "window_id": i,
                    "window_text": window_text[:500],
                    "matches": matches,
                }
            )

    return results


def group_has_prefix(matches: dict[str, list[str]], prefixes: list[str]) -> bool:
    return any(
        group == prefix or group.startswith(prefix + ".")
        for group in matches.keys()
        for prefix in prefixes
    )


def terms_for_prefix(matches: dict[str, list[str]], prefixes: list[str]) -> list[str]:
    terms = []

    for group, group_terms in matches.items():
        if any(group == prefix or group.startswith(prefix + ".") for prefix in prefixes):
            terms.extend(group_terms)

    return sorted(set(terms))


def window_has_combo(
    window_matches: list[dict[str, Any]],
    group_a_prefixes: list[str],
    group_b_prefixes: list[str],
) -> bool:
    """
    Return True if at least one sentence/window contains terms
    from both group A and group B.
    """
    for item in window_matches:
        matches = item["matches"]

        if group_has_prefix(matches, group_a_prefixes) and group_has_prefix(
            matches, group_b_prefixes
        ):
            return True

    return False


def collect_combo_evidence(
    window_matches: list[dict[str, Any]],
    group_a_prefixes: list[str],
    group_b_prefixes: list[str],
    max_examples: int = 3,
) -> list[dict[str, Any]]:
    """
    Collect examples of sentence/window evidence for manual review.
    """
    evidence = []

    for item in window_matches:
        matches = item["matches"]

        if group_has_prefix(matches, group_a_prefixes) and group_has_prefix(
            matches, group_b_prefixes
        ):
            evidence.append(item)

        if len(evidence) >= max_examples:
            break

    return evidence


def evidence_has_group_prefix(
    evidence: list[dict[str, Any]],
    prefixes: list[str],
) -> bool:
    """
    Return True if any collected evidence window contains a group prefix.
    """
    for item in evidence:
        matches = item.get("matches", {})
        if group_has_prefix(matches, prefixes):
            return True
    return False


def label_from_score(score: int, thresholds: dict | None = None) -> str:
    thresholds = thresholds or {}

    strong = thresholds.get("strong_candidate", 8)
    weak = thresholds.get("weak_candidate", 5)
    review = thresholds.get("review_candidate", 3)

    if score >= strong:
        return "strong_candidate"
    if score >= weak:
        return "weak_candidate"
    if score >= review:
        return "review_candidate"
    return "not_candidate"


def has_explicit_femicide_term(matches: dict[str, list[str]]) -> bool:
    """
    Detect the most explicit terms: femicide/feminicide or gender-based murder.
    """
    direct_terms = terms_for_prefix(matches, ["direct_terms.high_precision"])
    normalized = [normalize_text(t) for t in direct_terms]

    return any(
        "femicid" in t
        or "feminicid" in t
        or "asesinato por razones de genero" in t
        or "muerte violenta de una mujer" in t
        or "muerte violenta de mujeres" in t
        for t in normalized
    )


def has_male_victim_pattern(text: str) -> bool:
    """
    Detect windows where the violent event seems to have a male victim.

    This avoids classifying as violence against women cases where terms such
    as "mujer" or "esposa" appear, but the person killed/attacked is male.
    """
    norm = normalize_text(text)

    male_victim_patterns = [
        r"\bun hombre\b.{0,220}\b(murio|fallecio|fue asesinado|asesinaron|lo mataron|mataron|muerto|baleado|atacado)\b",
        r"\b(murio|fallecio|fue asesinado|asesinaron|lo mataron|mataron|muerto|baleado|atacado)\b.{0,220}\bun hombre\b",
        r"\bel hombre\b.{0,220}\b(murio|fallecio|fue asesinado|lo mataron|muerto|baleado|atacado)\b",
        r"\bun joven\b.{0,220}\b(murio|fallecio|fue asesinado|muerto|baleado|atacado)\b",
        r"\bun nene\b.{0,220}\b(murio|fallecio|fue asesinado|muerto|baleado)\b",
        r"\bun policia\b.{0,220}\b(murio|fallecio|fue asesinado|muerto|ejecutado)\b",
        r"\bla victima\b.{0,120}\b(identificada como jose|identificado como|era un hombre|fue un hombre)\b",
    ]

    return any(re.search(pattern, norm) for pattern in male_victim_patterns)


def any_evidence_has_male_victim(evidence: list[dict[str, Any]]) -> bool:
    return any(has_male_victim_pattern(item.get("window_text", "")) for item in evidence)


def compute_relevance_scores(
    matches_global: dict[str, list[str]],
    matches_by_window: list[dict[str, Any]],
    thresholds: dict | None = None,
) -> dict[str, Any]:
    """
    Compute retrieval scores and a retrieval bucket.

    This classifier is not a final truth classifier. It is a retrieval layer.

    Main buckets:
        case_strong:
            Strong candidate for a concrete case.

        case_review_*:
            Ambiguous or weak case evidence. Needs manual review.

        topic_*:
            General thematic coverage of gender-based violence, policy,
            prevention, sexual violence, digital violence, etc.

        not_candidate:
            No sufficient evidence.

    Core methodological rules:
        - Female reference terms do not classify alone.
        - Generic crime terms do not classify alone.
        - Relationship terms do not classify alone.
        - Proximity matters.
        - Accident/disaster context is marked, not automatically excluded.
        - Male-victim patterns only exclude weak review candidates.
    """
    thresholds = thresholds or {}

    case_score = 0
    topic_score = 0
    reasons = []
    proximity_evidence = []
    retrieval_bucket = "not_candidate"

    female_groups = [
        "female_reference_terms.strong",
        "female_reference_terms.gendered_phrases",
    ]

    direct_groups = [
        "direct_terms.high_precision",
    ]

    legal_groups = [
        "direct_terms.legal_policy_terms",
    ]

    death_strong_groups = [
        "indirect_terms.victim_death_phrases.violent_or_suspicious",
    ]

    death_review_groups = [
        "indirect_terms.victim_death_phrases.ambiguous_or_accidental",
    ]

    disappearance_groups = [
        "indirect_terms.disappearance_phrases",
    ]

    injury_or_method_groups = [
        "indirect_terms.injury_or_method_terms",
    ]

    generic_crime_groups = [
        "indirect_terms.generic_crime_terms",
    ]

    violence_topic_groups = [
        "violence_types_and_modalities.types",
        "violence_types_and_modalities.digital_or_cyber_violence",
        "digital_or_cyber_violence_terms",
        "sexual_violence_terms",
    ]

    structural_context_groups = [
        "structural_context_terms",
    ]

    help_groups = [
        "help_and_prevention_terms",
    ]

    accident_groups = [
        "accident_or_disaster_terms.strong",
    ]

    # 1. Explicit direct terms.
    if group_has_prefix(matches_global, direct_groups):
        if has_explicit_femicide_term(matches_global):
            case_score = max(case_score, 12)
            topic_score = max(topic_score, 10)
            retrieval_bucket = "case_strong"
            reasons.append("explicit_femicide_or_gender_murder_term")
        else:
            topic_score = max(topic_score, 10)
            if retrieval_bucket == "not_candidate":
                retrieval_bucket = "topic_gender_violence"
            reasons.append("direct_gender_violence_term")

    # 2. Legal/policy/help terms: topic.
    if group_has_prefix(matches_global, legal_groups):
        topic_score = max(topic_score, 6)
        if retrieval_bucket == "not_candidate":
            retrieval_bucket = "topic_policy_or_prevention"
        reasons.append("legal_or_policy_term")

    if group_has_prefix(matches_global, help_groups):
        topic_score = max(topic_score, 4)
        if retrieval_bucket == "not_candidate":
            retrieval_bucket = "topic_policy_or_prevention"
        reasons.append("help_or_prevention_resource")

    # 3. Structural context: machismo, misogyny, sexism, etc.
    if group_has_prefix(matches_global, structural_context_groups):
        topic_score = max(topic_score, 5)
        if retrieval_bucket == "not_candidate":
            retrieval_bucket = "topic_gender_violence"
        reasons.append("structural_gender_context")

    # 4. Female reference + violent/suspicious death.
    if window_has_combo(matches_by_window, female_groups, death_strong_groups):
        case_score = max(case_score, 10)
        retrieval_bucket = "case_strong"
        reasons.append("female_reference_plus_violent_or_suspicious_death")
        proximity_evidence.extend(
            collect_combo_evidence(matches_by_window, female_groups, death_strong_groups)
        )

    # 5. Female reference + disappearance.
    if window_has_combo(matches_by_window, female_groups, disappearance_groups):
        case_score = max(case_score, 7)

        if retrieval_bucket != "case_strong":
            retrieval_bucket = "case_review_female_disappearance"

        reasons.append("female_reference_plus_disappearance")
        proximity_evidence.extend(
            collect_combo_evidence(matches_by_window, female_groups, disappearance_groups)
        )

    # 6. Female reference + injury/method.
    if window_has_combo(matches_by_window, female_groups, injury_or_method_groups):
        case_score = max(case_score, 6)

        if retrieval_bucket != "case_strong":
            retrieval_bucket = "case_review_possible_violence"

        reasons.append("female_reference_plus_injury_or_method")
        proximity_evidence.extend(
            collect_combo_evidence(matches_by_window, female_groups, injury_or_method_groups)
        )

    # 7. Female reference + ambiguous or accidental death.
    # Example: "la joven ingresó sin vida", "murió ahogada".
    if window_has_combo(matches_by_window, female_groups, death_review_groups):
        case_score = max(case_score, 4)

        if retrieval_bucket == "not_candidate":
            retrieval_bucket = "case_review_female_death"

        reasons.append("female_reference_plus_ambiguous_or_accidental_death")
        proximity_evidence.extend(
            collect_combo_evidence(matches_by_window, female_groups, death_review_groups)
        )

    # 8. Female reference + generic crime.
    # Weak evidence only. Never strong by itself.
    if window_has_combo(matches_by_window, female_groups, generic_crime_groups):
        reasons.append("female_reference_plus_generic_crime_observed")
        proximity_evidence.extend(
            collect_combo_evidence(matches_by_window, female_groups, generic_crime_groups)
        )

    # 9. Explicit topic: sexual/digital/gender-based violence against women.
    if window_has_combo(matches_by_window, female_groups, violence_topic_groups):
        topic_score = max(topic_score, 4)

        if retrieval_bucket == "not_candidate":
            retrieval_bucket = "topic_gender_violence"

        reasons.append("female_reference_plus_explicit_violence_topic")
        proximity_evidence.extend(
            collect_combo_evidence(matches_by_window, female_groups, violence_topic_groups)
        )

    # 10. Accident/disaster context is marked, not automatically excluded.
    if evidence_has_group_prefix(proximity_evidence, accident_groups):
        reasons.append("possible_accident_or_disaster_context")

    # 11. Male-victim patterns exclude only weak review candidates.
    if retrieval_bucket.startswith("case_review") and any_evidence_has_male_victim(
        proximity_evidence
    ):
        case_score = 0
        retrieval_bucket = "not_candidate"
        reasons.append("male_victim_context_exclusion")

    case_label = label_from_score(case_score, thresholds)
    topic_label = label_from_score(topic_score, thresholds)

    is_case_candidate = retrieval_bucket.startswith("case_")
    is_topic_candidate = topic_label != "not_candidate"

    relevance_score = max(case_score, topic_score)

    if retrieval_bucket == "case_strong":
        candidate_label = "strong_candidate"
    elif retrieval_bucket.startswith("case_review"):
        candidate_label = "review_candidate"
    elif retrieval_bucket.startswith("topic"):
        candidate_label = topic_label
    else:
        candidate_label = "not_candidate"

    if is_case_candidate and is_topic_candidate:
        candidate_type = "case_and_topic"
    elif is_case_candidate:
        candidate_type = "case"
    elif is_topic_candidate:
        candidate_type = "topic"
    else:
        candidate_type = "none"

    return {
        "case_score": case_score,
        "topic_score": topic_score,
        "relevance_score": relevance_score,
        "case_label": case_label,
        "topic_label": topic_label,
        "candidate_label": candidate_label,
        "candidate_type": candidate_type,
        "retrieval_bucket": retrieval_bucket,
        "is_case_candidate": is_case_candidate,
        "is_topic_candidate": is_topic_candidate,
        "is_retrieval_candidate": is_case_candidate or is_topic_candidate,
        "match_reasons": sorted(set(reasons)),
        "proximity_evidence": proximity_evidence[:5],
    }


def classify_article(
    row: dict,
    lexicon: dict,
    prepared_lexicon: dict[str, list[tuple[str, re.Pattern]]] | None = None,
) -> dict:
    """
    Classify one article.

    For retrieval, we use:
        anchor_text + title + first 2500 chars of extracted text

    Reason:
        archived HTML often includes related links, sidebars, menus or
        recommendations near the bottom. Using the full text can create
        false positives.
    """
    if prepared_lexicon is None:
        prepared_lexicon = prepare_lexicon(lexicon)

    anchor_title_text = "\n".join(
        [
            str(row.get("anchor_text", "")),
            str(row.get("title", "")),
        ]
    )

    body_text = str(row.get("text", ""))
    body_head = body_text[:2500]

    retrieval_text = "\n".join(
        [
            anchor_title_text,
            body_head,
        ]
    )

    matches_global = find_matches(retrieval_text, prepared_lexicon)
    matches_by_window = find_matches_by_window(retrieval_text, prepared_lexicon)

    thresholds = lexicon.get("thresholds", {})

    scoring = compute_relevance_scores(
        matches_global=matches_global,
        matches_by_window=matches_by_window,
        thresholds=thresholds,
    )

    return {
        **scoring,
        "matched_terms_json": json.dumps(matches_global, ensure_ascii=False),
        "matched_groups_json": json.dumps(list(matches_global.keys()), ensure_ascii=False),
        "match_reasons_json": json.dumps(scoring["match_reasons"], ensure_ascii=False),
        "proximity_evidence_json": json.dumps(
            scoring["proximity_evidence"],
            ensure_ascii=False,
        ),
    }