from __future__ import annotations


MOJIBAKE_MARKERS = (
    "Ã",
    "Â",
    "â€",
    "â€™",
    "â€œ",
    "â€\x9d",
    "ðŸ",
)


def mojibake_score(text: str) -> int:
    return sum(
        text.count(marker)
        for marker in MOJIBAKE_MARKERS
    )


def repair_mojibake(value: object) -> str:
    """
    Repara conservadoramente mojibake típico producido cuando
    texto UTF-8 fue interpretado como Latin-1 o Windows-1252.

    La transformación solo se acepta si reduce los indicadores
    de mojibake. Nunca se descartan caracteres silenciosamente.
    """
    if value is None:
        return ""

    text = str(value)

    if not text:
        return ""

    best = text
    best_score = mojibake_score(best)

    # Si no hay evidencia de mojibake, no modificar el texto.
    if best_score == 0:
        return best

    # Hasta dos pasadas para casos de doble codificación.
    for _ in range(2):
        candidates: list[str] = []

        for encoding in ("latin1", "cp1252"):
            try:
                candidate = (
                    best
                    .encode(encoding)
                    .decode("utf-8")
                )
                candidates.append(candidate)

            except (
                UnicodeEncodeError,
                UnicodeDecodeError,
            ):
                continue

        if not candidates:
            break

        candidate = min(
            candidates,
            key=mojibake_score,
        )

        candidate_score = mojibake_score(candidate)

        if candidate_score >= best_score:
            break

        best = candidate
        best_score = candidate_score

    return best