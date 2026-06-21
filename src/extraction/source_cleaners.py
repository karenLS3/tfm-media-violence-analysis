from __future__ import annotations

import re


def repair_mojibake(text: str) -> str:
    """
    Repara casos típicos como:
    saÃ±a -> saña
    MarÃ­a -> María
    MÃ©xico -> México

    """
    if not isinstance(text, str):
        return ""

    suspicious = ["Ã", "Â", "â€™", "â€œ", "â€"]

    if not any(s in text for s in suspicious):
        return text

    try:
        fixed = text.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
        if fixed and fixed.count("Ã") < text.count("Ã"):
            return fixed
    except Exception:
        pass

    return text


def normalize_spaces(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_flipear_blocks(text: str) -> str:
    """
    El Universal suele extraer bloques como:

    -
    flipear
    Titular relacionado

    Esos titulares contaminan el filtro. Eliminamos el bloque entero:
    '-', 'flipear' y la línea siguiente.
    """
    lines = [line.strip() for line in text.splitlines()]
    cleaned = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        low = line.lower()

        if line == "-" and i + 1 < len(lines) and lines[i + 1].strip().lower() == "flipear":
            # Saltar "-", "flipear" y el titular relacionado siguiente.
            i += 3
            continue

        if low == "flipear":
            # Si aparece solo, saltar también posible titular siguiente.
            i += 2
            continue

        cleaned.append(line)
        i += 1

    return "\n".join(line for line in cleaned if line)


def remove_common_boilerplate(text: str) -> str:
    bad_contains = [
        "política de privacidad",
        "todos los derechos reservados",
        "desde su móvil acceda",
        "sitio desarrollado con software libre",
        "comentarios",
        "newsletter",
        "síguenos en",
        "compartir en facebook",
        "compartir en twitter",
    ]

    lines = []
    for line in text.splitlines():
        clean = line.strip()
        low = clean.lower()

        if not clean:
            continue

        if any(pattern in low for pattern in bad_contains):
            continue

        lines.append(clean)

    return "\n".join(lines)


def clean_eluniversal_text(text: str) -> str:
    text = repair_mojibake(text)
    text = remove_flipear_blocks(text)
    text = remove_common_boilerplate(text)
    text = normalize_spaces(text)
    return text


def clean_pagina12_text(text: str) -> str:
    text = repair_mojibake(text)
    text = remove_common_boilerplate(text)
    text = normalize_spaces(text)
    return text


def clean_generic_text(text: str) -> str:
    text = repair_mojibake(text)
    text = remove_common_boilerplate(text)
    text = normalize_spaces(text)
    return text


def clean_text_by_source(text: str, source: str) -> str:
    source = str(source or "").lower()

    if source == "eluniversal_mx":
        return clean_eluniversal_text(text)

    if source == "pagina12":
        return clean_pagina12_text(text)

    return clean_generic_text(text)