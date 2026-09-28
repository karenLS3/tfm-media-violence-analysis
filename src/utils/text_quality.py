from __future__ import annotations

import html
import re
import unicodedata

import ftfy


# =====================================================================
# DETECTORES
# =====================================================================

MOJIBAKE_RE = re.compile(
    r"Ã|Â|â|�|Ì"
)

# El detector anterior <[^>]+> confundía << texto >> con HTML.
HTML_RE = re.compile(
    r"</?[A-Za-z][A-Za-z0-9:_-]*"
    r"(?:\s+[^<>]*?)?/?>"
)

URL_RE = re.compile(
    r"https?://\S+|www\.\S+",
    flags=re.IGNORECASE,
)

ACUTE_FROM_LOST_COMBINING = {
    "a": "á",
    "e": "é",
    "i": "í",
    "o": "ó",
    "u": "ú",
    "A": "Á",
    "E": "É",
    "I": "Í",
    "O": "Ó",
    "U": "Ú",
}


KNOWN_GRAVE_CORRUPTIONS = {
    # Pronombres / adverbios
    "mà": "mí",
    "ahà": "ahí",
    "allà": "allí",

    # Gentilicios / sustantivos
    "iranà": "iraní",
    "israelà": "israelí",
    "haità": "haití",
    "otomà": "otomí",

    # Verbos
    "conocà": "conocí",
    "pedà": "pedí",
    "vivà": "viví",
    "sobrevivà": "sobreviví",
    "perdà": "perdí",
    "fungà": "fungí",
    "referà": "referí",
    "volvà": "volví",
    "decidà": "decidí",
    "sentà": "sentí",
    "salà": "salí",
    "prometà": "prometí",

    # Sustantivo
    "esquà": "esquí",

    # Nombres propios comprobados
    "rubà": "rubí",
    "nohemà": "nohemí",
    "noemà": "noemí",
    "anahà": "anahí",
    "alà": "alí",
    "analà": "analí",
    "anaà": "anaí",
    "cundapà": "cundapí",
    "marà": "marí",

    # Errores detectados anteriormente
    "asà": "así",
    "sà": "sí",
    "aquà": "aquí",
    "creà": "creí",
    "nacà": "nací",
    "escogà": "escogí",
    "neftalà": "neftalí",

    "ocurriò": "ocurrió",
    "vìa": "vía",
    "travès": "través",
    "comisiòn": "comisión",
    "bazaldùa": "bazaldúa",
    "mèxico": "méxico",
    "investigaciòn": "investigación",
    "desapariciòn": "desaparición",
    "secretarìa": "secretaría",
}


def repair_lost_combining_accents(text: str) -> str:
    """
    Repara una corrupción observada en el corpus.

    Ejemplos:
        viÌctimas       -> víctimas
        MeÌxico         -> México
        geÌnero         -> género
        investigacioÌn  -> investigación

    El carácter Ì es el residuo de una codificación incorrecta
    de un acento agudo combinante cuyo segundo byte desapareció.
    """

    if not isinstance(text, str):
        return ""

    pattern = re.compile(r"([aeiouAEIOU])Ì")

    return pattern.sub(
        lambda m: ACUTE_FROM_LOST_COMBINING[
            m.group(1)
        ],
        text,
    )


def _preserve_case(
    original: str,
    replacement: str,
) -> str:

    if original.isupper():
        return replacement.upper()

    if original[:1].isupper():
        return (
            replacement[:1].upper()
            + replacement[1:]
        )

    return replacement


def repair_known_grave_corruptions(
    text: str,
    source: str | None = None,
) -> str:
    """
    Corrige únicamente formas con acento grave espurio
    verificadas durante la auditoría.

    No se sustituyen globalmente à/è/ì/ò/ù, porque existen
    usos legítimos en nombres y expresiones extranjeras.
    """

    if not isinstance(text, str):
        return ""

    # Caso especial:
    # "Ademà s del encuentro" -> "Además del encuentro"
    text = re.sub(
        r"\bademà\s+s\b",
        lambda m: (
            "Además"
            if m.group(0)[0].isupper()
            else "además"
        ),
        text,
        flags=re.IGNORECASE,
    )

    for bad, good in KNOWN_GRAVE_CORRUPTIONS.items():

        pattern = re.compile(
            rf"\b{re.escape(bad)}\b",
            flags=re.IGNORECASE,
        )

        text = pattern.sub(
            lambda m: _preserve_case(
                m.group(0),
                good,
            ),
            text,
        )

    return text


# =====================================================================
# BOILERPLATE GENÉRICO
# =====================================================================

GENERIC_BOILERPLATE_PATTERNS = [
    r"\btodos los derechos reservados\b",
    r"\btodos derechos reservados\b",
    r"\bderechos reservados\b",

    r"\bpol[ií]tica de privacidad\b",
    r"\bpol[ií]tica de cookies\b",

    r"\bdesarrollado (?:con|en) software libre\b",
    r"\bsoftware libre\b",
    r"\bgnu\s*/?\s*linux\b",

    r"\bpic\.?\s*twitter\.?\s*com\b",
    r"\btwitter\.?\s*com\b",
]


# =====================================================================
# REPARACIÓN UNICODE
# =====================================================================

def repair_unicode(text: str) -> str:
    """
    Repara problemas de codificación observados en el corpus.

    Orden:
    1. entidades HTML;
    2. reparación general con ftfy;
    3. reparación de residuos producidos por pérdida previa
       de bytes/caracteres del mojibake;
    4. normalización Unicode NFC.
    """

    if not isinstance(text, str):
        return ""

    text = html.unescape(text)

    # Casos recuperables directamente:
    #
    # MÃ©xico -> México
    # gÃ©nero -> género
    # aÃ±os   -> años
    text = ftfy.fix_text(text)

    # -------------------------------------------------------------
    # Residuo específico observado en el corpus.
    #
    # La secuencia UTF-8 de í contiene el byte AD.
    # Tras una decodificación incorrecta aparece como U+00AD
    # (soft hyphen), que fue eliminado en una etapa anterior.
    #
    # Fiscalía -> FiscalÃ­a -> FiscalÃa
    # víctima  -> vÃ­ctima  -> vÃctima
    # país     -> paÃ­s      -> paÃs
    #
    # Después de ftfy, cualquier Ã residual observado en este
    # corpus corresponde a ese patrón.
    # -------------------------------------------------------------

    text = text.replace("Ã", "í")

    # Restos espurios de secuencias de codificación.
    text = text.replace("Â", "")

    # Restos observados de puntuación UTF-8 dañada.
    # â¦ corresponde a una elipsis mutilada.
    text = text.replace("â¦", "…")

    # Los restantes â observados corresponden principalmente
    # a comillas/apóstrofos/dashes cuyos bytes intermedios
    # desaparecieron. Para el corpus analítico es más seguro
    # separarlos que inventar el signo original.
    text = text.replace("â", " ")

    text = unicodedata.normalize(
        "NFC",
        text,
    )

    return text


# =====================================================================
# URLS
# =====================================================================

def remove_urls(text: str) -> str:
    if not isinstance(text, str):
        return ""

    return URL_RE.sub(" ", text)


# =====================================================================
# BOILERPLATE GENÉRICO
# =====================================================================

def remove_generic_boilerplate(text: str) -> str:
    if not isinstance(text, str):
        return ""

    for pattern in GENERIC_BOILERPLATE_PATTERNS:
        text = re.sub(
            pattern,
            " ",
            text,
            flags=re.IGNORECASE,
        )

    return text


# =====================================================================
# BOILERPLATE ESPECÍFICO POR FUENTE
# =====================================================================

def remove_source_boilerplate(
    text: str,
    source: str | None,
) -> str:

    if not isinstance(text, str):
        return ""

    source = str(source or "").lower()

    # -------------------------------------------------------------
    # Página/12
    # -------------------------------------------------------------

    if source == "pagina12":

        # Footer móvil de la plantilla histórica.
        text = re.sub(
            r"(?im)^\s*desde su móvil acceda "
            r"a través de https?://\S+\s*$",
            " ",
            text,
        )

        # © 2000-2015 www.pagina12.com.ar |
        # República Argentina |
        # Política de privacidad |
        # Todos los Derechos Reservados
        text = re.sub(
            r"(?im)^\s*©\s*\d{4}-\d{4}"
            r".*?todos los derechos reservados\s*$",
            " ",
            text,
        )

        # Footer GNU/Linux.
        text = re.sub(
            r"(?im)^\s*sitio desarrollado con "
            r"software libre gnu\s*/?\s*linux\.?\s*$",
            " ",
            text,
        )

    # -------------------------------------------------------------
    # Clarín
    # -------------------------------------------------------------

    elif source == "clarin_arg":

        text = re.sub(
            r"(?im)^\s*los videos más vistos\s*$",
            " ",
            text,
        )

    # -------------------------------------------------------------
    # La Nación
    # -------------------------------------------------------------

    elif source == "lanacion":

        text = re.sub(
            r"(?im)^\s*[-–—]?\s*"
            r"\d+\s+minutos?\s+de\s+lectura['’]?\s*$",
            " ",
            text,
        )

    return text


# =====================================================================
# ESPACIOS
# =====================================================================

def normalize_whitespace(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n[ \t]+\n",
        "\n\n",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


# =====================================================================
# PIPELINE CANÓNICO PARA TEXTO DE ANÁLISIS
# =====================================================================

def clean_analysis_text(
    text: str,
    source: str | None = None,
) -> str:

    # Reparación Unicode general
    text = repair_unicode(text)

    # Corrupción del tipo:
    # MeÌxico -> México
    text = repair_lost_combining_accents(text)

    # Corrupciones puntuales verificadas:
    # asà -> así, Comisiòn -> Comisión, etc.
    text = repair_known_grave_corruptions(
        text,
        source,
    )

    # Boilerplate dependiente del medio
    text = remove_source_boilerplate(
        text,
        source,
    )

    text = remove_urls(text)

    text = remove_generic_boilerplate(text)

    text = normalize_whitespace(text)

    return text

# =====================================================================
# DETECTORES
# =====================================================================

def contains_mojibake(text: str) -> bool:
    if not isinstance(text, str):
        return False

    return bool(
        MOJIBAKE_RE.search(text)
    )


def contains_html(text: str) -> bool:
    if not isinstance(text, str):
        return False

    return bool(
        HTML_RE.search(text)
    )


def contains_url(text: str) -> bool:
    if not isinstance(text, str):
        return False

    return bool(
        URL_RE.search(text)
    )

def contains_suspicious_grave_accents(
    text: str,
) -> bool:
    """
    Detecta vocales con acento grave, no esperadas en
    el texto periodístico español del corpus.
    """

    if not isinstance(text, str):
        return False

    return bool(
        re.search(
            r"[àèìòùÀÈÌÒÙ]",
            text,
        )
    )