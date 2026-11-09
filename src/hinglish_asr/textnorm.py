"""Training-target text normalization for mixed-script Hinglish.

This is NOT the scoring normalizer (see `scoring`). `normalize_target` keeps
case and punctuation, because the base model was trained to emit them; it only
removes differences that would corrupt labels:

- Devanagari digits (०-९) have no tokenizer pieces and would train as <unk>.
- Zero-width characters are not in the vocab and are invisible in review.
- Precomposed nukta letters (U+0958-095F) are Unicode composition exclusions,
  so NFC decomposes them to base + U+093C, the form the tokenizer produces.
- ॥ is not in the vocab; । is.
- Vaani transcripts gloss English words spoken by the speaker as
  "रेस्टोरेंट {restaurant}" (1.6% of Hush-Eval's hindi_test references). The
  convention here is English words in Latin script, so the gloss replaces the
  Devanagari spelling; an empty gloss "{}" is dropped.

`spoken_target` (data config `target_style: spoken`) then writes the plain
spoken style of hinden-ASR's recipe: numbers as words, lowercase, no
punctuation.
"""

from __future__ import annotations

import re
import unicodedata

_GLOSS = re.compile(r"(?:(\S+)\s+)?\{([^{}]*)\}")

_TRANSLATE = str.maketrans(
    {
        **{chr(0x0966 + d): str(d) for d in range(10)},
        "​": None,  # zero width space
        "‌": None,  # ZWNJ
        "‍": None,  # ZWJ
        "﻿": None,  # BOM / zero width no-break space
        "॥": "।",  # ॥ -> ।
    }
)


def _resolve_gloss(m: re.Match) -> str:
    gloss = m.group(2).strip()
    return gloss or (m.group(1) or "")


def normalize_target(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "").translate(_TRANSLATE)
    text = _GLOSS.sub(_resolve_gloss, text)
    return " ".join(text.split())


def _strip_punct(text: str) -> str:
    """Punctuation and symbols become spaces; an apostrophe between letters stays (ma'am, don't)."""
    out = []
    for i, ch in enumerate(text):
        if unicodedata.category(ch)[0] in "PS":
            inner = ch in "'’" and 0 < i < len(text) - 1 and text[i - 1].isalpha() and text[i + 1].isalpha()
            out.append("'" if inner else " ")
        else:
            out.append(ch)
    return "".join(out)


def spoken_target(text: str, lang: str) -> str:
    """The plain spoken style: numbers as words in the row's language, lowercase, no punctuation.

    `text` is a `normalize_target` result; `lang` is the row's language code (hi, en).
    """
    from .verbalize import verbalize_numbers

    return " ".join(_strip_punct(verbalize_numbers(text, lang).lower()).split())
