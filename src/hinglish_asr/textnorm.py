"""Training-target text normalization for mixed-script Hinglish.

This is NOT the scoring normalizer (see `scoring`). Training targets keep case
and punctuation, because the base model was trained to emit them; this only
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
