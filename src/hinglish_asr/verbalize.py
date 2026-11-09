"""Digits -> spoken number words in training targets (text normalization).

Most vendors already write numbers as words; Gnani and some Deepgram rows use
digits (`₹2,000`, `04:00 बजे`, `9876543210`). This writes them as spoken, in
the row's language, so every target reads numbers the same way:

    ₹2,000            -> दो हज़ार रुपये          two thousand rupees
    04:00 / 4:30      -> चार / साढ़े चार          four / four thirty
    50%               -> पचास प्रतिशत            fifty percent
    1,50,000          -> एक लाख पचास हज़ार        one lakh fifty thousand
    2.5, 3.5, 1.25    -> ढाई, साढ़े तीन, सवा एक      two point five
    9876543210, 0798  -> digit by digit (a leading zero, or 5+ digits not ending in 000:
                         phone, OTP, account; 50000 is an amount)
    3rd, 12th         -> third, twelfth (English in both languages: Hinglish says them so)

A digit run joined to letters (`4G`, `COVID-19`) is left as written; the
report counts rows that still hold a digit.
"""

from __future__ import annotations

import re

from .semantic import _HI_NUMBERS

# The first spelling listed for each value is the one written.
HI: dict[int, str] = {}
for _item in _HI_NUMBERS.split("|"):
    _word, _value = _item.split()
    if float(_value).is_integer():
        HI.setdefault(int(float(_value)), _word)

_EN_ONES = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen"
).split()
_EN_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
# Indian numbering in both languages: Indian English says lakh and crore.
_SCALES = ((10**7, "करोड़", "crore"), (10**5, "लाख", "lakh"), (1000, "हज़ार", "thousand"), (100, "सौ", "hundred"))
_WORDS = {
    "hi": {"rupees": "रुपये", "percent": "प्रतिशत", "point": "दशमलव"},
    "en": {"rupees": "rupees", "percent": "percent", "point": "point"},
}


def _below_100(n: int, lang: str) -> list[str]:
    if lang == "hi":
        return [HI[n]]
    if n < 20:
        return [_EN_ONES[n]]
    tens, ones = divmod(n, 10)
    return [_EN_TENS[tens]] + ([_EN_ONES[ones]] if ones else [])


def cardinal(n: int, lang: str) -> list[str]:
    if n < 100:
        return _below_100(n, lang)
    out = []
    for value, hi, en in _SCALES:
        q, n = divmod(n, value)
        if q:
            out += cardinal(q, lang) + [hi if lang == "hi" else en]
    if n:
        out += _below_100(n, lang)
    return out


def digits(s: str, lang: str) -> list[str]:
    return [_below_100(int(d), lang)[0] for d in s]


def _number(s: str, lang: str) -> list[str]:
    """A plain digit string: a count, or an id read digit by digit (a round 50000 is an amount)."""
    if (len(s) > 1 and s[0] == "0") or (len(s) >= 5 and not s.endswith("000")):
        return digits(s, lang)
    return cardinal(int(s), lang)


def _hi_fraction(whole: int, frac: str) -> list[str] | None:
    """2.5 -> ढाई, 3.5 -> साढ़े तीन, 2.25 -> सवा दो, 2.75 -> पौने तीन: how Hindi says halves and quarters."""
    frac = frac.rstrip("0")
    if frac == "5":
        return {1: ["डेढ़"], 2: ["ढाई"]}.get(whole) or (["साढ़े", *cardinal(whole, "hi")] if whole else None)
    if frac == "25" and whole:
        return ["सवा", *cardinal(whole, "hi")]
    if frac == "75":
        return ["पौने", *cardinal(whole + 1, "hi")]
    return None


def _amount(s: str, lang: str) -> list[str]:
    """Digits with Indian or Western grouping commas and an optional decimal part."""
    whole, _, frac = s.replace(",", "").partition(".")
    if frac and lang == "hi" and (spoken := _hi_fraction(int(whole or 0), frac)):
        return spoken
    out = cardinal(int(whole), lang) if whole else []
    return out + ([_WORDS[lang]["point"]] + digits(frac, lang) if frac else [])


def _time(h: int, m: int, lang: str) -> list[str]:
    if lang == "en":
        if m == 0:
            return cardinal(h, lang)
        return cardinal(h, lang) + (["oh"] + cardinal(m, lang) if m < 10 else cardinal(m, lang))
    if m == 0:
        return cardinal(h, lang)
    if m == 30:
        return {1: ["डेढ़"], 2: ["ढाई"]}.get(h, ["साढ़े", *cardinal(h, lang)])
    if m == 15:
        return ["सवा", *cardinal(h, lang)]
    if m == 45:
        return ["पौने", *cardinal(h % 12 + 1, lang)]
    return cardinal(h, lang) + cardinal(m, lang)


def _ordinal(n: int) -> list[str]:
    words = cardinal(n, "en")
    last = words[-1]
    special = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
               "nine": "ninth", "twelve": "twelfth"}
    words[-1] = special.get(last) or (last[:-1] + "ieth" if last.endswith("y") else last + "th")
    return words


_EDGE = r"(?<![0-9A-Za-zऀ-ॿ])"  # not glued to a letter or digit before
_END = r"(?![0-9A-Za-zऀ-ॿ])"
_PATTERNS = [
    ("money", re.compile(r"₹\s?(\d[\d,]*(?:\.\d+)?)" + _END)),
    ("time", re.compile(_EDGE + r"(\d{1,2}):(\d{2})" + _END)),
    ("percent", re.compile(_EDGE + r"(\d+(?:\.\d+)?)\s?%")),
    ("ordinal", re.compile(_EDGE + r"(\d+)(?:st|nd|rd|th)" + _END, re.IGNORECASE)),
    ("amount", re.compile(_EDGE + r"(\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+\.\d+)" + _END)),
    ("number", re.compile(_EDGE + r"(\d+)" + _END)),
]


def verbalize_numbers(text: str, lang: str) -> str:
    """Digits in `text` as spoken words in `lang` ("hi" or "en")."""
    lang = "en" if lang == "en" else "hi"
    w = _WORDS[lang]

    def sub(kind):
        def repl(m: re.Match) -> str:
            if kind == "money":
                words = _amount(m.group(1), lang) + [w["rupees"]]
            elif kind == "time":
                h, mi = int(m.group(1)), int(m.group(2))
                if h > 23 or mi > 59:
                    return m.group(0)
                words = _time(h, mi, lang)
            elif kind == "percent":
                words = _amount(m.group(1), lang) + [w["percent"]]
            elif kind == "ordinal":  # 12th, 1st are said in English in Hindi rows too
                words = _ordinal(int(m.group(1)))
            elif kind == "amount":
                words = _amount(m.group(1), lang)
            else:
                words = _number(m.group(1), lang)
            return f" {' '.join(words)} "
        return repl

    for kind, pattern in _PATTERNS:
        text = pattern.sub(sub(kind), text)
    return " ".join(text.split())
