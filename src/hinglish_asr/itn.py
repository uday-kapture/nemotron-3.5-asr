"""Number words -> digits in training targets (inverse text normalization).

Not used by the pipeline at the moment: `convert` keeps numbers as each
source wrote them. Kept for when digit targets come back.

Deepgram writes most numbers as Hindi words, and the base model's hi-IN mode
does too. This rewrites them:

    दो सौ पचास रुपये             -> 250 रुपये
    नौ आठ सात छह पांच चार        -> 987654       (digit by digit: joined)
    दो लाख 50 हज़ार / सवा लाख     -> 250000 / 125000
    पच्चीस तारीख, twenty five    -> 25 तारीख, 25

Left as words, because the digit form would be wrong or is unsure:

- a lone number below 10: एक बार, दो दिन, कर दो (दो is also "give", एक "a");
- two or three numbers in a row read as a range: एक दो दिन, दस बारह, दो तीन सौ.
  A digit-by-digit number needs at least 4 numbers, or 3 read in pairs;
- numbers in a row that are each whole: दो हज़ार तीन हज़ार, चार सौ बीस पांच सौ
  (2000, 3000 / 420, 500): place values only fall within one number;
- fractions: डेढ़, साढ़े तीन (बजे): the tokenizer has no ":" for 3:30;
- दो after a multiplier at the end of a sentence (पांच सौ दो = "give 500",
  not 502): Hindi puts the verb last.

Numbers already in digits are kept as written. Punctuation next to a word
ends a number. The number words and grammar are `semantic`'s, so lenient WER
reads both forms the same way.
"""

from __future__ import annotations

import unicodedata

from .semantic import _number_token, dev_fold, form_switch, run_pieces

_DO = dev_fold("दो")
MIN_DIGIT_RUN = 4  # numbers read digit by digit (phone, OTP, PIN); fewer is a range like "एक दो"
MIN_PAIR_RUN = 3  # two-digit numbers read in pairs (पचासी तैंतालीस सत्ताईस = 854327); two is a range like "दस बारह"


def _is_punct(ch: str) -> bool:
    return unicodedata.category(ch)[0] in "PS"


def _split_edges(tok: str) -> tuple[str, str, str]:
    i, j = 0, len(tok)
    while i < j and _is_punct(tok[i]):
        i += 1
    while j > i and _is_punct(tok[j - 1]):
        j -= 1
    return tok[:i], tok[i:j], tok[j:]


def _digits(nums: list[tuple[str, float, str | None]]) -> str | None:
    """The digit form of one run of number tokens, or None to keep its words."""
    if all(n[0] == "num" and n[2] is not None for n in nums) or all(n[0] == "mod" for n in nums):
        return None  # already digits, or only साढ़े/सवा/पौने
    if len(nums) == 1 and nums[0][0] == "num" and nums[0][1] < 10:
        return None
    pieces = run_pieces(nums)
    if any("." in p for p in pieces):
        return None
    if len(pieces) > 1 and any(len(p) > 2 for p in pieces):
        return None
    pairs = len(pieces) >= MIN_PAIR_RUN and all(len(p) == 2 for p in pieces)
    if 1 < len(pieces) < MIN_DIGIT_RUN and not pairs:
        return None
    return "".join(pieces)


def numbers_to_digits(text: str) -> str:
    toks = text.split()
    parts = [_split_edges(t) for t in toks]
    nums = [_number_token(core.casefold()) if core else None for _, core, _ in parts]
    out: list[str] = []
    i = 0
    while i < len(toks):
        if nums[i] is None:
            out.append(toks[i])
            i += 1
            continue
        j = i + 1  # the run is toks[i:j]
        while (
            j < len(toks) and nums[j] is not None and not parts[j - 1][2] and not parts[j][0]
            and not form_switch(nums[j - 1], nums[j])
        ):
            j += 1
        end = j
        sentence_end = j == len(toks) or bool(parts[j - 1][2])
        if sentence_end and end - i >= 2 and nums[end - 2][0] == "mult" and dev_fold(parts[end - 1][1]) == _DO:
            end -= 1
        digits = _digits(nums[i:end])
        if digits is None:
            out.extend(toks[i:end])
        else:
            out.append(parts[i][0] + digits + parts[end - 1][2])
        out.extend(toks[end:j])
        i = j
    return " ".join(out)
