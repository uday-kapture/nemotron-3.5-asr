"""Meaning-level scores for Hinglish ASR, reported next to plain WER.

Against machine (Deepgram) references, plain WER counts many differences that
keep the meaning. Two scores discount them:

lenient WER   WER from an alignment that treats these pairs as matches:
              - cross_script: the same word in the other script (`mobile` /
                `मोबाइल`): equal consonant skeletons of at least 2 consonants,
                or a small table of short words (ok, yes, sir, ...);
              - devanagari_variant: nasalization (है/हैं), nukta, vowel length
                (ि/ी, ु/ू), ये/ए (गये/गए);
              - number: number words vs digits (`दो सौ` / `200`, `एक लाख` /
                `1 लाख`), rewritten to one value on both sides first;
              - spacing: one word vs the same word split in two
                (`मिडकैप` / `मिड कैप`).
              Errors are divided by the reference's normalized word count, as
              for WER. Every accepted match is counted by rule,
              and cross-script pairs are listed so the rules can be audited.
              The rules come from general Hinglish patterns; do not tune them
              on the test split.
semdist       1 - cosine similarity of LaBSE sentence embeddings of the
              normalized reference and hypothesis (Kim et al. 2021). Needs the
              `semantic` extra (sentence-transformers).

Both normalize with `scoring.normalize_for_scoring` first (tags, case,
punctuation, equivalents), as plain WER does.
"""

from __future__ import annotations

import unicodedata
from collections import Counter

from .scoring import error_rate, normalize_for_scoring, script_of

# ---------- Devanagari folding ----------

_FOLD = str.maketrans(
    {
        "ं": None,  # anusvara
        "ँ": None,  # chandrabindu
        "़": None,  # nukta (NFKC splits precomposed nukta letters)
        "ॅ": None,  # candra e
        "ी": "ि",
        "ू": "ु",
        "ई": "इ",
        "ऊ": "उ",
        "ॉ": "ो",
        "ऑ": "ओ",
    }
)


def dev_fold(word: str) -> str:
    """Spelling-variant key for a Devanagari word: है/हैं, गये/गए, कीजिये/कीजिए match."""
    w = unicodedata.normalize("NFKC", word).translate(_FOLD)
    return w.replace("ये", "ए").replace("यि", "इ")


# ---------- cross-script skeletons ----------

_DEV_CONSONANT = {
    "क": "k", "ख": "k", "ग": "g", "घ": "g", "ङ": "n",
    "च": "c", "छ": "c", "ज": "g", "झ": "g", "ञ": "n",
    "ट": "t", "ठ": "t", "ड": "d", "ढ": "d", "ण": "n",
    "त": "t", "थ": "t", "द": "d", "ध": "d", "न": "n",
    "प": "p", "फ": "f", "ब": "b", "भ": "b", "म": "m",
    "य": "", "र": "r", "ल": "l", "व": "v",
    "श": "s", "ष": "s", "स": "s", "ह": "h",
}
_NASALS = set("नमणङञ")
_VIRAMA = "्"

# Latin spellings -> the same consonant classes. g, j and z share one class:
# English g is often /dʒ/ (manager = मैनेजर) and Hindi writes z as ज़.
_LATIN_DIGRAPHS = [
    ("tion", "sn"), ("sion", "sn"), ("ch", "c"), ("sh", "s"), ("th", "t"), ("ph", "f"),
    ("gh", "g"), ("kh", "k"), ("ck", "k"), ("qu", "kv"), ("x", "ks"),
]
_LATIN_SINGLE = {
    "b": "b", "d": "d", "f": "f", "g": "g", "h": "h", "j": "g", "k": "k", "l": "l", "m": "m", "n": "n",
    "p": "p", "q": "k", "r": "r", "s": "s", "t": "t", "v": "v", "w": "v", "z": "g",
}
_VOWELS = set("aeiouy")

# Short words whose skeleton is under 2 consonants; Devanagari side is folded.
_SHORT = {
    "ok": {"ओके", "ओक"}, "okay": {"ओके"}, "yes": {"यस", "येस"}, "no": {"नो"}, "sir": {"सर"},
    "hi": {"हाय", "हाई"}, "bye": {"बाय", "बाई"}, "so": {"सो"}, "ji": {"जी"},
}
_SHORT = {k: {dev_fold(v) for v in vs} for k, vs in _SHORT.items()}


def _collapse(s: str) -> str:
    return "".join(ch for i, ch in enumerate(s) if i == 0 or ch != s[i - 1])


def dev_skeleton(word: str) -> str:
    w = dev_fold(word)
    out = []
    for i, ch in enumerate(w):
        if ch in _NASALS and i + 1 < len(w) and w[i + 1] == _VIRAMA:
            continue  # half nasal before a consonant (सेन्टर = सेंटर)
        out.append(_DEV_CONSONANT.get(ch, ""))
    return _collapse("".join(out))


def latin_skeleton(word: str) -> str:
    w = word.casefold()
    # c is s before e/i/y (service, city), else k (call)
    w = "".join(("s" if w[i + 1 : i + 2] in ("e", "i", "y") else "k") if ch == "c" and w[i : i + 2] != "ch" else ch
                for i, ch in enumerate(w))
    for src, dst in _LATIN_DIGRAPHS:
        w = w.replace(src, dst.upper())  # upper case marks letters already mapped
    out = []
    for i, ch in enumerate(w):
        if ch.isupper():
            out.append(ch.lower())
            continue
        if ch in "nm" and i + 1 < len(w) and w[i + 1] not in _VOWELS and w[i + 1].isalpha():
            continue  # nasal before a consonant: Devanagari writes it as anusvara
        out.append(_LATIN_SINGLE.get(ch, ""))
    return _collapse("".join(out))


def same_word_across_scripts(a: str, b: str) -> bool:
    sa, sb = script_of(a), script_of(b)
    if {sa, sb} != {"devanagari", "latin"}:
        return False
    latin, dev = (a, b) if sa == "latin" else (b, a)
    if dev_fold(dev) in _SHORT.get(latin.casefold(), ()):
        return True
    skel = latin_skeleton(latin)
    return len(skel) >= 2 and skel == dev_skeleton(dev)


# ---------- numbers ----------

_HI_NUMBERS = (
    "शून्य 0|एक 1|दो 2|तीन 3|चार 4|पांच 5|पाँच 5|छह 6|छः 6|छे 6|सात 7|आठ 8|नौ 9|दस 10|"
    "ग्यारह 11|बारह 12|तेरह 13|चौदह 14|पंद्रह 15|पन्द्रह 15|सोलह 16|सत्रह 17|अठारह 18|उन्नीस 19|बीस 20|"
    "इक्कीस 21|बाईस 22|तेईस 23|चौबीस 24|पच्चीस 25|छब्बीस 26|सत्ताईस 27|अट्ठाईस 28|उनतीस 29|तीस 30|"
    "इकतीस 31|बत्तीस 32|तैंतीस 33|चौंतीस 34|पैंतीस 35|छत्तीस 36|सैंतीस 37|अड़तीस 38|उनतालीस 39|चालीस 40|"
    "इकतालीस 41|बयालीस 42|तैंतालीस 43|चवालीस 44|पैंतालीस 45|छियालीस 46|सैंतालीस 47|अड़तालीस 48|उनचास 49|पचास 50|"
    "इक्यावन 51|बावन 52|तिरेपन 53|चौवन 54|पचपन 55|छप्पन 56|सत्तावन 57|अट्ठावन 58|उनसठ 59|साठ 60|"
    "इकसठ 61|बासठ 62|तिरसठ 63|चौंसठ 64|पैंसठ 65|छियासठ 66|सड़सठ 67|अड़सठ 68|उनहत्तर 69|सत्तर 70|"
    "इकहत्तर 71|बहत्तर 72|तिहत्तर 73|चौहत्तर 74|पचहत्तर 75|छिहत्तर 76|सतहत्तर 77|अठहत्तर 78|उन्यासी 79|अस्सी 80|"
    "इक्यासी 81|बयासी 82|तिरासी 83|चौरासी 84|पचासी 85|छियासी 86|सत्तासी 87|अट्ठासी 88|नवासी 89|नब्बे 90|"
    "इक्यानवे 91|बानवे 92|तिरानवे 93|चौरानवे 94|पचानवे 95|छियानवे 96|सत्तानवे 97|अट्ठानवे 98|निन्यानवे 99|"
    "डेढ़ 1.5|ढाई 2.5"
)
_EN_NUMBERS = (
    "zero 0|one 1|two 2|three 3|four 4|five 5|six 6|seven 7|eight 8|nine 9|ten 10|eleven 11|twelve 12|"
    "thirteen 13|fourteen 14|fifteen 15|sixteen 16|seventeen 17|eighteen 18|nineteen 19|twenty 20|"
    "thirty 30|forty 40|fifty 50|sixty 60|seventy 70|eighty 80|ninety 90"
)
_MULTIPLIERS = {
    "सौ": 100, "हज़ार": 1000, "हजार": 1000, "लाख": 100_000, "करोड़": 10_000_000, "करोड": 10_000_000,
    "hundred": 100, "thousand": 1000, "lakh": 100_000, "lakhs": 100_000, "crore": 10_000_000,
    "crores": 10_000_000, "million": 1_000_000,
}
# साढ़े तीन = 3.5, सवा दो = 2.25, पौने चार = 3.75
_MODIFIERS = {"साढ़े": 0.5, "सवा": 0.25, "पौने": -0.25}


def _table(spec: str) -> dict[str, float]:
    return {word: float(value) for word, value in (item.split() for item in spec.split("|"))}


_NUM_KEY = {dev_fold(k): v for k, v in _table(_HI_NUMBERS).items()} | _table(_EN_NUMBERS)
_MULT_KEY = {dev_fold(k): v for k, v in _MULTIPLIERS.items()}
_MOD_KEY = {dev_fold(k): v for k, v in _MODIFIERS.items()}


def _number_token(tok: str) -> tuple[str, float] | None:
    if tok.isdigit():
        return ("num", float(int(tok)))
    key = dev_fold(tok)
    if key in _NUM_KEY:
        return ("num", _NUM_KEY[key])
    if key in _MULT_KEY:
        return ("mult", float(_MULT_KEY[key]))
    if key in _MOD_KEY:
        return ("mod", _MOD_KEY[key])
    return None


def _fmt(x: float) -> str:
    return str(int(x)) if x == int(x) else str(x)


def _run_value(run: list[tuple[str, float]]) -> str:
    """One canonical string for a run of number tokens.

    Multipliers combine (दो लाख पचास हज़ार = 250000); English tens take a unit
    (twenty five = 25); other adjacent numbers are read digit by digit, as in
    phone numbers (नौ आठ सात = 987, and "9 8 7" too).
    """
    pieces: list[str] = []
    total, cur, mod = 0.0, None, 0.0
    for kind, val in run:
        if kind == "mod":
            mod = val
        elif kind == "num":
            val += mod
            mod = 0.0
            if cur is None:
                cur = val
            elif 20 <= cur < 100 and cur % 10 == 0 and val < 10:
                cur += val
            else:
                pieces.append(_fmt(total + cur))
                total, cur = 0.0, val
        else:  # multiplier
            base = cur if cur is not None else 1.0 + mod
            mod = 0.0
            total += base * val
            cur = None
    if cur is not None or total:
        pieces.append(_fmt(total + (cur or 0.0)))
    return "".join(pieces) or "0"


def canonical_tokens(tokens: list[str]) -> list[tuple[str, str]]:
    """(canonical, surface) pairs; a run of number words/digits becomes one token."""
    out: list[tuple[str, str]] = []
    run: list[tuple[str, float]] = []
    surface: list[str] = []

    def flush():
        if run:
            out.append(("#" + _run_value(run), " ".join(surface)))
            run.clear()
            surface.clear()

    for tok in tokens:
        num = _number_token(tok)
        if num is None:
            flush()
            out.append((tok, tok))
        else:
            run.append(num)
            surface.append(tok)
    flush()
    return out


# ---------- alignment ----------


def match_rule(r: tuple[str, str], h: tuple[str, str]) -> str | None:
    """How reference token `r` matches hypothesis token `h`, or None."""
    if r[0] == h[0]:
        return "exact" if r[1] == h[1] else "number"
    a, b = r[0], h[0]
    if a.startswith("#") or b.startswith("#"):
        return None
    if script_of(a) == script_of(b) == "devanagari" and dev_fold(a) == dev_fold(b):
        return "devanagari_variant"
    if same_word_across_scripts(a, b):
        return "cross_script"
    return None


def _joined_equal(a: str, b: str) -> bool:
    return a == b or (script_of(a) == script_of(b) == "devanagari" and dev_fold(a) == dev_fold(b))


def lenient_align(ref: list[tuple[str, str]], hyp: list[tuple[str, str]]) -> tuple[int, Counter, list[tuple[str, str]]]:
    """Minimum edit cost with equivalent pairs free; also which rules matched."""
    n, m = len(ref), len(hyp)
    inf = float("inf")
    cost = [[inf] * (m + 1) for _ in range(n + 1)]
    back: list[list[tuple | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0
    for i in range(n + 1):
        for j in range(m + 1):
            c = cost[i][j]
            if c == inf:
                continue
            steps = []
            if i < n:
                steps.append((i + 1, j, 1, None))  # deletion
            if j < m:
                steps.append((i, j + 1, 1, None))  # insertion
            if i < n and j < m:
                rule = match_rule(ref[i], hyp[j])
                steps.append((i + 1, j + 1, 0 if rule else 1, rule))
            if i < n and j + 1 < m and _joined_equal(ref[i][0], hyp[j][0] + hyp[j + 1][0]):
                steps.append((i + 1, j + 2, 0, "spacing"))
            if i + 1 < n and j < m and _joined_equal(ref[i][0] + ref[i + 1][0], hyp[j][0]):
                steps.append((i + 2, j + 1, 0, "spacing"))
            for ni, nj, step_cost, rule in steps:
                if c + step_cost < cost[ni][nj]:
                    cost[ni][nj] = c + step_cost
                    back[ni][nj] = (i, j, rule)
    rules, pairs = Counter(), []
    i, j = n, m
    while (i, j) != (0, 0):
        pi, pj, rule = back[i][j]
        if rule and rule != "exact":
            rules[rule] += 1
            if rule == "cross_script":
                pairs.append((ref[pi][1], hyp[pj][1]))
        i, j = pi, pj
    return int(cost[n][m]), rules, pairs


def lenient_scores(
    references: list[str],
    hypotheses: list[str],
    equivalents: dict[str, str] | None = None,
) -> dict:
    """Corpus lenient WER, per-utterance (errors, reference words), and rule counts."""
    if len(references) != len(hypotheses):
        raise ValueError(f"{len(references)} references but {len(hypotheses)} hypotheses")
    per: list[tuple[int, int] | None] = []
    rules, pairs = Counter(), Counter()
    for r, h in zip(references, hypotheses):
        rt = normalize_for_scoring(r, equivalents).split()
        if not rt:
            per.append(None)
            continue
        ht = normalize_for_scoring(h, equivalents).split()
        errors, used, matched = lenient_align(canonical_tokens(rt), canonical_tokens(ht))
        per.append((errors, len(rt)))
        rules.update(used)
        pairs.update(matched)
    return {
        "lenient_wer": error_rate(per),
        "per_utterance": per,
        "matches_by_rule": dict(rules),
        "cross_script_pairs": pairs.most_common(200),
    }


# ---------- semdist ----------


class SemDist:
    """1 - cosine(LaBSE(ref), LaBSE(hyp)); embeddings are cached by text, so the
    references are encoded once across candidates."""

    def __init__(self, model_name: str = "sentence-transformers/LaBSE", device: str | None = None, batch_size: int = 256):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        self.batch_size = batch_size
        self._cache: dict[str, object] = {}

    def _embed(self, texts: list[str]):
        new = sorted({t for t in texts if t not in self._cache})
        if new:
            vecs = self.model.encode(new, batch_size=self.batch_size, normalize_embeddings=True, convert_to_numpy=True)
            self._cache.update(zip(new, vecs))
        return [self._cache[t] for t in texts]

    def score(self, references: list[str], hypotheses: list[str], equivalents: dict[str, str] | None = None) -> list[float | None]:
        refs = [normalize_for_scoring(r, equivalents) for r in references]
        hyps = [normalize_for_scoring(h, equivalents) for h in hypotheses]
        keep = [i for i, r in enumerate(refs) if r]
        er = self._embed([refs[i] for i in keep])
        eh = self._embed([hyps[i] for i in keep])
        out: list[float | None] = [None] * len(refs)
        for i, a, b in zip(keep, er, eh):
            out[i] = float(1.0 - (a * b).sum())
        return out


def semdist_summary(values: list[float | None]) -> dict:
    v = sorted(x for x in values if x is not None)
    if not v:
        return {"semdist_mean": None, "semdist_p90": None}
    return {"semdist_mean": sum(v) / len(v), "semdist_p90": v[int(0.9 * (len(v) - 1))]}
