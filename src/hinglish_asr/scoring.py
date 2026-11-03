"""WER/CER for Hinglish, compatible with Hush-Eval's scoring.

`normalize_for_scoring` reproduces Hush-Eval's `metrics.normalize` (NFKC,
casefold, drop Unicode punctuation/symbol/control categories, equivalence map)
so baseline numbers here are comparable with the numbers already reported
there. Two additions: the model's `<xx-XX>` language tags are stripped first,
and errors are split by script so a model that only changes how it writes
English words shows up as such.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import jiwer

# Hush-Eval writes proposed-but-unreviewed groups with this exact source; they
# must not affect any score.
PENDING = "auto — REVIEW ME"

LANG_TAG = re.compile(r"<[a-z]{2}-[A-Z]{2}>")


def _base(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").casefold()
    kept = "".join(" " if unicodedata.category(ch)[0] in "PSC" else ch for ch in text)
    return " ".join(kept.split())


def normalize_for_scoring(text: str, equivalents: dict[str, str] | None = None) -> str:
    text = _base(LANG_TAG.sub(" ", text or ""))
    if equivalents:
        text = " ".join(equivalents.get(w, w) for w in text.split())
    return text


def load_equivalents(path: str | Path) -> dict[str, str]:
    """`{token: canonical}` from a Hush-Eval equivalents.json, reviewed groups only."""
    groups = json.loads(Path(path).read_text(encoding="utf-8")).get("groups", [])
    out: dict[str, str] = {}
    for group in groups:
        if group.get("source") == PENDING:
            continue
        canonical = _base(group.get("canonical", ""))
        if not canonical:
            continue
        for token in [canonical, *group.get("variants", [])]:
            if token := _base(token):
                out[token] = canonical
    return out


def script_of(word: str) -> str:
    if any("ऀ" <= ch <= "ॿ" for ch in word):
        return "devanagari"
    if any("a" <= ch <= "z" for ch in word.casefold()):
        return "latin"
    return "other"


def score_corpus(
    references: list[str],
    hypotheses: list[str],
    equivalents: dict[str, str] | None = None,
) -> dict:
    """Corpus-level (edit-weighted) WER and CER, with errors split by script.

    Substitutions and deletions count against the reference word's script,
    insertions against the inserted word's script.
    """
    if len(references) != len(hypotheses):
        raise ValueError(f"{len(references)} references but {len(hypotheses)} hypotheses")
    refs, hyps, skipped = [], [], 0
    for r, h in zip(references, hypotheses):
        r = normalize_for_scoring(r, equivalents)
        if not r:
            skipped += 1
            continue
        refs.append(r)
        hyps.append(normalize_for_scoring(h, equivalents))

    out = {"utterances": len(refs), "skipped_empty_refs": skipped}
    if not refs:
        return out

    words = jiwer.process_words(refs, hyps)
    ref_words = words.hits + words.substitutions + words.deletions
    errors = {"devanagari": 0, "latin": 0, "other": 0}
    ref_counts = {"devanagari": 0, "latin": 0, "other": 0}
    for ref_toks, hyp_toks, chunks in zip(words.references, words.hypotheses, words.alignments):
        for tok in ref_toks:
            ref_counts[script_of(tok)] += 1
        for c in chunks:
            if c.type in ("substitute", "delete"):
                for tok in ref_toks[c.ref_start_idx : c.ref_end_idx]:
                    errors[script_of(tok)] += 1
            elif c.type == "insert":
                for tok in hyp_toks[c.hyp_start_idx : c.hyp_end_idx]:
                    errors[script_of(tok)] += 1

    out.update(
        ref_words=ref_words,
        wer=words.wer,
        cer=jiwer.cer(refs, hyps),
        substitutions=words.substitutions,
        deletions=words.deletions,
        insertions=words.insertions,
    )
    for script in ("devanagari", "latin"):
        out[f"{script}_ref_words"] = ref_counts[script]
        out[f"{script}_errors"] = errors[script]
        out[f"wer_{script}"] = errors[script] / ref_counts[script] if ref_counts[script] else None
    return out


def utterance_errors(
    references: list[str],
    hypotheses: list[str],
    equivalents: dict[str, str] | None = None,
) -> list[tuple[int, int] | None]:
    """(word errors, reference words) per utterance, same normalization as
    `score_corpus`; None where the normalized reference is empty (skipped there too)."""
    if len(references) != len(hypotheses):
        raise ValueError(f"{len(references)} references but {len(hypotheses)} hypotheses")
    out: list[tuple[int, int] | None] = []
    keep, refs, hyps = [], [], []
    for i, (r, h) in enumerate(zip(references, hypotheses)):
        r = normalize_for_scoring(r, equivalents)
        out.append(None)
        if r:
            keep.append(i)
            refs.append(r)
            hyps.append(normalize_for_scoring(h, equivalents))
    if not refs:
        return out
    words = jiwer.process_words(refs, hyps)
    for i, ref_toks, chunks in zip(keep, words.references, words.alignments):
        errors = 0
        for c in chunks:
            if c.type in ("substitute", "delete"):
                errors += c.ref_end_idx - c.ref_start_idx
            elif c.type == "insert":
                errors += c.hyp_end_idx - c.hyp_start_idx
        out[i] = (errors, len(ref_toks))
    return out


def error_rate(pairs) -> float | None:
    pairs = [p for p in pairs if p is not None]
    words = sum(n for _, n in pairs)
    return sum(e for e, _ in pairs) / words if words else None


def paired_bootstrap(
    groups: list[str],
    a: list[tuple[int, int] | None],
    b: list[tuple[int, int] | None],
    resamples: int = 1000,
    seed: int = 0,
) -> dict:
    """Error rate of `b` minus `a`, with a 95% interval from resampling whole groups.

    A customer's utterances are not independent (same voice, line, topics), so
    resampling utterances one by one would make the interval too narrow.
    """
    import numpy as np

    totals: dict[str, list[int]] = {}
    for g, x, y in zip(groups, a, b):
        if x is None or y is None:
            continue
        t = totals.setdefault(g, [0, 0, 0, 0])
        t[0] += x[0]
        t[1] += x[1]
        t[2] += y[0]
        t[3] += y[1]
    if not totals:
        return {"delta": None, "ci95": None, "groups": 0, "resamples": resamples}
    arr = np.array(list(totals.values()), dtype=float)
    n = len(arr)
    rng = np.random.default_rng(seed)
    deltas = []
    for start in range(0, resamples, 100):  # chunks keep the count matrix small
        counts = rng.multinomial(n, np.full(n, 1 / n), size=min(100, resamples - start))
        s = counts @ arr
        deltas.append(s[:, 2] / s[:, 3] - s[:, 0] / s[:, 1])
    deltas = np.concatenate(deltas)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    point = arr[:, 2].sum() / arr[:, 3].sum() - arr[:, 0].sum() / arr[:, 1].sum()
    return {"delta": float(point), "ci95": [float(lo), float(hi)], "groups": n, "resamples": resamples}
