"""Keep the empty-target candidates a second model also hears nothing in.

    uv run --extra nemo python -m hinglish_asr.verify_empty --data /data-g/uday/calls-spoken

`convert` (with `empty_targets` in the data config) sets aside clips the vendor
left empty: empty_candidates.jsonl and empty/audio/. A vendor's empty
transcript is not proof of silence (they miss quiet or unclear speech), and VAD
cannot tell (every chunk is VAD-cut), so this decodes each candidate with the
base model at the deployment lookahead and the row's own prompt, and keeps it
only if the model writes nothing either. Two independent systems agreeing is
the verification.

Writes, next to convert's output:

    train_empty_manifest.jsonl    verified training clips: a second input_cfg entry
    val_empty_manifest.jsonl      verified held-out clips: pass to benchmark next to
    test_empty_manifest.jsonl       the split's manifest; it reports the hallucination
    client_test_empty_manifest.jsonl  rate on them (WER skips empty references)
    empty_verify.json             counts per split, and what the model wrote on the
                                  clips it rejected (to check the rule)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from .scoring import normalize_for_scoring


def split_verified(rows: list[dict], hyps: list[str]) -> tuple[dict[str, list[dict]], list[tuple[dict, str]]]:
    """Candidates the model left empty, by the split they belong to; and the rejected ones with what it wrote."""
    kept: dict[str, list[dict]] = defaultdict(list)
    rejected = []
    for row, hyp in zip(rows, hyps):
        if normalize_for_scoring(hyp):
            rejected.append((row, hyp))
            continue
        out = {k: v for k, v in row.items() if k != "for_split"}
        kept[row["for_split"]].append(out)
    return kept, rejected


def summary(rows: list[dict], kept: dict[str, list[dict]], rejected: list[tuple[dict, str]]) -> dict:
    candidates = Counter(r["for_split"] for r in rows)
    out = {
        split: {
            "candidates": n,
            "verified": len(kept.get(split, [])),
            "verified_share": len(kept.get(split, [])) / n,
            "verified_hours": round(sum(r["duration"] for r in kept.get(split, [])) / 3600, 3),
        }
        for split, n in sorted(candidates.items())
    }
    words = Counter(w for _, h in rejected for w in normalize_for_scoring(h).split())
    return {
        "splits": out,
        "rejected_top_words": words.most_common(30),
        "rejected_examples": [{"id": r["id"], "lang": r["target_lang"], "model_wrote": h} for r, h in rejected[:50]],
    }


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="convert's --out")
    p.add_argument("--model", help="base .nemo (default: base_model in --train-config)")
    p.add_argument("--train-config", default="configs/train.yaml")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)

    from .baseline import load_model, transcribe_offline
    from .benchmark import transcribe_by_prompt

    data = Path(args.data)
    with (data / "empty_candidates.jsonl").open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    if not rows:
        sys.exit(f"no candidates in {data / 'empty_candidates.jsonl'}: is empty_targets set in the data config?")
    base = args.model or yaml.safe_load(Path(args.train_config).read_text())["base_model"]
    model = load_model(base, args.device)
    print(f"decoding {len(rows)} candidates with {base}", file=sys.stderr)
    hyps = transcribe_by_prompt(transcribe_offline, model, rows, args.batch_size)

    kept, rejected = split_verified(rows, hyps)
    for split in sorted({r["for_split"] for r in rows}):
        with (data / f"{split}_empty_manifest.jsonl").open("w", encoding="utf-8") as fh:
            for r in kept.get(split, []):
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    report = {"model": base, **summary(rows, kept, rejected)}
    (data / "empty_verify.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report["splits"], indent=1), file=sys.stderr)
    return report


if __name__ == "__main__":
    main()
