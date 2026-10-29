"""Source utterances -> 16 kHz WAVs + NeMo manifests (+ report.json).

    # Local WAVs listed in a JSONL manifest (audio_filepath, text[, offset, duration, group])
    uv run python -m hinglish_asr.prepare --source jsonl --input calls.jsonl \\
        --out data/prepared/calls --val-frac 0.01 --test-frac 0.01 --workers 32

    # An HF parquet dataset, minus the Hush-Eval held-out rows
    uv run python -m hinglish_asr.prepare --source hf \\
        --dataset kapturecx/Hindi-Transcription-DS --revision <40-char sha> \\
        --exclude-hush-eval ../Hush-Eval/data/test.json \\
        --group-pattern '^IISc_VaaniProject_[A-Z]_[^_]+_[^_]+_([^_]+)_' \\
        --out data/prepared/hindi_ds --val-frac 0.01 --workers 32

    # A Hush-Eval frozen test list, as one test manifest (baseline input)
    uv run python -m hinglish_asr.prepare --source hush-eval --input ../Hush-Eval/data/test.json \\
        --limit 1000 --out data/prepared/hindi_test --single-split test

Every manifest row carries `target_lang` (read by the prompt model's training
dataloader) and `lang` (read by NeMo's transcribe scripts). Rerunning reuses
WAVs already written, so an interrupted run resumes.
"""

from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

from .audio import AudioError, convert_audio
from .scoring import script_of
from .sources import ParquetSource, Record, read_hush_eval_manifest, read_jsonl_source
from .split import assign_split, sample_rows_by_row_group
from .textnorm import normalize_target


@dataclass(frozen=True)
class _Cfg:
    audio_dir: str
    target_lang: str
    val_frac: float
    test_frac: float
    single_split: str | None
    min_duration: float
    max_duration: float
    seed: str


def _sidecar(wav: Path) -> Path:
    """Written after the WAV: its existence means the conversion finished.

    It records what a resumed run cannot recover from the 16 kHz WAV (the
    source sample rate), so finished rows never need their source audio again.
    """
    return wav.with_suffix(".json")


def _process(job: tuple[Record, _Cfg]) -> tuple:
    rec, cfg = job
    text = normalize_target(rec.text)
    if not text:
        return ("drop", "empty_text", rec.id, "")
    dst = Path(cfg.audio_dir) / f"{rec.id}.wav"
    meta = _sidecar(dst)
    try:
        if meta.exists():
            done = json.loads(meta.read_text())
            duration, orig_sr = done["duration"], done["orig_sr"]
        else:
            if dst.exists():  # written before sidecars existed
                duration = sf.info(dst).duration
                orig_sr = sf.info(io.BytesIO(rec.audio) if isinstance(rec.audio, bytes) else rec.audio).samplerate
            else:
                info = convert_audio(rec.audio, dst, rec.offset, rec.duration)
                duration, orig_sr = info.duration, info.orig_sr
            meta.write_text(json.dumps({"duration": duration, "orig_sr": orig_sr}))
    except (AudioError, sf.LibsndfileError, OSError, TypeError) as exc:
        return ("drop", "bad_audio", rec.id, str(exc))
    if duration < cfg.min_duration or duration > cfg.max_duration:
        dst.unlink(missing_ok=True)
        meta.unlink(missing_ok=True)
        reason = "too_short" if duration < cfg.min_duration else "too_long"
        return ("drop", reason, rec.id, f"{duration:.2f}s")
    split = cfg.single_split or assign_split(rec.group, cfg.val_frac, cfg.test_frac, cfg.seed)
    row = {
        "audio_filepath": str(dst.resolve()),
        "duration": round(duration, 3),
        "text": text,
        "target_lang": cfg.target_lang,
        "lang": cfg.target_lang,
        "id": rec.id,
        "group": rec.group,
        "orig_sr": orig_sr,
    }
    if rec.row is not None:
        row["row"] = rec.row
    return ("ok", split, row, text != rec.text)


def prepare(
    records: Iterable[Record],
    out_dir: str | Path,
    *,
    target_lang: str = "hi-IN",
    val_frac: float = 0.0,
    test_frac: float = 0.0,
    single_split: str | None = None,
    min_duration: float = 0.1,
    max_duration: float = 20.0,
    workers: int = 1,
    seed: str = "hinglish-v1",
) -> dict:
    out_dir = Path(out_dir)
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)
    cfg = _Cfg(str(out_dir / "audio"), target_lang, val_frac, test_frac, single_split, min_duration, max_duration, seed)
    jobs = ((rec, cfg) for rec in records)

    counts, seconds, sr_seconds, groups = Counter(), defaultdict(float), defaultdict(float), defaultdict(set)
    dropped, drop_examples = Counter(), defaultdict(list)
    text_changed = mixed = latin_words = words = 0
    writers = {}
    try:
        if workers > 1:
            pool = mp.get_context("spawn").Pool(workers)
            results = pool.imap(_process, jobs, chunksize=8)
        else:
            pool, results = None, map(_process, jobs)
        for res in results:
            if res[0] == "drop":
                _, reason, rid, detail = res
                dropped[reason] += 1
                if len(drop_examples[reason]) < 5:
                    drop_examples[reason].append({"id": rid, "detail": detail})
                continue
            _, split, row, changed = res
            if split not in writers:
                writers[split] = (out_dir / f"{split}_manifest.jsonl").open("w", encoding="utf-8")
            writers[split].write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[split] += 1
            seconds[split] += row["duration"]
            sr_seconds[str(row["orig_sr"])] += row["duration"]
            groups[split].add(row["group"])
            text_changed += changed
            scripts = [script_of(w) for w in row["text"].split()]
            mixed += "latin" in scripts and "devanagari" in scripts
            latin_words += scripts.count("latin")
            words += len(scripts)
    finally:
        for fh in writers.values():
            fh.close()
        if workers > 1:
            pool.close()
            pool.join()

    report = {
        "utterances": dict(counts),
        "hours": {k: round(v / 3600, 3) for k, v in seconds.items()},
        "hours_by_orig_sr": {k: v / 3600 for k, v in sr_seconds.items()},
        "groups": {k: len(v) for k, v in groups.items()},
        "dropped": dict(dropped),
        "drop_examples": dict(drop_examples),
        "text_changed_by_normalization": text_changed,
        "mixed_script_utterances": mixed,
        "latin_word_share": latin_words / words if words else None,
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _hush_eval_rows(path: str) -> set[int]:
    _, samples = read_hush_eval_manifest(path)
    return {int(s["row_index"]) for s in samples}


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["jsonl", "hf", "hush-eval"], required=True)
    p.add_argument("--input", help="JSONL manifest (jsonl) or Hush-Eval manifest (hush-eval)")
    p.add_argument("--dataset", help="HF dataset repo (hf)")
    p.add_argument("--revision", help="HF commit sha (hf); pin it so row indices stay meaningful")
    p.add_argument("--split", default="train", help="HF split name (hf)")
    p.add_argument("--parquet", nargs="+", help="local parquet file(s) instead of fetching from the HF Hub")
    p.add_argument("--audio-column", default="audio")
    p.add_argument("--text-column", default="text")
    p.add_argument("--id-prefix", help="id prefix for parquet rows (default: dataset name)")
    p.add_argument("--group-pattern", help="regex; first capture group on the audio file name = speaker/call")
    p.add_argument("--exclude-hush-eval", help="Hush-Eval manifest whose rows must not be used (parquet sources)")
    p.add_argument("--limit", type=int, help="hush-eval: at most N rows, drawn from few parquet row groups")
    p.add_argument("--seed", type=int, default=0, help="hush-eval --limit sampling seed")
    p.add_argument("--out", required=True)
    p.add_argument("--target-lang", default="hi-IN")
    p.add_argument("--val-frac", type=float, default=0.0)
    p.add_argument("--test-frac", type=float, default=0.0)
    p.add_argument("--single-split", help="put every utterance in this split (e.g. test)")
    p.add_argument("--min-duration", type=float, default=0.1)
    p.add_argument("--max-duration", type=float, default=20.0, help="the base model was pre-trained on <= 20 s")
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args(argv)

    if args.source == "jsonl":
        if not args.input:
            p.error("--source jsonl needs --input")
        records = read_jsonl_source(args.input, args.group_pattern)
    else:
        rows, references = None, None
        dataset, revision, split = args.dataset, args.revision, args.split
        if args.source == "hush-eval":
            if not args.input:
                p.error("--source hush-eval needs --input")
            meta, samples = read_hush_eval_manifest(args.input)
            if meta.get("type") != "huggingface":
                p.error("only HF-backed Hush-Eval manifests are supported")
            dataset, revision, split = meta["dataset"], meta["revision"], meta["split"]
            references = {int(s["row_index"]): s for s in samples}
            rows = sorted(references)
        if args.parquet:
            source = ParquetSource(args.parquet)
        else:
            if not (dataset and revision):
                p.error("--source hf needs --dataset and --revision (or --parquet)")
            source = ParquetSource.from_hf(dataset, revision, split)
        if args.limit is not None and rows is not None:
            rows = sample_rows_by_row_group(rows, source.row_group_bounds(), args.limit, args.seed)
        exclude = _hush_eval_rows(args.exclude_hush_eval) if args.exclude_hush_eval else set()
        prefix = args.id_prefix or re.sub(r"\W+", "_", (dataset or "row").rsplit("/", 1)[-1])
        audio_dir = Path(args.out) / "audio"
        records = source.records(
            converted=lambda rid: _sidecar(audio_dir / f"{rid}.wav").exists(),
            rows=rows,
            exclude=exclude,
            audio_column=args.audio_column,
            text_column=args.text_column,
            group_pattern=args.group_pattern,
            id_prefix=prefix,
            references=references,
        )

    report = prepare(
        records,
        args.out,
        target_lang=args.target_lang,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        single_split=args.single_split,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        workers=args.workers,
    )
    summary = {k: report[k] for k in ("utterances", "hours", "groups", "dropped")}
    print(json.dumps(summary, ensure_ascii=False, indent=2), file=sys.stderr)
    return report


if __name__ == "__main__":
    main()
