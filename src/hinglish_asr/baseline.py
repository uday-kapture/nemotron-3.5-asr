"""Baseline the unmodified model on a prepared test manifest, before any training.

    uv run --extra nemo python -m hinglish_asr.baseline \\
        --manifest data/prepared/hindi_test/test_manifest.jsonl \\
        --equivalents ../Hush-Eval/data/equivalents.json \\
        --out reports/baseline/hindi_test

Runs every (mode x prompt x lookahead) combination and writes:

    predictions/<mode>_<prompt>_<ms>ms.jsonl   NeMo fields: text (ref), pred_text (raw, tags kept)
    summary.json, summary.md                   WER/CER, per-script WER, tag emission, script share, speed

What it answers (research doc §10.2): which language prompt suits Hinglish
(`hi-IN` vs `auto`), how the model writes English words (Latin vs
Devanagari) and numbers, whether it emits `<xx-XX>` tags, and how accuracy
moves with latency. Scoring matches Hush-Eval (see `scoring`).

"offline" runs model.transcribe at the given lookahead; for a cache-aware
model it should equal true streaming at the same att_context_size. "streaming"
runs the cache-aware loop of NeMo's speech_to_text_cache_aware_streaming_infer.py
to confirm that on your audio. Both run in float32: NeMo does not support
other dtypes for cache-aware streaming.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

from .scoring import LANG_TAG, load_equivalents, normalize_for_scoring, score_corpus, script_of

DEFAULT_MODEL = "nvidia/nemotron-3.5-asr-streaming-0.6b"


def parse_att_context(spec: str) -> list[int]:
    parts = spec.split(",")
    if len(parts) != 2:
        raise ValueError(f"att_context_size must be LEFT,RIGHT (e.g. 56,0), got {spec!r}")
    return [int(p) for p in parts]


def latency_ms(att_context: list[int]) -> int:
    return (att_context[1] + 1) * 80


def output_stats(texts: list[str]) -> dict:
    tags, tagged, empty, words, latin = Counter(), 0, 0, 0, 0
    for text in texts:
        found = LANG_TAG.findall(text)
        tags.update(found)
        tagged += bool(found)
        toks = normalize_for_scoring(text).split()
        empty += not toks
        words += len(toks)
        latin += sum(script_of(t) == "latin" for t in toks)
    n = len(texts) or 1
    return {
        "tag_rate": tagged / n,
        "tags": dict(tags),
        "empty_rate": empty / n,
        "latin_word_share": latin / words if words else None,
    }


def summarize(
    config: dict,
    refs: list[str],
    hyps: list[str],
    equivalents: dict[str, str] | None,
    audio_seconds: float,
    wall_seconds: float,
) -> dict:
    row = dict(config)
    row["latency_ms"] = latency_ms(config["att_context"])
    row.update(score_corpus(refs, hyps, equivalents))
    row.update(output_stats(hyps))
    row["ref_latin_word_share"] = output_stats(refs)["latin_word_share"]
    row["rtfx"] = audio_seconds / wall_seconds if wall_seconds else None
    return row


# ---------- model ----------


def load_model(name: str, device: str):
    import torch
    from nemo.collections.asr.models import ASRModel
    from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
    from omegaconf import OmegaConf

    if Path(name).is_file():
        model = ASRModel.restore_from(name, map_location=device)
    else:
        model = ASRModel.from_pretrained(name, map_location=device)
    # Both NVIDIA eval scripts (transcribe_speech.py and the cache-aware
    # streaming script) decode with RNNTDecodingConfig(fused_batch_size=-1).
    model.change_decoding_strategy(OmegaConf.structured(RNNTDecodingConfig(fused_batch_size=-1)))
    # Keep <xx-XX> tags in the raw output: whether the model emits them is a
    # finding. Scoring strips them.
    model.decoding.set_strip_lang_tags(False)
    return model.to(device=device, dtype=torch.float32).eval()


def _texts(hyps) -> list[str]:
    if isinstance(hyps, tuple):  # (best, all) from some RNNT transcribe paths
        hyps = hyps[0]
    return [h.text if hasattr(h, "text") else str(h) for h in hyps]


def transcribe_offline(model, paths: list[str], target_lang: str, att_context: list[int], batch_size: int) -> list[str]:
    model.encoder.set_default_att_context_size(att_context)
    return _texts(model.transcribe(paths, batch_size=batch_size, target_lang=target_lang, verbose=False))


def transcribe_streaming(
    model,
    paths: list[str],
    target_lang: str,
    att_context: list[int],
    batch_size: int,
    pad_and_drop_preencoded: bool = False,
) -> list[str]:
    import torch
    from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

    model.encoder.set_default_att_context_size(att_context)
    model.set_inference_prompt(target_lang)
    buffer = CacheAwareStreamingAudioBuffer(
        model=model, online_normalization=False, pad_and_drop_preencoded=pad_and_drop_preencoded
    )
    out: list[str] = []
    for i in range(0, len(paths), batch_size):
        buffer.reset_buffer()
        for path in paths[i : i + batch_size]:
            buffer.append_audio_file(path, stream_id=-1)
        # Mirrors perform_streaming() in speech_to_text_cache_aware_streaming_infer.py.
        cache_last_channel, cache_last_time, cache_last_channel_len = model.encoder.get_initial_cache_state(
            batch_size=len(buffer.streams_length)
        )
        previous_hypotheses = pred_out = texts = None
        for step, (chunk_audio, chunk_lengths) in enumerate(iter(buffer)):
            drop = 0 if step == 0 and not pad_and_drop_preencoded else model.encoder.streaming_cfg.drop_extra_pre_encoded
            with torch.inference_mode():
                (
                    pred_out,
                    texts,
                    cache_last_channel,
                    cache_last_time,
                    cache_last_channel_len,
                    previous_hypotheses,
                ) = model.conformer_stream_step(
                    processed_signal=chunk_audio,
                    processed_signal_length=chunk_lengths,
                    cache_last_channel=cache_last_channel,
                    cache_last_time=cache_last_time,
                    cache_last_channel_len=cache_last_channel_len,
                    keep_all_outputs=buffer.is_buffer_empty(),
                    previous_hypotheses=previous_hypotheses,
                    previous_pred_out=pred_out,
                    drop_extra_pre_encoded=drop,
                    return_transcription=True,
                )
        out.extend(_texts(texts))
    return out


# ---------- reporting ----------


def _pct(x) -> str:
    return "–" if x is None else f"{100 * x:.1f}%"


def render_markdown(rows: list[dict], meta: dict) -> str:
    lines = [
        f"# Baseline: {meta['model']}",
        "",
        f"- Manifest: `{meta['manifest']}` ({meta['utterances']} utterances, {meta['audio_hours']:.2f} h)",
        f"- Equivalents map: {meta['equivalents'] or 'none'}",
        "- WER/CER: corpus-level, Hush-Eval normalization (NFKC, casefold, no punctuation), language tags stripped.",
        "- Latin share: fraction of words written in Latin script (reference share in the last column).",
        "",
        "| mode | prompt | latency | WER | CER | WER Devanagari | WER Latin | tags emitted | Latin share | ref Latin share | RTFx |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        rtfx = "–" if r.get("rtfx") is None else f"{r['rtfx']:.0f}"
        lines.append(
            f"| {r['mode']} | {r['target_lang']} | {r['latency_ms']} ms | {_pct(r.get('wer'))} | {_pct(r.get('cer'))} "
            f"| {_pct(r.get('wer_devanagari'))} | {_pct(r.get('wer_latin'))} | {_pct(r['tag_rate'])} "
            f"| {_pct(r['latin_word_share'])} | {_pct(r['ref_latin_word_share'])} | {rtfx} |"
        )
    return "\n".join(lines) + "\n"


def log_to_mlflow(experiment: str, run_name: str | None, meta: dict, rows: list[dict], out_dir: Path) -> None:
    import mlflow

    if os.environ.get("MLFLOW_TRACKING_USERNAME") and os.environ.get("MLFLOW_TRACKING_TOKEN"):
        # MLflow sends Basic auth when username+password are set and never the token.
        print("warning: both MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_TOKEN are set; "
              "MLflow will use Basic auth only", file=sys.stderr)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("stage", "baseline")
        mlflow.log_params({k: meta[k] for k in ("model", "manifest", "utterances", "equivalents")})
        mlflow.log_param("audio_hours", round(meta["audio_hours"], 3))
        for r in rows:
            key = f"{r['mode']}/{r['target_lang']}/{r['latency_ms']}ms"
            for metric in ("wer", "cer", "wer_devanagari", "wer_latin", "tag_rate", "latin_word_share", "rtfx"):
                if r.get(metric) is not None:
                    mlflow.log_metric(f"{metric}/{key}", r[metric])
        mlflow.log_artifacts(str(out_dir))


def main(argv: list[str] | None = None) -> list[dict]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True, help="NeMo JSONL from hinglish_asr.prepare")
    p.add_argument("--out", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL, help="HF model id or local .nemo")
    p.add_argument("--target-langs", nargs="+", default=["hi-IN", "auto"])
    p.add_argument("--att-context-sizes", nargs="+", default=["56,0", "56,3", "56,13"], help="80/320/1120 ms")
    p.add_argument("--modes", nargs="+", choices=["offline", "streaming"], default=["offline"])
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit", type=int, help="only the first N manifest rows")
    p.add_argument("--equivalents", help="Hush-Eval equivalents.json (reviewed groups are applied to both sides)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--pad-and-drop-preencoded", action="store_true", help="streaming mode option (NeMo script flag)")
    p.add_argument("--mlflow-experiment", help="also log to MLflow (MLFLOW_TRACKING_URI etc. from the environment)")
    p.add_argument("--mlflow-run-name")
    args = p.parse_args(argv)

    with open(args.manifest, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    rows = rows[: args.limit] if args.limit else rows
    paths = [r["audio_filepath"] for r in rows]
    refs = [r["text"] for r in rows]
    audio_seconds = sum(r["duration"] for r in rows)
    equivalents = load_equivalents(args.equivalents) if args.equivalents else None
    contexts = [parse_att_context(c) for c in args.att_context_sizes]

    out_dir = Path(args.out)
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
    model = load_model(args.model, args.device)
    meta = {
        "model": args.model,
        "manifest": args.manifest,
        "utterances": len(rows),
        "audio_hours": audio_seconds / 3600,
        "equivalents": args.equivalents,
    }

    results = []
    for mode in args.modes:
        for lang in args.target_langs:
            for ctx in contexts:
                t0 = time.perf_counter()
                if mode == "offline":
                    hyps = transcribe_offline(model, paths, lang, ctx, args.batch_size)
                else:
                    hyps = transcribe_streaming(model, paths, lang, ctx, args.batch_size, args.pad_and_drop_preencoded)
                wall = time.perf_counter() - t0
                name = f"{mode}_{lang}_{latency_ms(ctx)}ms"
                with (out_dir / "predictions" / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
                    for r, h in zip(rows, hyps):
                        fh.write(json.dumps({**r, "pred_text": h}, ensure_ascii=False) + "\n")
                row = summarize(
                    {"mode": mode, "target_lang": lang, "att_context": ctx}, refs, hyps, equivalents, audio_seconds, wall
                )
                results.append(row)
                print(f"{name}: WER {_pct(row.get('wer'))}  tags {_pct(row['tag_rate'])}  RTFx {row['rtfx']:.0f}",
                      file=sys.stderr)

    (out_dir / "summary.json").write_text(
        json.dumps({"meta": meta, "results": results}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "summary.md").write_text(render_markdown(results, meta), encoding="utf-8")
    if args.mlflow_experiment:
        log_to_mlflow(args.mlflow_experiment, args.mlflow_run_name, meta, results, out_dir)
    return results


if __name__ == "__main__":
    main()
