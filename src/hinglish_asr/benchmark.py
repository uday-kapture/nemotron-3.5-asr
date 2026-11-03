"""Score a training run's checkpoints at the deployment setting; pick on val, report on test.

    uv run --extra nemo --extra mlflow --extra semantic python -m hinglish_asr.benchmark \\
        --run runs/calls-v1 \\
        --val /data/calls/val_manifest.jsonl --test /data/calls/test_manifest.jsonl

Candidates: the base model, each top-k checkpoint, the last checkpoint and the
average of the top-k weights. Top-k checkpoints come from <run>/checkpoints/,
or, if the run wrote checkpoint_hub.json, from its private HF model repo: each
is downloaded once, loaded, and deleted (the average is accumulated as they
load). Each is decoded on the full val split at 80 ms
with the `auto` prompt (offline decoding at that lookahead equals cache-aware
streaming; the winner is re-checked with the streaming loop on a sample). The
winner has the lowest val WER. Test is decoded once, for the winner and the
base model only, so it is never used to choose, and the gain over the base
model is judged on test: on val it is inflated by having picked the best of
several candidates there.

val and test hold out customers, not the agents on the other side of the
call, who also appear in train. --client-test (convert's
client_test_manifest.jsonl, whole clients and their agents held out) is
decoded and judged like test.

Per candidate and split: WER/CER and WER by script (`scoring`), lenient WER
and semdist (`semantic`), breakdowns by language, channel, duration and
client, and the change against the base model with a 95% interval from a
customer-level paired bootstrap.

Writes <run>/benchmark/: benchmark.json, benchmark.md, winner.nemo and, kept
on the box because they hold call transcripts, predictions/*.jsonl and
cross_script_pairs.json (the cross-script matches to audit). MLflow gets one
nested run per candidate under the training run, metrics only.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml

from .hub import rank_checkpoints
from .scoring import error_rate, paired_bootstrap, score_corpus, utterance_errors
from .semantic import lenient_scores, semdist_summary

DEPLOY_CONTEXT = [56, 0]
DEPLOY_PROMPT = "auto"


# ---------- candidates ----------


def find_checkpoints(ckpt_dir: str | Path) -> tuple[list[tuple[Path, float]], Path | None]:
    """Top-k checkpoints with their val_wer (best first) and the last checkpoint."""
    paths = sorted(Path(ckpt_dir).glob("*.ckpt"))
    last = next((p for p in paths if p.stem.endswith("-last")), None)
    by_name = {p.name: p for p in paths}
    return [(by_name[n], w) for n, w in rank_checkpoints(list(by_name))], last


class RunningAverage:
    """Mean of state dicts added one at a time: float tensors averaged, others from the first."""

    def __init__(self):
        self.sum: dict | None = None
        self.dtypes: dict = {}
        self.n = 0

    def add(self, sd: dict) -> None:
        if self.sum is None:
            self.dtypes = {k: v.dtype for k, v in sd.items()}
            self.sum = {k: v.detach().float().clone() if v.is_floating_point() else v.clone() for k, v in sd.items()}
        else:
            for k, v in sd.items():
                if v.is_floating_point():
                    self.sum[k] += v.float()
        self.n += 1

    def result(self) -> dict:
        return {k: (v / self.n).to(self.dtypes[k]) if v.is_floating_point() else v for k, v in self.sum.items()}


# ---------- scoring ----------


def duration_bucket(d: float) -> str:
    return "<2s" if d < 2 else "2-5s" if d <= 5 else ">5s"


def breakdown(rows: list[dict], per: list, key, top: int | None = None) -> dict[str, dict]:
    """WER per value of `key(row)`; with `top`, the `top` values with most audio and "other"."""
    hours = defaultdict(float)
    for r in rows:
        hours[str(key(r))] += r["duration"] / 3600
    keep = set(sorted(hours, key=hours.get, reverse=True)[:top]) if top else set(hours)
    groups: dict[str, list] = defaultdict(list)
    for r, p in zip(rows, per):
        groups[str(key(r)) if str(key(r)) in keep else "other"].append((r, p))
    out = {}
    for value, items in sorted(groups.items()):
        out[value] = {
            "utterances": len(items),
            "hours": round(sum(r["duration"] for r, _ in items) / 3600, 3),
            "wer": error_rate(p for _, p in items),
        }
    return out


def evaluate(rows: list[dict], hyps: list[str], equivalents=None, semdist=None) -> tuple[dict, list]:
    refs = [r["text"] for r in rows]
    per = utterance_errors(refs, hyps, equivalents)
    lenient = lenient_scores(refs, hyps, equivalents)
    out = score_corpus(refs, hyps, equivalents)
    out["lenient_wer"] = lenient["lenient_wer"]
    out["lenient_matches"] = lenient["matches_by_rule"]
    if semdist is not None:
        out.update(semdist_summary(semdist.score(refs, hyps, equivalents)))
    out["by"] = {
        "language": breakdown(rows, per, lambda r: r.get("language")),
        "channel": breakdown(rows, per, lambda r: r.get("channel")),
        "duration": breakdown(rows, per, lambda r: duration_bucket(r["duration"])),
        "client": breakdown(rows, per, lambda r: r.get("client_id"), top=10),
    }
    out["_cross_script_pairs"] = lenient["cross_script_pairs"]
    return out, per


def choose(candidates: dict[str, dict], metric: str = "wer") -> str:
    return min(candidates, key=lambda c: candidates[c]["val"][metric])


# ---------- reporting ----------


def _pct(x) -> str:
    return "–" if x is None else f"{100 * x:.2f}%"


def _delta(d: dict | None) -> str:
    if not d or d.get("delta") is None:
        return "–"
    lo, hi = d["ci95"]
    return f"{100 * d['delta']:+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]"


REPORTED = {"test": "Test (reported once)", "client_test": "Held-out clients (reported once)"}


def render_markdown(result: dict) -> str:
    meta, cands = result["meta"], result["candidates"]
    reported = [s for s in REPORTED if meta.get(s)]
    lines = [
        f"# Benchmark: {meta['run']}",
        "",
        f"- Setting: {meta['latency_ms']} ms lookahead {meta['att_context']}, prompt `{meta['prompt']}`, offline decoding",
        "- " + "; ".join(f"{s}: `{meta[s]}` ({meta[f'{s}_utterances']} utterances, {meta[f'{s}_hours']:.2f} h)"
                         for s in ("val", *reported)),
        "- References are Deepgram transcripts: WER here is agreement with Deepgram.",
        "- Δ vs base: WER change in points, with a 95% interval from resampling customers.",
        "- val and test share no customer with training, but do share agents and clients"
        + ("; held-out clients share neither." if "client_test" in reported
           else ": test WER is optimistic for the agent side and for new clients (see --client-test)."),
        "",
        "## Val (chooses the winner)",
        "",
        "| candidate | WER | Δ vs base | CER | WER Devanagari | WER Latin | lenient WER | semdist |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, c in cands.items():
        v = c["val"]
        mark = " **(winner)**" if name == result["winner"] else ""
        sd = "–" if v.get("semdist_mean") is None else f"{v['semdist_mean']:.4f}"
        lines.append(
            f"| {name}{mark} | {_pct(v['wer'])} | {_delta(c.get('val_vs_base'))} | {_pct(v['cer'])} "
            f"| {_pct(v.get('wer_devanagari'))} | {_pct(v.get('wer_latin'))} | {_pct(v['lenient_wer'])} | {sd} |"
        )
    for split in reported:
        lines += ["", f"## {REPORTED[split]}", "",
                  "| model | WER | Δ vs base | CER | WER Devanagari | WER Latin | lenient WER | semdist |",
                  "|---|---|---|---|---|---|---|---|"]
        for name, c in cands.items():
            if split not in c:
                continue
            t = c[split]
            sd = "–" if t.get("semdist_mean") is None else f"{t['semdist_mean']:.4f}"
            lines.append(
                f"| {name} | {_pct(t['wer'])} | {_delta(c.get(f'{split}_vs_base'))} | {_pct(t['cer'])} "
                f"| {_pct(t.get('wer_devanagari'))} | {_pct(t.get('wer_latin'))} | {_pct(t['lenient_wer'])} | {sd} |"
            )
    lines += ["", "## Checks", ""]
    for name, check in result["checks"].items():
        lines.append(f"- **{name}**: {'pass' if check['pass'] else 'FAIL'} — {check['detail']}")
    winner_test = cands[result["winner"]].get("test", {})
    for key in ("language", "channel", "duration", "client"):
        rows = winner_test.get("by", {}).get(key)
        if not rows:
            continue
        lines += ["", f"### Winner on test by {key}", "", "| value | utterances | hours | WER |", "|---|---|---|---|"]
        for value, b in rows.items():
            lines.append(f"| {value} | {b['utterances']} | {b['hours']:.2f} | {_pct(b['wer'])} |")
    return "\n".join(lines) + "\n"


# ---------- main ----------


def _read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _strip_private(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="training run directory (config.yaml, checkpoints/)")
    p.add_argument("--val", required=True, help="full val manifest from convert")
    p.add_argument("--test", required=True)
    p.add_argument("--client-test", help="convert's client_test_manifest.jsonl (whole clients held out)")
    p.add_argument("--base", help="base .nemo (default: base_model in the run's config.yaml)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--equivalents", help="Hush-Eval equivalents.json, applied to both sides")
    p.add_argument("--no-semdist", action="store_true", help="skip LaBSE semdist (no `semantic` extra)")
    p.add_argument("--no-average", action="store_true")
    p.add_argument("--streaming-sample", type=int, default=500, help="val utterances re-decoded with true streaming")
    p.add_argument("--streaming-tolerance", type=float, default=0.002, help="max |streaming - offline| WER")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--limit", type=int, help="only the first N rows of each split (smoke test)")
    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument("--log-model", action="store_true", help="upload winner.nemo to the training run in MLflow")
    args = p.parse_args(argv)

    import torch

    from .baseline import latency_ms, load_model, transcribe_offline, transcribe_streaming
    from .scoring import load_equivalents

    run_dir = Path(args.run)
    run_cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    base = args.base or run_cfg["base_model"]
    out_dir = run_dir / "benchmark"
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
    splits = {"val": _read_jsonl(args.val), "test": _read_jsonl(args.test)}
    if args.client_test:
        splits["client_test"] = _read_jsonl(args.client_test)
    reported = [s for s in REPORTED if s in splits]
    if args.limit:
        splits = {k: v[: args.limit] for k, v in splits.items()}
    equivalents = load_equivalents(args.equivalents) if args.equivalents else None
    semdist = None
    if not args.no_semdist:
        from .semantic import SemDist

        semdist = SemDist(device=args.device)

    local_topk, last = find_checkpoints(run_dir / "checkpoints")
    hub = None
    if (run_dir / "checkpoint_hub.json").exists():
        from .hub import HubCheckpoints

        info = json.loads((run_dir / "checkpoint_hub.json").read_text())
        hub = HubCheckpoints(info["repo"], info["run"], check=False)
        topk = [(("hub", name), wer) for name, wer in hub.list()]
        where = f"{info['repo']}/{hub.prefix}"
    else:
        topk = [(("local", path), wer) for path, wer in local_topk]
        where = str(run_dir / "checkpoints")
    if not topk:
        sys.exit(f"no val_wer checkpoints in {where}")
    model = load_model(base, args.device)
    base_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    sources: dict[str, tuple] = {"base": ("state", base_sd)}
    for i, (source, _) in enumerate(topk, 1):
        sources[f"top{i}"] = source
    if last is not None:
        sources["last"] = ("local", last)
    average = RunningAverage() if not args.no_average and len(topk) > 1 else None
    avg_name = f"avg-top{len(topk)}"

    def state_dict_of(source: tuple) -> dict:
        kind, ref = source
        if kind == "state":
            return ref
        if kind == "hub":
            path = hub.download(ref, out_dir / "ckpt-cache")
            try:
                return torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
            finally:
                path.unlink(missing_ok=True)
        return torch.load(ref, map_location="cpu", weights_only=False)["state_dict"]

    def load(source: tuple) -> dict:
        sd = state_dict_of(source)
        model.load_state_dict(sd)
        model.eval()
        return sd

    def decode(name: str, split: str) -> list[str]:
        rows = splits[split]
        t0 = time.perf_counter()
        hyps = transcribe_offline(model, [r["audio_filepath"] for r in rows], DEPLOY_PROMPT, DEPLOY_CONTEXT, args.batch_size)
        wall = time.perf_counter() - t0
        with (out_dir / "predictions" / f"{name}_{split}.jsonl").open("w", encoding="utf-8") as fh:
            for r, h in zip(rows, hyps):
                fh.write(json.dumps({**r, "pred_text": h}, ensure_ascii=False) + "\n")
        print(f"{name}/{split}: decoded {len(rows)} in {wall:.0f} s", file=sys.stderr)
        return hyps

    candidates: dict[str, dict] = {}
    per: dict[tuple[str, str], list] = {}
    pairs: dict[str, list] = {}
    for name in [*sources, *([avg_name] if average else [])]:
        if name == avg_name:
            sources[name] = ("state", average.result())
        sd = load(sources[name])
        if average is not None and name.startswith("top"):
            average.add(sd)
        del sd
        scores, per[(name, "val")] = evaluate(splits["val"], decode(name, "val"), equivalents, semdist)
        pairs[name] = scores.pop("_cross_script_pairs")
        kind, ref = sources[name]
        candidates[name] = {"source": name if kind == "state" else f"{kind}:{ref}", "val": scores}
        print(f"{name}: val WER {_pct(scores['wer'])}  lenient {_pct(scores['lenient_wer'])}", file=sys.stderr)

    val_groups = [r["group"] for r in splits["val"]]
    for name in candidates:
        if name != "base":
            candidates[name]["val_vs_base"] = paired_bootstrap(
                val_groups, per[("base", "val")], per[(name, "val")], args.bootstrap
            )
    winner = choose(candidates)
    lenient_winner = choose(candidates, "lenient_wer")

    # Test (and held-out clients): winner and base only, decoded once.
    for name in dict.fromkeys(["base", winner]):
        load(sources[name])
        for split in reported:
            scores, per[(name, split)] = evaluate(splits[split], decode(name, split), equivalents, semdist)
            scores.pop("_cross_script_pairs")
            candidates[name][split] = scores
    if winner != "base":
        for split in reported:
            candidates[winner][f"{split}_vs_base"] = paired_bootstrap(
                [r["group"] for r in splits[split]], per[("base", split)], per[(winner, split)], args.bootstrap
            )

    # The winner's weights are loaded now (test decoded last): streaming check, then save.
    sample = random.Random(0).sample(splits["val"], min(args.streaming_sample, len(splits["val"])))
    paths, refs = [r["audio_filepath"] for r in sample], [r["text"] for r in sample]
    offline = score_corpus(refs, transcribe_offline(model, paths, DEPLOY_PROMPT, DEPLOY_CONTEXT, args.batch_size), equivalents)["wer"]
    streaming = score_corpus(refs, transcribe_streaming(model, paths, DEPLOY_PROMPT, DEPLOY_CONTEXT, args.batch_size), equivalents)["wer"]
    if winner != "base":
        model.save_to(str(out_dir / "winner.nemo"))

    checks = {}
    for split in reported:
        gain = candidates[winner].get(f"{split}_vs_base")
        checks[f"beats base on {split}"] = {
            "pass": bool(gain and gain["ci95"][1] < 0),
            "detail": "winner is the base model" if not gain else f"Δ {_delta(gain)} points; interval must be below 0",
        }
    checks.update({
        "streaming matches offline": {
            "pass": abs(streaming - offline) <= args.streaming_tolerance,
            "detail": f"{len(sample)} val utterances: offline {_pct(offline)}, streaming {_pct(streaming)}",
        },
        "lenient WER agrees": {
            "pass": lenient_winner == winner,
            "detail": f"lowest WER: {winner}; lowest lenient WER: {lenient_winner}",
        },
    })
    meta = {
        "run": str(run_dir),
        "base": base,
        "att_context": DEPLOY_CONTEXT,
        "latency_ms": latency_ms(DEPLOY_CONTEXT),
        "prompt": DEPLOY_PROMPT,
        "val": args.val,
        "test": args.test,
        "client_test": args.client_test,
        "equivalents": args.equivalents,
        "semdist_model": semdist.model_name if semdist else None,
        **{f"{s}_utterances": len(rows) for s, rows in splits.items()},
        **{f"{s}_hours": sum(r["duration"] for r in rows) / 3600 for s, rows in splits.items()},
    }
    result = {"meta": meta, "winner": winner, "candidates": candidates, "checks": checks}
    (out_dir / "benchmark.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n")
    (out_dir / "benchmark.md").write_text(render_markdown(result), encoding="utf-8")
    (out_dir / "cross_script_pairs.json").write_text(json.dumps(pairs, ensure_ascii=False, indent=1) + "\n")
    print((out_dir / "benchmark.md").read_text(), file=sys.stderr)

    if not args.no_mlflow:
        _log_mlflow(run_dir, run_cfg, result, out_dir, args.log_model and winner != "base")
    return result


def _flat_metrics(scores: dict) -> dict:
    out = {k: v for k, v in scores.items() if k not in ("by", "lenient_matches")}
    for rule, n in scores.get("lenient_matches", {}).items():
        out[f"lenient_matches.{rule}"] = n
    return out


def _log_mlflow(run_dir: Path, run_cfg: dict, result: dict, out_dir: Path, log_model: bool) -> None:
    import mlflow

    from . import tracking

    experiment = run_cfg["mlflow"]["experiment"]
    id_file = run_dir / "mlflow_run_id"
    train_run_id = id_file.read_text().strip() if id_file.exists() else None
    for name, c in result["candidates"].items():
        with tracking.run(experiment, f"{run_dir.name}/{name}", stage="benchmark", parent_run_id=train_run_id,
                          tags={"candidate": name, "winner": str(name == result["winner"])}):
            tracking.log_params({"source": c["source"], **{k: result["meta"][k] for k in ("att_context", "prompt", "val")}})
            for split in ("val", *REPORTED):
                if split in c:
                    tracking.log_metrics(_flat_metrics(c[split]), prefix=f"{split}/")
                vs = c.get(f"{split}_vs_base")
                if vs and vs["delta"] is not None:
                    tracking.log_metrics({"delta": vs["delta"], "ci95_lo": vs["ci95"][0], "ci95_hi": vs["ci95"][1]},
                                         prefix=f"{split}_vs_base/")
    if train_run_id:
        with tracking.run(experiment, run_id=train_run_id, stage="train"):
            winner = result["candidates"][result["winner"]]
            mlflow.set_tag("benchmark_winner", result["winner"])
            for split in ("val", *REPORTED):
                if split in winner:
                    tracking.log_metrics(_flat_metrics(winner[split]), prefix=f"benchmark/{split}/")
            # benchmark.* only: predictions and cross-script pairs hold transcripts.
            mlflow.log_artifact(str(out_dir / "benchmark.md"), "benchmark")
            mlflow.log_artifact(str(out_dir / "benchmark.json"), "benchmark")
            if log_model:
                mlflow.log_artifact(str(out_dir / "winner.nemo"), "model")


if __name__ == "__main__":
    main()
