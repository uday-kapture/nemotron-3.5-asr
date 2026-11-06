"""Planned bolAIndia files -> NeMo tarred training shards + held-out manifests.

    HF_XET_HIGH_PERFORMANCE=1 uv run python -m hinglish_asr.convert \\
        --plan /data/calls/plan.json --out /data/calls --workers 8

    # first look: three files, then read report.json
    uv run python -m hinglish_asr.convert --plan /data/calls/plan.json --out /data/calls --limit-files 3

    # short on disk: push training tars to a private HF dataset repo and stream them in training
    HF_XET_HIGH_PERFORMANCE=1 uv run python -m hinglish_asr.convert \\
        --plan /data/calls/plan.json --out /data/calls --upload-to kapturecx/calls-train-shards

Each planned file k is downloaded whole (range reads of these files are slow:
~185 tiny row groups each), filtered, split by customer, and deleted:

    train/audio_<k>.tar               training rows of file k, FLAC bytes unchanged, shuffled
    train/sharded_manifests/manifest_<k>.json
    {val,test}/audio/<id>.flac        held-out rows, as plain files for evaluation
    client_test/audio/<id>.flac       rows of the config's `test_clients`, if any
    parts/<k>.jsonl                   held-out manifest rows of file k
    done/<k>.json                     written last; a rerun skips file k

then merges them into val_manifest.jsonl, test_manifest.jsonl,
val_small_manifest.jsonl (customer-disjoint subset of val used during
training), client_test_manifest.jsonl and report.json.

Only files of the config's `heldout_subsets` (the calls) give val and test
rows; the public subsets mixed in for diversity go to training whole, so
evaluation stays on calls.

val and test hold out customers, but not the agents who talk to them: an
agent works many calls and is in train too. `test_clients` holds out whole
clients (businesses), and with them their agents and their vocabulary; pick
them from report.json's `top_clients` after a trial run. A done file records
the config's hash, and a rerun with a changed config refuses to mix the two.

Training shard k comes from planned file k, so the shards are numbered
0..N-1 as `input_cfg` requires: pass `train/` as its PATH.

Manifests keep `group` (hashed customer phone), `client_id`, `channel`,
`language` and `confidence` for breakdowns.

With --upload-to, each finished training shard (tar + manifest) goes to that
private HF dataset repo in batched commits and the local tar is deleted, so
local disk holds only a batch of tars, the manifests (~0.5 MB per shard) and
val/test. uploaded/<k>.json marks each shard; a rerun uploads what is missing,
including tars left by an earlier local-only run. train/remote.json records
the repo and commit, and `input_cfg` turns it into streamed tar paths. The
repo then holds call audio and transcripts: it must stay private (checked),
and it is never the source dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import multiprocessing as mp
import os
import random
import sys
import tarfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import CommitOperationAdd

from .bolai import COLUMNS, TARGET_LANG, RowFilter, customer_key, subset_of
from .hub import ensure_private_repo
from .scoring import script_of
from .split import assign_split
from .textnorm import normalize_target

SAMPLE_RATE = 16000


@dataclass(frozen=True)
class _Job:
    index: int
    path: str
    out: str
    cfg: dict
    cache: str
    local_root: str | None = None
    config_sha: str | None = None


def _fetch(job: _Job) -> tuple[Path, bool]:
    """Local path of the file and whether it was downloaded (and so is ours to delete)."""
    if job.local_root:
        return Path(job.local_root) / job.path, False
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(
        job.cfg["dataset"], job.path, repo_type="dataset", revision=job.cfg["revision"], local_dir=job.cache
    )
    return Path(local), True


def convert_file(job: _Job) -> dict:
    cfg, k, out = job.cfg, job.index, Path(job.out)
    flt = RowFilter.from_config(cfg)
    t0 = time.perf_counter()
    src, downloaded = _fetch(job)
    size = src.stat().st_size
    test_clients = {str(c) for c in cfg.get("test_clients") or ()}
    dropped, kept, seconds = Counter(), Counter(), Counter()
    train, held, ids = [], [], set()
    rows = text_changed = 0
    # Only these subsets (the calls) give val/test rows; the rest is training data only.
    heldout_subsets = cfg.get("heldout_subsets")
    holds_out = heldout_subsets is None or subset_of(job.path) in heldout_subsets
    try:
        pf = pq.ParquetFile(src)
        names = [c for c in COLUMNS if c in pf.schema_arrow.names]
        for batch in pf.iter_batches(batch_size=2048, columns=names):
            rows += batch.num_rows
            # Skip other vendors' rows before building Python objects.
            mine = batch.filter(pc.is_in(batch.column("provider"), pa.array(flt.providers))) if flt.providers else batch
            if flt.providers:
                dropped["other_vendor"] += batch.num_rows - mine.num_rows
            for row in mine.to_pylist():
                reason = flt.reason(row)
                if reason:
                    dropped[reason] += 1
                    continue
                text = normalize_target(row["text"])
                if not text:
                    dropped["empty_text"] += 1
                    continue
                audio = (row.get("audio") or {}).get("bytes")
                try:
                    info = sf.info(io.BytesIO(audio))
                except (sf.LibsndfileError, RuntimeError, TypeError):
                    dropped["bad_audio"] += 1
                    continue
                if info.samplerate != SAMPLE_RATE or info.channels != 1:
                    dropped["bad_audio"] += 1
                    continue
                # duration_s is the dataset's metadata; the manifest records the audio's own length.
                if info.duration < flt.min_duration:
                    dropped["too_short"] += 1
                    continue
                if info.duration > flt.max_duration:
                    dropped["too_long"] += 1
                    continue
                # The id names the tar member and the held-out file: NeMo pairs tar
                # members with manifest rows by name, so a repeat would mix up audio and text.
                # Public subsets' ids are paths (shrutilipi/<recording>_chunk_9): flatten them.
                rid = str(row.get("chunk_id") or "").replace("/", "__")
                if not rid or rid.startswith("."):
                    dropped["bad_id"] += 1
                    continue
                if rid in ids:
                    dropped["duplicate_id"] += 1
                    continue
                ids.add(rid)
                group = customer_key(row)
                if not holds_out:
                    split = "train"
                elif str(row.get("client_id")) in test_clients:
                    split = "client_test"
                else:
                    split = assign_split(group, cfg["val_frac"], cfg["test_frac"], cfg["split_seed"])
                lang = TARGET_LANG.get(row["language_code"], row["language_code"])
                conf = row.get("confidence")
                entry = {
                    "duration": round(info.duration, 3),
                    "text": text,
                    "target_lang": lang,
                    "lang": lang,
                    "id": rid,
                    "group": group,
                    "language": row["language_code"],
                    "client_id": row.get("client_id"),
                    "channel": row.get("channel"),
                    "confidence": None if conf is None else round(conf, 4),
                }
                text_changed += text != row["text"]
                kept[split] += 1
                seconds[split] += info.duration
                if split == "train":
                    train.append((entry, audio))
                else:
                    held.append((split, entry, audio))
    finally:
        if downloaded:
            src.unlink(missing_ok=True)

    # Rows arrive grouped by call; shuffle so a shard is not one call after another.
    random.Random(f"{cfg['split_seed']}:{k}").shuffle(train)
    tar_dir, man_dir = out / "train", out / "train" / "sharded_manifests"
    tar_tmp, man_tmp = tar_dir / f"audio_{k}.tar.tmp", man_dir / f"manifest_{k}.json.tmp"
    with tarfile.open(tar_tmp, "w") as tar, man_tmp.open("w", encoding="utf-8") as man:
        for entry, audio in train:
            name = f"{entry['id']}.flac"
            member = tarfile.TarInfo(name)
            member.size = len(audio)
            tar.addfile(member, io.BytesIO(audio))
            man.write(json.dumps({"audio_filepath": name, **entry, "shard_id": k}, ensure_ascii=False) + "\n")

    part_tmp = out / "parts" / f"{k}.jsonl.tmp"
    with part_tmp.open("w", encoding="utf-8") as part:
        for split, entry, audio in held:
            dst = out / split / "audio" / f"{entry['id']}.flac"
            dst.write_bytes(audio)
            part.write(json.dumps({"audio_filepath": str(dst.resolve()), **entry, "split": split}, ensure_ascii=False) + "\n")

    os.replace(tar_tmp, tar_dir / f"audio_{k}.tar")
    os.replace(man_tmp, man_dir / f"manifest_{k}.json")
    os.replace(part_tmp, out / "parts" / f"{k}.jsonl")
    done = {
        "index": k,
        "path": job.path,
        "config_sha": job.config_sha,
        "rows": rows,
        "kept": dict(kept),
        "hours": {s: v / 3600 for s, v in seconds.items()},
        "dropped": dict(dropped),
        "text_changed_by_normalization": text_changed,
        "bytes": size,
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    (out / "done" / f"{k}.json").write_text(json.dumps(done) + "\n")
    return done


def _run(job: _Job) -> dict:
    try:
        return convert_file(job)
    except Exception as exc:  # one bad file must not stop the other workers
        return {"index": job.index, "path": job.path, "error": f"{type(exc).__name__}: {exc}"}


class ShardUploader:
    """Pushes finished training shards to a private HF dataset repo, then deletes the local tars.

    Shards go up in batches, one commit each (the Hub rate-limits commits).
    """

    def __init__(self, repo: str, out: str | Path, *, source_repo: str, batch: int = 25, api=None,
                 retries: int = 5, sleep=None):
        if api is None:
            from huggingface_hub import HfApi

            api = HfApi()
        ensure_private_repo(api, repo, "dataset", forbidden=(source_repo,))
        self.api, self.repo, self.out, self.batch = api, repo, Path(out), batch
        self.retries, self.sleep = retries, sleep or time.sleep
        (self.out / "uploaded").mkdir(parents=True, exist_ok=True)
        self.pending: list[int] = []

    def is_uploaded(self, k: int) -> bool:
        return (self.out / "uploaded" / f"{k}.json").exists()

    def add(self, k: int) -> None:
        if not self.is_uploaded(k) and k not in self.pending:
            self.pending.append(k)
        if len(self.pending) >= self.batch:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        shards = sorted(self.pending)
        ops = []
        for k in shards:
            ops.append(CommitOperationAdd(f"train/audio_{k}.tar", str(self.out / "train" / f"audio_{k}.tar")))
            ops.append(CommitOperationAdd(
                f"train/sharded_manifests/manifest_{k}.json",
                str(self.out / "train" / "sharded_manifests" / f"manifest_{k}.json"),
            ))
        for attempt in range(self.retries + 1):
            try:
                info = self.api.create_commit(
                    self.repo, operations=ops, repo_type="dataset", commit_message=f"training shards {shards}"
                )
                break
            except Exception as exc:  # a Hub hiccup must not stop a day-long conversion
                if attempt == self.retries:
                    raise
                wait = 30 * 2**attempt
                print(f"upload of shards {shards} failed ({type(exc).__name__}: {exc}); retrying in {wait} s",
                      file=sys.stderr)
                self.sleep(wait)
        for k in shards:
            (self.out / "uploaded" / f"{k}.json").write_text(json.dumps({"commit": info.oid}) + "\n")
            (self.out / "train" / f"audio_{k}.tar").unlink()
        print(f"uploaded {len(shards)} shards to {self.repo}", file=sys.stderr)
        self.pending = []

    def remote_info(self, n_shards: int) -> dict:
        missing = [k for k in range(n_shards) if not self.is_uploaded(k)]
        return {
            "repo": self.repo,
            "revision": self.api.repo_info(self.repo, repo_type="dataset").sha,
            "prefix": "train",
            "shards": n_shards,
            "missing_count": len(missing),
            "missing": missing[:50],
        }


def pick_val_small(rows: list[dict], hours: float, seed: str) -> list[dict]:
    """Whole customers from `rows`, in a seeded order, until `hours` of audio."""
    by_group: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_group[r["group"]].append(r)
    order = sorted(by_group, key=lambda g: hashlib.sha256(f"{seed}\0small\0{g}".encode()).digest())
    picked, total = [], 0.0
    for g in order:
        if total >= hours * 3600:
            break
        picked.extend(by_group[g])
        total += sum(r["duration"] for r in by_group[g])
    return sorted(picked, key=lambda r: r["id"])


class _Tally:
    def __init__(self):
        self.n = 0
        self.seconds = 0.0
        self.groups: set[str] = set()
        self.clients: Counter = Counter()
        self.words = self.latin = self.mixed = 0

    def add(self, row: dict) -> None:
        self.n += 1
        self.seconds += row["duration"]
        self.groups.add(row["group"])
        self.clients[row.get("client_id")] += row["duration"]
        scripts = [script_of(w) for w in row["text"].split()]
        self.words += len(scripts)
        self.latin += scripts.count("latin")
        self.mixed += "latin" in scripts and "devanagari" in scripts

    def summary(self) -> dict:
        top10 = sum(s for _, s in self.clients.most_common(10))
        return {
            "utterances": self.n,
            "hours": round(self.seconds / 3600, 3),
            "customers": len(self.groups),
            "clients": len(self.clients),
            "top10_client_hour_share": top10 / self.seconds if self.seconds else None,
            "top_clients": [[c, round(s / 3600, 3)] for c, s in self.clients.most_common(20)],
            "latin_word_share": self.latin / self.words if self.words else None,
            "mixed_script_share": self.mixed / self.n if self.n else None,
        }


def _jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _quantiles(values: list[float], qs=(0.01, 0.05, 0.1, 0.25, 0.5)) -> dict:
    values = sorted(values)
    if not values:
        return {}
    return {f"p{round(q * 100)}": values[int(q * (len(values) - 1))] for q in qs}


def finalize(out: str | Path, plan: dict, files: list[dict]) -> dict:
    out, cfg = Path(out), plan["config"]
    done = {}
    for f in files:
        marker = out / "done" / f"{f['index']}.json"
        if marker.exists():
            done[f["index"]] = json.loads(marker.read_text())
    missing = [f["index"] for f in files if f["index"] not in done]

    held: dict[str, list[dict]] = {"val": [], "test": [], "client_test": []}
    for k in sorted(done):
        for row in _jsonl(out / "parts" / f"{k}.jsonl"):
            held[row.pop("split")].append(row)
    if not cfg.get("test_clients"):
        del held["client_test"]
    # Held-out audio is <split>/audio/<id>.flac: an id in two files is one file
    # holding the later row's audio, so neither row's text can be trusted.
    seen = Counter(r["id"] for rows in held.values() for r in rows)
    clashes = sum(n for n in seen.values() if n > 1)
    held = {split: [r for r in rows if seen[r["id"]] == 1] for split, rows in held.items()}
    for split, rows in held.items():
        rows.sort(key=lambda r: r["id"])
        _write_jsonl(out / f"{split}_manifest.jsonl", rows)
    small = pick_val_small(held["val"], cfg["val_small_hours"], cfg["split_seed"])
    _write_jsonl(out / "val_small_manifest.jsonl", small)

    tallies = {s: _Tally() for s in ("train", *held)}
    by_language, by_channel, confidence = Counter(), Counter(), []
    for k in sorted(done):
        for row in _jsonl(out / "train" / "sharded_manifests" / f"manifest_{k}.json"):
            tallies["train"].add(row)
            if row.get("confidence") is not None:
                confidence.append(row["confidence"])
    for split, rows in held.items():
        for row in rows:
            tallies[split].add(row)
    for split in held:
        for row in held[split]:
            by_language[row["language"]] += row["duration"]
            by_channel[row["channel"]] += row["duration"]

    dropped = Counter()
    for d in done.values():
        dropped.update(d["dropped"])
    report = {
        "config_sha": plan.get("config_sha"),
        "files": {"planned": len(files), "done": len(done), "missing": len(missing), "missing_indices": missing[:50]},
        "rows_read": sum(d["rows"] for d in done.values()),
        "splits": {s: t.summary() for s, t in tallies.items()},
        "val_small": {"utterances": len(small), "hours": round(sum(r["duration"] for r in small) / 3600, 3)},
        "dropped": dict(dropped.most_common()),
        "heldout_id_clashes_dropped": clashes,
        "train_confidence_quantiles": _quantiles(confidence),
        "heldout_hours_by_language": {k: round(v / 3600, 3) for k, v in by_language.items()},
        "heldout_hours_by_channel": {str(k): round(v / 3600, 3) for k, v in by_channel.items()},
        "text_changed_by_normalization": sum(d["text_changed_by_normalization"] for d in done.values()),
        "downloaded_gb": round(sum(d["bytes"] for d in done.values()) / 1e9, 2),
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def convert(
    plan: dict,
    out: str | Path,
    *,
    workers: int = 8,
    limit_files: int | None = None,
    local_root: str | None = None,
    cache: str | Path | None = None,
    upload_to: str | None = None,
    upload_batch: int = 25,
    api=None,
) -> dict:
    out = Path(out)
    for d in ("train/sharded_manifests", "val/audio", "test/audio", "client_test/audio", "parts", "done"):
        (out / d).mkdir(parents=True, exist_ok=True)
    files = plan["files"][:limit_files] if limit_files else plan["files"]
    cache = str(cache or out / "cache")
    sha = plan.get("config_sha")
    stale = [
        f["index"] for f in files
        if (marker := out / "done" / f"{f['index']}.json").exists()
        and ((d := json.loads(marker.read_text())).get("config_sha", sha) != sha or d.get("path", f["path"]) != f["path"])
    ]
    if stale:
        raise ValueError(
            f"files {stale[:10]} in {out} were converted from a different file or config than this plan "
            f"(config_sha {sha}); use a new --out, or delete their done/<k>.json to convert them again"
        )
    jobs = [
        _Job(f["index"], f["path"], str(out), plan["config"], cache, local_root, sha)
        for f in files
        if not (out / "done" / f"{f['index']}.json").exists()
    ]
    print(f"{len(files) - len(jobs)} of {len(files)} files already done; converting {len(jobs)}", file=sys.stderr)
    uploader = None
    if upload_to:
        uploader = ShardUploader(upload_to, out, source_repo=plan["config"]["dataset"], batch=upload_batch, api=api)
        for f in files:  # done earlier but not uploaded (an interrupted or local-only run)
            if (out / "done" / f"{f['index']}.json").exists():
                uploader.add(f["index"])
    errors = []
    if workers > 1 and len(jobs) > 1:
        pool = mp.get_context("spawn").Pool(workers)
        results = pool.imap_unordered(_run, jobs)
    else:
        pool, results = None, map(_run, jobs)
    try:
        for i, res in enumerate(results, 1):
            if "error" in res:
                errors.append(res)
                print(f"[{i}/{len(jobs)}] file {res['index']} FAILED: {res['error']}", file=sys.stderr)
            else:
                print(
                    f"[{i}/{len(jobs)}] file {res['index']}: kept {sum(res['kept'].values())} of {res['rows']} rows "
                    f"({sum(res['hours'].values()):.2f} h) in {res['elapsed_s']} s",
                    file=sys.stderr,
                )
                if uploader:
                    uploader.add(res["index"])
    except BaseException:
        if pool is not None:
            # close() would let every queued file convert first, writing its tar locally with nothing uploading it.
            pool.terminate()
        raise
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    if uploader:
        uploader.flush()
    report = finalize(out, plan, files)
    if uploader:
        report["shard_repo"] = uploader.remote_info(len(files))
        (out / "train" / "remote.json").write_text(json.dumps(report["shard_repo"], indent=1) + "\n")
        (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["errors"] = errors
    if errors:
        (out / "errors.json").write_text(json.dumps(errors, indent=1) + "\n")
    return report


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=8, help="files downloaded and converted at once")
    p.add_argument("--limit-files", type=int, help="only the first N planned files (a trial run)")
    p.add_argument("--local-root", help="read planned files from this local copy of the repo; never delete them")
    p.add_argument("--cache", help="where downloads land while being converted (default OUT/cache)")
    p.add_argument("--upload-to", help="private HF dataset repo for the training tars (deleted locally once uploaded)")
    p.add_argument("--upload-batch", type=int, default=25, help="shards per commit")
    args = p.parse_args(argv)

    plan = json.loads(Path(args.plan).read_text())
    report = convert(
        plan,
        args.out,
        workers=args.workers,
        limit_files=args.limit_files,
        local_root=args.local_root,
        cache=args.cache,
        upload_to=args.upload_to,
        upload_batch=args.upload_batch,
    )
    summary = {k: report[k] for k in ("files", "splits", "dropped")}
    print(json.dumps(summary, ensure_ascii=False, indent=2), file=sys.stderr)
    if report["errors"]:
        sys.exit(f"{len(report['errors'])} files failed; see {Path(args.out) / 'errors.json'}. Rerun to retry them.")
    return report


if __name__ == "__main__":
    main()
