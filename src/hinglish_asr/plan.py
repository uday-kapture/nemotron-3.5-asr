"""Choose which bolAIndia files to convert, from the dataset's stats alone.

    uv run python -m hinglish_asr.plan --config configs/data/calls_mix.yaml \\
        --out /data/calls/plan.json

Fetches stats/*.json at the pinned revision (a few MB) and estimates each
file's usable hours: seconds in the config's `languages`, times the share of
the file in non-test splits (spring-inx files hold dev/eval rows; mucs-hinglish
is all test). `subsets` maps each subset to its share of the planned hours.
Each subset's files are in a seeded shuffle (calls are written in time order,
a few clients at a time, so any prefix is a spread-out sample), and the plan
takes the next file from whichever subset would be furthest below its share
with that file in.
`max_hours` cuts that sequence; raising it only appends files, so `convert`
keeps the ones it already made. Changing the shares reorders the plan: use a
new --out. `convert` then drops more (a quarter of call rows are empty,
unintelligible chunks). plan.json embeds the config, so `convert` needs nothing
else.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import yaml

from .bolai import TEST_LIKE_COLLECTIONS, fetch_stats, load_stats, subset_of


def usable_hours(stat: dict, languages: list[str]) -> float:
    """Hours of `languages` in a file, scaled by its share of non-test splits."""
    seconds = sum(stat.get("languages", {}).get(lang, {}).get("seconds", 0.0) for lang in languages)
    splits = stat.get("splits") or {}
    total = sum(s["seconds"] for s in splits.values())
    if total:
        seconds *= sum(s["seconds"] for name, s in splits.items() if name.lower() not in TEST_LIKE_COLLECTIONS) / total
    return seconds / 3600


def plan_files(
    stats: dict[str, dict],
    *,
    subsets: dict[str, float],
    languages: list[str],
    seed: str,
    max_hours: float | None = None,
    sizes: dict[str, int] | None = None,
) -> list[dict]:
    queues: dict[str, list[dict]] = {s: [] for s in sorted(subsets)}
    for path in sorted(stats):
        subset = subset_of(path)
        hours = usable_hours(stats[path], languages)
        if subset in queues and hours > 0:
            queues[subset].append({"path": path, "subset": subset, "hours": hours, "file_hours": stats[path]["seconds"] / 3600})
    for subset, queue in queues.items():
        random.Random(f"{seed}:{subset}").shuffle(queue)
        queue.reverse()  # pop() takes from the front of the shuffle
    taken = dict.fromkeys(queues, 0.0)
    files, hours = [], 0.0
    while max_hours is None or hours < max_hours:
        open_ = [s for s in queues if queues[s]]
        if not open_:
            break
        # Furthest below its share once its next file is in: a small share
        # does not take a whole file before the plan is big enough for it.
        subset = min(open_, key=lambda s: (taken[s] + queues[s][-1]["hours"]) / subsets[s])
        f = queues[subset].pop()
        if sizes is not None:
            if f["path"] not in sizes:  # listed in stats but absent at this revision
                continue
            f["bytes"] = sizes[f["path"]]
        f["index"] = len(files)
        files.append(f)
        taken[subset] += f["hours"]
        hours += f["hours"]
    return files


def repo_sizes(repo: str, revision: str, subsets: list[str]) -> dict[str, int]:
    from huggingface_hub import HfApi

    api = HfApi()
    sizes = {}
    for subset in subsets:
        for entry in api.list_repo_tree(repo, path_in_repo=f"data/{subset}", repo_type="dataset", revision=revision):
            if entry.path.endswith(".parquet"):
                sizes[entry.path] = entry.size
    return sizes


def config_sha(cfg: dict) -> str:
    """Hash of what decides how a file is converted. `max_hours` only
    decides how many files: a bigger plan starts with the same files, so
    `convert` keeps the ones it already made."""
    kept = {k: v for k, v in cfg.items() if k != "max_hours"}
    return hashlib.sha256(json.dumps(kept, sort_keys=True).encode()).hexdigest()[:12]


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--stats-dir", help="use these stats/*.json instead of fetching them")
    p.add_argument("--no-sizes", action="store_true", help="skip listing the repo (file sizes, existence check)")
    args = p.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    if not cfg["subsets"] or any(not share or share <= 0 for share in cfg["subsets"].values()):
        sys.exit("subsets: map each subset to a positive share of the planned hours")
    out = Path(args.out)
    stats_dir = args.stats_dir or fetch_stats(cfg["dataset"], cfg["revision"], out.parent / "bolai-meta")
    sizes = None if args.no_sizes else repo_sizes(cfg["dataset"], cfg["revision"], list(cfg["subsets"]))
    files = plan_files(
        load_stats(stats_dir),
        subsets=cfg["subsets"],
        languages=cfg["languages"],
        seed=cfg["plan_seed"],
        max_hours=cfg.get("max_hours"),
        sizes=sizes,
    )
    by_subset: dict[str, float] = {}
    for f in files:
        by_subset[f["subset"]] = by_subset.get(f["subset"], 0.0) + f["hours"]
    plan = {
        "config": cfg,
        "config_sha": config_sha(cfg),
        "files": files,
        "totals": {
            "files": len(files),
            "hours": round(sum(f["hours"] for f in files), 1),
            "hours_by_subset": {s: round(h, 1) for s, h in sorted(by_subset.items())},
            "file_hours": round(sum(f["file_hours"] for f in files), 1),
            "download_gb": round(sum(f.get("bytes", 0) for f in files) / 1e9, 1) if sizes is not None else None,
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=1) + "\n")
    print(json.dumps(plan["totals"]), file=sys.stderr)
    return plan


if __name__ == "__main__":
    main()
