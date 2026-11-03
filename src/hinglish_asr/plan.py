"""Choose which bolAIndia files to convert, from the dataset's stats alone.

    uv run python -m hinglish_asr.plan --config configs/data/calls_deepgram.yaml \\
        --out /data/calls/plan.json

Fetches stats/*.json at the pinned revision (a few MB), keeps the files of
`subsets` that hold rows from `vendor`, and orders them by a seeded shuffle:
calls are written in time order, a few clients at a time, so any prefix of
the plan is a spread-out sample. `max_vendor_hours` cuts the list at that many
vendor hours as the stats count them; `convert` then drops more (a quarter of
Deepgram rows are empty, unintelligible chunks). plan.json embeds the config,
so `convert` needs nothing else.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import yaml

from .bolai import fetch_stats, load_stats, subset_of


def plan_files(
    stats: dict[str, dict],
    *,
    subsets: list[str],
    vendor: str,
    seed: str,
    max_vendor_hours: float | None = None,
    sizes: dict[str, int] | None = None,
) -> list[dict]:
    chosen = []
    for path in sorted(stats):
        v = stats[path].get("vendors", {}).get(vendor)
        if subset_of(path) in subsets and v and v["seconds"] > 0:
            chosen.append({"path": path, "vendor_hours": v["seconds"] / 3600, "file_hours": stats[path]["seconds"] / 3600})
    random.Random(seed).shuffle(chosen)
    files, hours = [], 0.0
    for f in chosen:
        if max_vendor_hours is not None and hours >= max_vendor_hours:
            break
        if sizes is not None:
            if f["path"] not in sizes:  # listed in stats but absent at this revision
                continue
            f["bytes"] = sizes[f["path"]]
        f["index"] = len(files)
        files.append(f)
        hours += f["vendor_hours"]
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
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--stats-dir", help="use these stats/*.json instead of fetching them")
    p.add_argument("--no-sizes", action="store_true", help="skip listing the repo (file sizes, existence check)")
    args = p.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.out)
    stats_dir = args.stats_dir or fetch_stats(cfg["dataset"], cfg["revision"], out.parent / "bolai-meta")
    sizes = None if args.no_sizes else repo_sizes(cfg["dataset"], cfg["revision"], cfg["subsets"])
    files = plan_files(
        load_stats(stats_dir),
        subsets=cfg["subsets"],
        vendor=cfg["vendor"],
        seed=cfg["plan_seed"],
        max_vendor_hours=cfg.get("max_vendor_hours"),
        sizes=sizes,
    )
    plan = {
        "config": cfg,
        "config_sha": config_sha(cfg),
        "files": files,
        "totals": {
            "files": len(files),
            "vendor_hours": round(sum(f["vendor_hours"] for f in files), 1),
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
