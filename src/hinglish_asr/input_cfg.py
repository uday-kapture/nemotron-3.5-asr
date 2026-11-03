"""Build the Lhotse `input_cfg` that blends training sources for the prompt model.

    uv run python -m hinglish_asr.input_cfg \\
        --entry data/tarred/hinglish:0.8:auto \\
        --entry data/tarred/hindi_replay:0.2:langID \\
        --out configs/input_cfg.yaml

Each entry is PATH:WEIGHT[:PROMPT_MODE]. PATH is a directory written by NeMo's
convert_to_tarred_audio_dataset.py (audio_<i>.tar + sharded_manifests/) or a
plain JSONL manifest. PROMPT_MODE becomes the `prompt_mode` tag the prompt
model's dataloader reads per cut (`prompt_mode_field: prompt_mode`):
"auto" (NVIDIA's config labels it "use for code-switching"), "langID" or
"unified" (the config default). Use the YAML as `model.train_ds.input_cfg`.

A directory whose tars were pushed to the Hub by `convert --upload-to` has a
remote.json and local manifests only. Its tar paths become streamed reads,
`pipe:curl ... https://huggingface.co/datasets/<repo>/resolve/<commit>/...`,
pinned to the commit convert recorded. NeMo opens each shard as a stream, so
nothing lands on disk. The repo is private: curl reads the token from a header
file (--auth-header, written from your HF login with mode 600 if missing), so
the token never appears in the YAML or the process list.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

PROMPT_MODES = ("auto", "langID", "unified")
DEFAULT_AUTH_HEADER = Path.home() / ".cache" / "hinglish-asr" / "hf-auth-header"
# NeMo turns these into brace-expansion braces anywhere in a shard path.
_NEMO_BRACKETS = set("()[]<>")


class InputCfgError(ValueError):
    pass


def _check_manifests(p: Path, n: int) -> None:
    for i in range(n):
        need = p / "sharded_manifests" / f"manifest_{i}.json"
        if not need.exists():
            raise InputCfgError(f"{p}: shards must be numbered 0..{n - 1}; missing {need.name}")


def streamed_tars(remote: dict, auth_header: str | Path | None) -> str:
    n = remote["shards"]
    url = (
        f"https://huggingface.co/datasets/{remote['repo']}/resolve/{remote['revision']}"
        f"/{remote['prefix']}/audio__OP_0..{n - 1}_CL_.tar"
    )
    # Retries: a failed read makes NeMo skip the whole shard with only a warning.
    cmd = "pipe:curl -sfL --retry 10 --retry-delay 5 --retry-all-errors"
    if auth_header is not None:
        if _NEMO_BRACKETS & set(str(auth_header)):
            raise InputCfgError(f"auth header path {auth_header} contains one of {''.join(sorted(_NEMO_BRACKETS))}")
        cmd += f" -H @{auth_header}"
    return f"{cmd} {url}"


def write_auth_header(path: str | Path) -> Path:
    from huggingface_hub import get_token

    token = get_token()
    if not token:
        raise InputCfgError("no HF token: run `uv run hf auth login` or export HF_TOKEN")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(f"Authorization: Bearer {token}\n")
    return path


def build_entry(
    path: str, weight: float, prompt_mode: str | None = None, auth_header: str | Path | None = None
) -> dict:
    if prompt_mode is not None and prompt_mode not in PROMPT_MODES:
        raise InputCfgError(f"prompt_mode {prompt_mode!r} not in {PROMPT_MODES}")
    p = Path(path)
    if p.is_dir() and (p / "remote.json").exists():
        remote = json.loads((p / "remote.json").read_text())
        if remote["missing_count"]:
            raise InputCfgError(
                f"{p}: {remote['missing_count']} shards not uploaded to {remote['repo']} yet "
                f"(e.g. {remote['missing'][:5]}); rerun convert --upload-to"
            )
        _check_manifests(p, remote["shards"])
        entry = {
            "type": "nemo_tarred",
            "manifest_filepath": f"{p}/sharded_manifests/manifest__OP_0..{remote['shards'] - 1}_CL_.json",
            "tarred_audio_filepath": streamed_tars(remote, auth_header),
            "weight": weight,
        }
    elif p.is_dir():
        n = len(list(p.glob("audio_*.tar")))
        if n == 0:
            raise InputCfgError(f"{p}: no audio_*.tar shards")
        for i in range(n):
            for need in (p / f"audio_{i}.tar", p / "sharded_manifests" / f"manifest_{i}.json"):
                if not need.exists():
                    raise InputCfgError(f"{p}: shards must be numbered 0..{n - 1}; missing {need.name}")
        entry = {
            "type": "nemo_tarred",
            "manifest_filepath": f"{p}/sharded_manifests/manifest__OP_0..{n - 1}_CL_.json",
            "tarred_audio_filepath": f"{p}/audio__OP_0..{n - 1}_CL_.tar",
            "weight": weight,
        }
    elif p.is_file():
        entry = {"type": "nemo", "manifest_filepath": str(p), "weight": weight}
    else:
        raise InputCfgError(f"{p}: not a tarred dataset directory or a manifest file")
    if prompt_mode is not None:
        entry["tags"] = {"prompt_mode": prompt_mode}
    return entry


def _parse(spec: str) -> tuple[str, float, str | None]:
    parts = spec.rsplit(":", 2)
    if len(parts) == 3 and parts[2] in PROMPT_MODES:
        return parts[0], float(parts[1]), parts[2]
    path, weight = spec.rsplit(":", 1)
    return path, float(weight), None


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--entry", action="append", required=True, help="PATH:WEIGHT[:PROMPT_MODE]")
    p.add_argument("--out", required=True)
    p.add_argument("--auth-header", default=str(DEFAULT_AUTH_HEADER),
                   help="curl header file for streamed private shards; created from your HF login if missing")
    args = p.parse_args(argv)
    specs = [_parse(spec) for spec in args.entry]
    header = None
    if any((Path(path) / "remote.json").exists() for path, _, _ in specs):
        header = Path(args.auth_header)
        if not header.exists():
            write_auth_header(header)
    cfg = [build_entry(path, weight, mode, header) for path, weight, mode in specs]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")


if __name__ == "__main__":
    main()
