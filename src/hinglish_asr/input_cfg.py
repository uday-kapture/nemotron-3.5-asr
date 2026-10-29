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
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

PROMPT_MODES = ("auto", "langID", "unified")


class InputCfgError(ValueError):
    pass


def build_entry(path: str, weight: float, prompt_mode: str | None = None) -> dict:
    if prompt_mode is not None and prompt_mode not in PROMPT_MODES:
        raise InputCfgError(f"prompt_mode {prompt_mode!r} not in {PROMPT_MODES}")
    p = Path(path)
    if p.is_dir():
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
    args = p.parse_args(argv)
    cfg = [build_entry(*_parse(spec)) for spec in args.entry]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")


if __name__ == "__main__":
    main()
