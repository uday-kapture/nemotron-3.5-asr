"""Blend a fine-tuned model's weights with the base model's (weight interpolation, "WiSE-FT").

    uv run --extra nemo python -m hinglish_asr.blend \\
        --tuned runs/calls-raw-numbers-v1/benchmark/winner.nemo \\
        --out-dir runs/calls-raw-numbers-v1/blends --alphas 0.3 0.5 0.7

Each blend is alpha * fine-tuned + (1 - alpha) * base, weight by weight:
alpha 1 is the fine-tune, 0 the base model. Fine-tuning moves the weights
toward the calls and away from what the base model knew; a blend stops part
of the way, which usually keeps most of the gain and recovers much of what
was forgotten. No training: score each blend on the calls test set (baseline)
and on general speech (the arena) and pick the alpha that trades best.

Writes <out-dir>/blend-<alpha>.nemo (~2.4 GB each), with the fine-tune's config.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def blend(base: dict, tuned: dict, alpha: float) -> dict:
    """alpha * tuned + (1 - alpha) * base for float tensors; integer buffers (step counters) from tuned."""
    if base.keys() != tuned.keys():
        diff = sorted(base.keys() ^ tuned.keys())
        raise ValueError(f"the models have different layers, e.g. {diff[:3]}: not the same architecture")
    out = {}
    for k, t in tuned.items():
        if base[k].shape != t.shape:
            raise ValueError(f"{k}: shape {tuple(base[k].shape)} in base, {tuple(t.shape)} in the fine-tune")
        out[k] = base[k].lerp(t, alpha) if t.is_floating_point() else t
    return out


def main(argv: list[str] | None = None) -> list[Path]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default="models/base/nemotron-3.5-asr-streaming-0.6b.nemo")
    p.add_argument("--tuned", required=True, help="the fine-tuned .nemo (a run's benchmark/winner.nemo)")
    p.add_argument("--alphas", nargs="+", type=float, default=[0.3, 0.5, 0.7], help="share of the fine-tune, 0..1")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args(argv)
    if any(not 0 <= a <= 1 for a in args.alphas):
        p.error("--alphas: each between 0 (base) and 1 (fine-tune)")

    from nemo.collections.asr.models import ASRModel

    base = ASRModel.restore_from(args.base, map_location="cpu").state_dict()
    model = ASRModel.restore_from(args.tuned, map_location="cpu")  # keeps the fine-tune's config
    tuned = {k: v.detach().clone() for k, v in model.state_dict().items()}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for alpha in args.alphas:
        model.load_state_dict(blend(base, tuned, alpha))
        path = out_dir / f"blend-{alpha:g}.nemo"
        model.save_to(str(path))
        print(path)
        paths.append(path)
    return paths


if __name__ == "__main__":
    main()
