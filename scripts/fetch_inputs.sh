#!/usr/bin/env bash
# Inputs every later stage needs, checked and fetched on the training box:
#   - an HF token that can read the gated kapturecx/bolAIndia
#   - the base model .nemo, at the commit the research doc pins, in models/base/
#
#   scripts/fetch_inputs.sh
#
# Log in once with `uv run hf auth login`, or export HF_TOKEN (a fine-grained,
# read-only token scoped to the dataset). Never commit a token.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"

uv run python - <<'EOF'
import sys
from pathlib import Path

import yaml
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

MODEL = "nvidia/nemotron-3.5-asr-streaming-0.6b"
MODEL_REVISION = "ea30d66debe3740a08b573244286791d423d6b3e"
NEMO = "nemotron-3.5-asr-streaming-0.6b.nemo"

cfg = yaml.safe_load(Path("configs/data/calls_deepgram.yaml").read_text())
api = HfApi()
try:
    user = api.whoami()["name"]
except Exception:
    sys.exit("no HF token: run `uv run hf auth login` or export HF_TOKEN")
try:
    api.auth_check(cfg["dataset"], repo_type="dataset")
except (GatedRepoError, RepositoryNotFoundError) as exc:
    sys.exit(f"{user} cannot read {cfg['dataset']}: request access on its HF page ({type(exc).__name__})")
print(f"HF: {user} can read {cfg['dataset']}", file=sys.stderr)

path = hf_hub_download(MODEL, NEMO, revision=MODEL_REVISION, local_dir="models/base")
print(f"base model: {path}", file=sys.stderr)
EOF
