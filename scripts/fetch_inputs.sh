#!/usr/bin/env bash
# Inputs every later stage needs, checked and fetched on the training box:
#   - an HF token that can read the gated kapturecx/bolAIndia
#   - write access to configs/train.yaml's checkpoint_hub_repo (created private
#     here if missing), when it is set
#   - an MLflow server at MLFLOW_TRACKING_URI that answers with these credentials
#   - the base model .nemo, at the commit the research doc pins, in models/base/
#
#   scripts/fetch_inputs.sh
#
# Log in once with `uv run hf auth login`, or export HF_TOKEN. With a
# checkpoint repo the token needs write access to it (a fine-grained token:
# read on the dataset, write on the checkpoint repo). Never commit a token.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"

uv run --extra mlflow python - <<'EOF'
import os
import sys
from pathlib import Path

import yaml
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

MODEL = "nvidia/nemotron-3.5-asr-streaming-0.6b"
MODEL_REVISION = "ea30d66debe3740a08b573244286791d423d6b3e"
NEMO = "nemotron-3.5-asr-streaming-0.6b.nemo"

cfg = yaml.safe_load(Path("configs/data/calls_deepgram.yaml").read_text())
train_cfg = yaml.safe_load(Path("configs/train.yaml").read_text())
api = HfApi()
try:
    me = api.whoami()
except Exception:
    sys.exit("no HF token: run `uv run hf auth login` or export HF_TOKEN")
user = me["name"]
try:
    api.auth_check(cfg["dataset"], repo_type="dataset")
except (GatedRepoError, RepositoryNotFoundError) as exc:
    sys.exit(f"{user} cannot read {cfg['dataset']}: request access on its HF page ({type(exc).__name__})")
print(f"HF: {user} can read {cfg['dataset']}", file=sys.stderr)

repo = train_cfg.get("checkpoint_hub_repo")
if repo:
    from hinglish_asr.hub import ensure_private_repo

    role = ((me.get("auth") or {}).get("accessToken") or {}).get("role")
    if role == "read":
        sys.exit(f"the HF token is read-only; checkpoints and winner.nemo go to {repo}, which needs write access")
    try:
        ensure_private_repo(api, repo, "model")
    except Exception as exc:
        sys.exit(f"checkpoint repo {repo}: {type(exc).__name__}: {exc}"[:400])
    note = "" if role == "write" else " (fine-grained token: make sure it can write to this repo)"
    print(f"HF: checkpoints go to {repo}, private{note}", file=sys.stderr)

from hinglish_asr import tracking

try:
    uri = tracking.tracking_uri()
except tracking.TrackingError as exc:
    sys.exit(str(exc))
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")  # fail in seconds, not minutes
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "15")
import mlflow

try:
    mlflow.MlflowClient(uri).search_experiments(max_results=1)
except Exception as exc:
    sys.exit(f"MLflow at {uri} did not answer: {type(exc).__name__}: {exc}"[:400])
print(f"MLflow: {uri} answers; runs go to experiment {train_cfg['mlflow']['experiment']}", file=sys.stderr)

path = hf_hub_download(MODEL, NEMO, revision=MODEL_REVISION, local_dir="models/base")
print(f"base model: {path}", file=sys.stderr)
EOF
