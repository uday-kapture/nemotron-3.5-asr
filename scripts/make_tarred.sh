#!/usr/bin/env bash
# Shard a prepared training manifest into NeMo tarred audio for Lhotse training.
#
#   scripts/make_tarred.sh data/prepared/calls/train_manifest.jsonl data/tarred/calls 256
#
# NUM_SHARDS: use a multiple of the total GPU count so every rank gets the same
# number of shards (NeMo datasets.rst). Tar only training data: validation and
# test stay as plain manifests.
#
# The converter lives in NeMo's repo scripts/, not in the pip package, so it is
# fetched once at the same tag as the nemo extra in pyproject.toml.
set -euo pipefail

if [ $# -ne 3 ]; then
    echo "usage: $0 MANIFEST OUT_DIR NUM_SHARDS" >&2
    exit 2
fi
manifest=$1 out_dir=$2 num_shards=$3

nemo_ref=v3.0.0
root=$(cd "$(dirname "$0")/.." && pwd)
converter=$root/.cache/nemo-$nemo_ref/convert_to_tarred_audio_dataset.py
if [ ! -f "$converter" ]; then
    mkdir -p "$(dirname "$converter")"
    curl -fsSL -o "$converter.tmp" \
        "https://raw.githubusercontent.com/NVIDIA-NeMo/Speech/$nemo_ref/scripts/speech_recognition/convert_to_tarred_audio_dataset.py"
    mv "$converter.tmp" "$converter"
fi

# Durations match prepare.py's defaults and the base model's pre-training
# range; FLAC halves the tar size with no loss.
cd "$root"
uv run --extra nemo python "$converter" \
    --manifest_path="$manifest" \
    --target_dir="$out_dir" \
    --num_shards="$num_shards" \
    --max_duration=20.0 \
    --min_duration=0.1 \
    --shuffle --shuffle_seed=1 \
    --sort_in_shards \
    --force_codec=flac \
    --workers=-1
