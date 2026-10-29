# nemotron-3.5-fine-tune

Data prep and baseline evaluation for fine-tuning
[`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
on Hinglish (Hindi in Devanagari, English in Latin, mixed in one utterance).
Background and sources: [docs/research/nemotron-3.5-asr-hinglish-finetuning.md](docs/research/nemotron-3.5-asr-hinglish-finetuning.md).

```
uv sync                                 # data prep + scoring (no torch)
uv sync --extra nemo --extra mlflow     # + NeMo Speech v3.0.0 for the baseline and training
uv run pytest
```

Python 3.12 is required: `audioop`, which production uses to upsample 8 kHz
calls, is gone in 3.13.

## 1. Prepare data → 16 kHz WAVs + NeMo manifests

`python -m hinglish_asr.prepare` reads utterances from one of three sources,
normalizes the transcripts, converts the audio, assigns splits and writes
`<out>/{train,val,test}_manifest.jsonl`, `<out>/audio/*.wav` and
`<out>/report.json` (hours per split and per source sample rate, drops by
reason, script mix).

| `--source` | input |
|---|---|
| `jsonl` | `--input calls.jsonl`, one row per utterance: `audio_filepath`, `text`, optional `offset`, `duration`, `group`, `id` |
| `hf` | `--dataset org/name --revision <sha>` (or `--parquet file...`): parquet with `audio{bytes,path}` + `text` columns |
| `hush-eval` | `--input ../Hush-Eval/data/test.json`: a Hush-Eval frozen list; its audited `reference`s become the text |

```bash
# Your telephony WAVs (segments of long calls via offset/duration are cut out)
uv run python -m hinglish_asr.prepare --source jsonl --input calls.jsonl \
    --out data/prepared/calls --val-frac 0.01 --test-frac 0.01 --workers 32

# Hindi-Transcription-DS for training, minus Hush-Eval's held-out rows, split by Vaani speaker
uv run python -m hinglish_asr.prepare --source hf \
    --dataset kapturecx/Hindi-Transcription-DS --revision 3557b0ccb3309dd9672dd44f433cb437ee38fb38 \
    --exclude-hush-eval ../Hush-Eval/data/test.json \
    --group-pattern '^IISc_VaaniProject_[A-Z]_[^_]+_[^_]+_([^_]+)_' \
    --out data/prepared/hindi_ds --val-frac 0.01 --workers 32
```

What it does to each utterance:

- **Text** (`textnorm.py`): NFC (precomposed nukta letters decompose, as the
  tokenizer expects), Devanagari digits → ASCII (the tokenizer has no pieces
  for ०-९), zero-width characters removed, ॥ → ।, and Vaani's
  `रेस्टोरेंट {restaurant}` glosses → `restaurant`. Case and punctuation are
  kept: the base model emits them.
- **Audio** (`audio.py`): mono PCM_16 at 16 kHz. 8 kHz is upsampled with
  `audioop.ratecv`, byte-identical to production's LITranscriber, so the model
  trains on what it will hear. 16 kHz is copied. Stereo or other rates are
  dropped as `bad_audio` and listed in the report.
- **Filters**: empty text, < 0.1 s, > 20 s (the base model's pre-training
  range). Long calls must be segmented upstream.
- **Splits** (`split.py`): hashed by `group` (speaker or call), so a speaker
  never straddles train/val/test. Without `group`/`--group-pattern`, the group
  is the audio file (JSONL) or the row (parquet).
- **Manifest rows** carry `target_lang` (read by the prompt model's training
  dataloader) and `lang` (read by NeMo's transcribe scripts), both `hi-IN` by
  default.

Rerunning reuses WAVs already written, so an interrupted run resumes.
Parquet sources only download the row groups that hold wanted rows.

## 2. Baseline the unmodified model

```bash
# 1,000 audited Hinglish test utterances (~36 parquet row groups, ~3 GB fetched)
uv run python -m hinglish_asr.prepare --source hush-eval --input ../Hush-Eval/data/test.json \
    --limit 1000 --out data/prepared/hindi_test --single-split test

uv run --extra nemo python -m hinglish_asr.baseline \
    --manifest data/prepared/hindi_test/test_manifest.jsonl \
    --equivalents ../Hush-Eval/data/equivalents.json \
    --out reports/baseline/hindi_test
```

Runs every combination of `--modes` (offline, streaming), `--target-langs`
(default `hi-IN auto`) and `--att-context-sizes` (default 80/320/1120 ms) and
writes `summary.md`, `summary.json` and raw predictions per combination
(`text` = reference, `pred_text` = model output with language tags kept).

The summary answers the open questions from the research doc: which prompt
suits Hinglish, whether the model writes English words in Latin or Devanagari
(Latin share vs the reference's), whether it emits `<xx-XX>` tags, and how
WER moves with latency. WER/CER use Hush-Eval's normalization
(`scoring.py`; checked identical on all 25,405 hindi_test references) with
tags stripped, plus WER split by the script of each word.

Add `--modes offline streaming --limit 200` once to confirm that offline
decoding at a lookahead matches true cache-aware streaming on your audio.
Add `--mlflow-experiment NAME` to log the run; the MLflow client reads
`MLFLOW_TRACKING_URI` and either `MLFLOW_TRACKING_USERNAME`/`PASSWORD` (Basic)
or `MLFLOW_TRACKING_TOKEN` (Bearer). If both are set, only Basic is sent.

## 3. Shard and blend for training

```bash
scripts/make_tarred.sh data/prepared/calls/train_manifest.jsonl data/tarred/calls 256

uv run python -m hinglish_asr.input_cfg \
    --entry data/tarred/calls:0.7:auto \
    --entry data/tarred/hindi_ds:0.3:unified \
    --out configs/input_cfg.yaml
```

`make_tarred.sh` runs NeMo's `convert_to_tarred_audio_dataset.py` (fetched
once at v3.0.0) with FLAC shards. Use a shard count divisible by the GPU
count. `input_cfg.py` writes the Lhotse blend for `model.train_ds.input_cfg`:
one entry per source with its weight and `prompt_mode` tag.

## Things to know

- **`Hindi-Transcription-DS` is not telephony.** It is 517,035 rows of 16 kHz
  audio, mostly IISc Vaani (crowd-sourced image descriptions from many
  districts), with Sarvam clips at the end. Baselines on Hush-Eval's
  `hindi_test` measure Hinglish handling on wideband speech, not on 8 kHz calls.
  Build a telephony test set from your call data with `--source jsonl`.
- **Hush-Eval's `hindi_test` is a row split, not a speaker split.** Training on
  the rest of the dataset (with `--exclude-hush-eval`) still shares Vaani
  speakers with it, so post-fine-tune WER there will look better than it is
  for new speakers. The `val`/`test` splits written here are speaker-disjoint
  when `--group-pattern` is set.
- **Gloss resolution changes 1.6% of `hindi_test` references** (`X {y}` → `y`),
  so baseline WER here can differ slightly from Hush-Eval's on the same rows.
