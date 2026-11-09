# nemotron-3.5-fine-tune

Data prep, baseline, fine-tuning and benchmarking for
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

## Calls pipeline: bolAIndia → fine-tune → benchmark

Everything below runs on the GPU box. Training text is Kapture call chunks
from the gated `kapturecx/bolAIndia`, from every transcription vendor, mixed
with human-labelled public hi/en speech from the same dataset, so that no
single vendor's mistakes dominate the labels. Rows labelled `hi` and `en` both
train, each with its own language prompt (`hi-IN`, `en-US`). The model is
trained and judged at the deployment lookahead: 80 ms
(`att_context_size [56,0]`).

```bash
uv sync --extra nemo --extra mlflow --extra semantic
cp .env.example .env && chmod 600 .env       # fill in MLflow (required), HF token, Chat webhook; git-ignored
source .env                                  # in every new shell / tmux window
uv run hf auth login                         # unless HF_TOKEN is in .env: read kapturecx/bolAIndia, write the checkpoint repo
scripts/fetch_inputs.sh                      # checks dataset read, checkpoint repo write, MLflow; fetches models/base/*.nemo

# 1. plan: which files, mixed by subset share, from the dataset's stats only
uv run python -m hinglish_asr.plan --config configs/data/calls_mix.yaml --out /data-g/uday/calls/plan.json

# 2. convert: download each file whole, keep hi/en rows, write tar shards + held-out manifests
uv run python -m hinglish_asr.convert \
    --plan /data-g/uday/calls/plan.json --out /data-g/uday/calls --limit-files 3      # look at report.json first
uv run python -m hinglish_asr.convert --plan /data-g/uday/calls/plan.json --out /data-g/uday/calls   # HF_XET_HIGH_PERFORMANCE from .env
#   short on disk: add --upload-to kapturecx/<private-shard-repo> to stream the training tars instead

# 3. blend
uv run python -m hinglish_asr.input_cfg --entry /data-g/uday/calls/train:1.0:langID --out configs/input_cfg.yaml

# 4. pilot: batch size, speed, max_steps   5. LR sweep   6. full run
uv run python -m hinglish_asr.train pilot --arm cosine --batch-durations 300 400 500 600 --passes 3
uv run python -m hinglish_asr.train sweep --arms noam cosine --set model.train_ds.batch_duration=500
uv run python -m hinglish_asr.train run --arm cosine --name calls-v1 --retries 2 \
    --set model.train_ds.batch_duration=500 --set trainer.max_steps=<from runs/pilot.json>

# 7. benchmark the checkpoints: choose on val, report on test (and held-out clients);
#    --push-winner puts winner.nemo and the report in runs/calls-v1/ of the checkpoint repo
uv run python -m hinglish_asr.benchmark --run runs/calls-v1 --push-winner \
    --val /data-g/uday/calls/val_manifest.jsonl --test /data-g/uday/calls/test_manifest.jsonl \
    --client-test /data-g/uday/calls/client_test_manifest.jsonl     # if test_clients is set
```

**Data** (`bolai.py`, `plan.py`, `convert.py`, `configs/data/calls_mix.yaml`).
`subsets` maps each subset to its share of the planned hours. The calls
(`combined`, `inhouse-asr-1`, `inhouse-asr-2`) take 69%: at the pinned
revision they hold ~20,200 h of hi+en, transcribed by AssemblyAI (85%),
Deepgram (15%) and Gnani (1%). Human-labelled public sets take the rest:
indicvoices, spring-inx, vaani, shrutilipi, kathbath (hi) and nptel
(Indian English). `plan` estimates each file's usable hours from the stats
(hours in `languages`, scaled by the share outside test splits), shuffles
each subset's files with a seed, and keeps taking the next file from whichever
subset would be furthest below its share with that file in. `max_hours: 1000`
plans 208 files, ~21 GB to download; about a quarter of call rows are empty
"unintelligible" chunks and are dropped. `report.json` has the hours
actually kept.
Label styles differ by source: Deepgram writes English words in Latin
(`mobile`), AssemblyAI and Gnani in Devanagari (`मोबाइल`); numbers stay as
each source wrote them (`itn.py`, number words → digits, is not used for now).
Only `heldout_subsets` (the calls) give val and test rows; the public subsets
are training data only.
The audio is already 16 kHz mono FLAC chunks, so the converter
copies the bytes into `train/audio_<k>.tar` without
resampling. Call rows are split by customer: the phone hash of the called party
on outbound calls, the caller on inbound ones. 1% of customers each go to
val and test, and `val_small_manifest.jsonl` (3 h) is the validation set used
during training. That keeps customers apart, not agents: an agent works many
calls, so the agent side of val and test is speech the model trained on.
`test_clients` holds whole clients out of every other split
(`client_test_manifest.jsonl`), and with them their agents; pick them from
`report.json`'s `splits.train.top_clients` after the trial run.

**Spoken-style variant** (`configs/data/calls_spoken.yaml`): the same files,
prepared the way `kapturecx/hinden-ASR-1_7B-vaani` prepared its data:
- `min_confidence` per provider: AssemblyAI 0.65 and Deepgram 0.60. Below
  about 0.6, AssemblyAI often writes another language on Hindi audio. Rows
  without a confidence (human labels, Gnani) are not gated.
- `target_style: spoken`: lowercase, no punctuation (apostrophes inside words
  stay), and numbers as words in the row's language (`verbalize.py`):
  `₹2,000` → `दो हज़ार रुपये`, `4:30` → `साढ़े चार`, `9876543210` digit by digit.
  `report.json`'s `digits_left` lists rows that still hold a digit.
- `empty_targets`: clips the vendor left empty are set aside as candidates
  (`empty_candidates.jsonl`, `empty/audio/`). VAD can't vouch for them, since
  every chunk is VAD-cut. `verify_empty` decodes them with the base model and
  keeps those it also leaves empty.

```bash
uv run python -m hinglish_asr.plan --config configs/data/calls_spoken.yaml --out /data-g/uday/calls-spoken/plan.json
uv run python -m hinglish_asr.convert --plan /data-g/uday/calls-spoken/plan.json --out /data-g/uday/calls-spoken
uv run --extra nemo python -m hinglish_asr.verify_empty --data /data-g/uday/calls-spoken
uv run python -m hinglish_asr.input_cfg --out configs/input_cfg.yaml \
    --entry /data-g/uday/calls-spoken/train:0.97:langID \
    --entry /data-g/uday/calls-spoken/train_empty_manifest.jsonl:0.03:langID
# benchmark: --val val_manifest.jsonl val_empty_manifest.jsonl (same for --test);
# it reports the hallucination rate on the empty clips, which WER skips
```

To grow the data, raise `max_hours` (or set it to null), rerun `plan`,
and rerun `convert` on the same `--out`: a bigger plan starts with the same
files, and the cap is left out of the config hash, so only the new files are
converted. Changing the shares reorders the plan: use a new `--out`. Then rerun `input_cfg`. val and test grow too, so compare an older
model with a newer one by benchmarking both on the new test split.

Reruns skip finished files (`done/<k>.json`). A done file records the
config's hash and the file's path, so after changing the config (and rerunning
`plan`) convert refuses an `--out` that holds files made with the old one; use a new `--out`
or delete those `done/<k>.json`. Rows whose `chunk_id` repeats within a file,
or is unusable as a file name, are dropped (`duplicate_id`, `bad_id`): NeMo
pairs tar members with manifest rows by name. A held-out id found in two files
is dropped from both (`heldout_id_clashes_dropped`), as they share one audio
file. Duration limits apply to the audio's own length, not `duration_s`. `report.json` gives hours per
split, drops by reason, the Latin-word share, hours by client, and
confidence quantiles (vendor rows only; human rows have none).

**Streaming the training shards** (`convert --upload-to REPO`).
- **During conversion:** each finished tar and its manifest go to a private
  HF dataset repo in batched commits (25 shards each), and the local tar is
  deleted. A failed commit is retried 5 times (30 s to 8 min apart); if it
  still fails, conversion stops at once, rather than converting the remaining
  files into local tars. Rerun to continue.
- **Peak local disk:** one batch of tars, plus the parquet files in flight
  (about 2 GB at `--workers 8`), plus manifests and val/test (about 3 GB).
- **During training:** `input_cfg` sees `train/remote.json` and writes the tar
  paths as `pipe:curl … https://huggingface.co/datasets/REPO/resolve/<commit>/…`,
  pinned to the commit `convert` recorded. NeMo reads each shard as a stream.
  Training needs about 13 MB/s at 1,000× realtime, or about 47 GB per pass over 1,000 h.
- **Auth:** curl reads the token from a header file (mode 600, written from
  your HF login), so the token is never in the YAML or the process list. Use a
  token with write access to the shard repo for convert, and read access for
  training.
- **Requirements:** curl ≥ 7.71, for `--retry-all-errors`. A failed read makes
  NeMo skip the whole shard with only a warning.
- **Safeguards:** the converter refuses a public repo and the source dataset.
  The repo holds call audio and transcripts.

**Training** (`train.py`, `configs/train.yaml`) follows NeMo's
`speech_to_text_finetune.py`. It adds:
- validation at 80 ms with each row's language prompt; `val_wer` scores the
  benchmark's normalization (tags, case, punctuation), so the top-k it keeps
  are the ones the benchmark would rank best;
- lookahead sampling weighted to 80 ms (`att_context_probs`; NeMo's default
  is uniform over the four trained lookaheads);
- validation loss, plus the top-5 and last checkpoints by `val_wer` (one
  .nemo at the end, not one per checkpoint);
- MLflow logging and a throughput log.

The loss is the .nemo's: RNNT (`warprnnt_numba`) with FastEmit λ=5e-3, no
auxiliary CTC. The targets carry no `<xx-XX>` tags, so the fine-tuned model
stops emitting them; scoring strips them either way. `pilot` runs ~300 steps
per `batch_duration` in separate processes. It picks the fastest one whose
peak memory stays under 85% of the GPU, and turns the measured audio seconds
per step into `max_steps` for N passes. `sweep` runs the `noam` (peak 3e-4)
and `cosine` (1e-4) arms for equal steps and compares their best `val_wer`.
Each arm's LR schedule spans the full run's `max_steps` (from
`runs/pilot.json`, or `--schedule-steps`), so a 4,000-step arm follows the
first 4,000 steps of the schedule it would have in the full run, instead of
annealing to `min_lr` within the sweep. Child runs log to
`runs/logs/<name>.log`. Rerunning the same `--name` resumes from its last
checkpoint (saved every 1,000 steps). After running out of GPU memory, or
being killed by the host's OOM killer, each sweep arm (and `run --retries N`)
does that by itself, up to 2 times by default. It stops when an attempt dies
at the same checkpoint as the one before. Failed attempts' logs are kept as
`runs/logs/<name>.oom<n>.log`.

**Checkpoints on the Hub** (`checkpoint_hub_repo` in `configs/train.yaml`,
set to `kapturecx/nemotron-3.5-fine-tune`; `null` keeps everything local).
With a private HF model repo there, the top-k checkpoints are kept on local
disk only until they are uploaded. Each goes to `runs/<name>/` in the repo as weights
only, about 2.4 GB instead of 8 GB with optimizer state, and the same commit
deletes whatever falls out of the top 5. Only the `-last` checkpoint (about
8 GB, with optimizer state) stays on disk, for resuming. NeMo's end-of-run
`.nemo` is switched off, because it reloads the best checkpoint from disk;
the benchmark writes `winner.nemo`. The benchmark reads `checkpoint_hub.json`
in the run directory. It downloads each candidate once, loads it, deletes it,
and builds the top-k average while they load. An upload can happen at each
validation (every 1,000 steps). Uploads run on a background thread, one at a
time and in order, while training goes on: at 3 MB/s one takes about 14
minutes, longer than an epoch, so a few copies of 2.4 GB can queue in
`<run>/upload-tmp/`. A queued checkpoint that has fallen out of the top 5 by
its turn is skipped. The run waits for the queue before it finishes, and a
failed upload fails the run. A process killed while uploads are queued loses
those checkpoints. The pilot uploads nothing; each sweep arm uploads its own
`runs/<sweep>-<arm>/`. `benchmark --push-winner` adds `runs/<name>/winner.nemo`
(unless the base model won) and `runs/<name>/benchmark/benchmark.{md,json}`;
those names carry no `val_wer`, so the top-k trimming never touches them.

**Benchmark** (`benchmark.py`, `semantic.py`).
- **Candidates:** the base model, the top-k checkpoints, the last checkpoint,
  and the average of the top-k weights. Each is decoded on the full val split
  at 80 ms, each row with its own prompt (`hi-IN` or `en-US`).
- **Choosing:** lowest val WER wins. Test (and `--client-test`) is decoded only
  for the winner and the base model.
- **Metrics:**
  - WER/CER and WER by script;
  - **lenient WER**: cross-script spellings (`order`/`ऑर्डर`), Devanagari
    variants (`है`/`हैं`), number words vs digits and split words count as
    matches. A number that is wrong or missing costs every word it was
    written with, as in plain WER. Written forms match spoken ones (`0987` /
    `शून्य नौ आठ सात`, `1,50,000` / `डेढ़ लाख`), and lenient WER is never above
    plain WER;
  - **semdist** (LaBSE);
  - breakdowns by language, channel, duration and client;
  - the change against the base model, with a 95% interval from resampling
    customers.
- **Checks:** the gain's interval on test (and on held-out clients) is below
  zero; val is not used for this, because picking the best of several
  candidates there inflates its gain. Also: true streaming matches offline
  decoding on 500 val utterances, and lenient WER picks the same winner.
- **Outputs:** `runs/<name>/benchmark/benchmark.md` and `winner.nemo`.
  Predictions and `cross_script_pairs.json` (the cross-script matches to
  audit) hold transcripts and stay on the box.

All eval splits are scored against the vendors' transcripts, so every call WER
here measures agreement with them, not accuracy. Vendors write English words in
different scripts, so plain WER counts `mobile`/`मोबाइल` as an error; lenient
WER does not.

**MLflow** (`tracking.py`). Every stage refuses to run without
`MLFLOW_TRACKING_URI`, because the fallback would be a local store nobody
sees. Training logs through Lightning's MLFlowLogger, as NeMo's exp_manager
hookup breaks on `run_name`. The benchmark adds one nested run per
candidate under the training run, metrics only. Weights are uploaded only
with `benchmark --log-model`. Basic auth wins if both
`MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_TOKEN` are set; set one.

**Google Chat updates** (`notify.py`). In a Chat space, add an incoming
webhook (Apps & integrations → Webhooks) and export its URL as
`GCHAT_WEBHOOK_URL` on the training box. The URL lets anyone post into the
space, so keep it out of configs and git. Without it nothing is sent.
- **Per run**, in its own thread: start (or resume step), progress every
  `notify.every_n_steps` steps (loss, lr, s/step, × realtime, ETA), each
  epoch's `val_wer` and `val_loss` against the best so far, a non-finite loss,
  any error with the end of its traceback, and the finish with the best
  `val_wer`.
- **Drivers:** `sweep` posts each arm's best `val_wer` and the winner;
  `pilot` posts its table and chosen batch size. Pilot runs themselves stay
  quiet, since an out-of-memory run is how the pilot finds the limit.
- **What it can't report:** a run killed by a signal under `pilot`/`sweep`
  is reported by the parent, but a `train run` killed outright (`kill -9`,
  the host's OOM killer, a lost machine) sends nothing. Progress messages
  stopping is the sign.
- Messages carry metrics and step counts, never transcripts. A failed post
  is printed to stderr and training continues.

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
