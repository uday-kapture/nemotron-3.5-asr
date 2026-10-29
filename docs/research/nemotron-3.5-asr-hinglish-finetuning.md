# Fine-tuning NVIDIA Nemotron 3.5 ASR (streaming) on Hinglish: research notes

- **Date:** 2026-10-27. **Scope:** research only (no training code or configs).
- **Sources pinned to:** HF model repo commit `ea30d66` (last modified 2026-09-10), NeMo Speech `main` @ `2bf0867` (2026-10-27) plus the `v3.0.0` tag (2026-08-07), Lightning 2.4.0, MLflow v3.17.2.
- **User context:**
  - **Transcripts:** mixed script inside one utterance (English in Latin, Hindi in Devanagari), e.g. "मुझे meeting के लिए late हो गया".
  - **Audio:** thousands of hours of telephony WAV at a mix of 8 kHz and 16 kHz.
  - **Compute:** one NVIDIA H200 (141 GB).
  - **Tracking and publishing:** a company MLflow server that uses password and token auth; best checkpoints go to the Hugging Face Hub.
- **Conventions:**
  - Every factual claim links to its primary source. Line anchors point at the pinned commits.
  - **(analysis)** marks my own inference or arithmetic.
  - **(measured)** marks something I checked locally against the published files, without running any model.
  - **UNVERIFIED** marks anything I could not confirm.

---

## TL;DR

- **Model.** The model is [`nvidia/nemotron-3.5-asr-streaming-0.6b`][card]: a 600M-parameter (637,997,088 FP32) cache-aware FastConformer-RNNT with language-ID prompt conditioning. It covers 40 locales, including `hi-IN`, and is licensed under **OpenMDW-1.1**, which allows commercial use and redistributing derivatives as long as the license text and notices are kept ([card][card-intro], [license][openmdw]).
- **Official recipe.** NVIDIA's [blog][blog] and its companion [Riva notebook][riva-nb] fine-tune the model with `examples/asr/speech_to_text_finetune.py` in two parts:
  - **Config and init:** the config is `fastconformer_transducer_bpe_streaming_prompt.yaml`, with `+init_from_nemo_model=<the HF .nemo>`.
  - **Data and training:** Lhotse/tarred data where every clip carries a `target_lang`, a full fine-tune driven by a step budget, and evaluation in **streaming** mode at the latency you will deploy.
  - **Version floor:** the model needs **NeMo Speech ≥ 3.0.0** (the prompt model landed 2026-05-29 and is not in 2.7.x) ([commit][nemo-pr15666]).
- **Tokenizer (biggest decision): keep the base tokenizer.**
  - **(measured)** Its SentencePiece model already covers Devanagari and Latin. "मुझे meeting के लिए late हो गया।" encodes as 12 pieces with no `<unk>`.
  - **Gaps:** Devanagari digits (०-९) become `<unk>`, so normalize them away.
  - **If you change the tokenizer:** the RNNT decoder and joint are re-initialized. NeMo also has a known trap: a new tokenizer with the *same* vocab size silently reloads the old decoder and joint weights, which leaves WER stuck at 1.0 ([issue #15778][issue15778]).
- **Prompt for code-mixing.** The official config labels prompt mode `auto` "use for code-switching". The default `unified` mode randomly mixes `auto` with the language ID ([yaml L169-181][prompt-yaml-train]). The choice between `hi-IN`, `auto` and `unified` should be settled by experiment.
- **8 kHz telephony.** NVIDIA recommends **upsampling 8 kHz to 16 kHz offline** and fine-tuning the 16 kHz model ([telephony tutorial][tel-nb]). NeMo's Lhotse loader also resamples on the fly as a safeguard, but that is slow ([dataloader L866-869][lhotse-dl-resample]). Mixed narrowband and wideband training is how NVIDIA's earlier English models were built.
- **Hardware.** One H200 is ample.
  - **Memory (analysis):** about 10 GB of weights, gradients and AdamW state at FP32 precision, plus activations.
  - **Settings:** use bf16 with duration-based Lhotse batching (the official config uses `batch_duration: 400`, `quadratic_duration: 15`), and size the batches with OOMptimizer. NVIDIA publishes no training-throughput numbers, so measure in a pilot run.
  - **Laptop:** the 6 GB RTX 4050 is not viable for full fine-tuning.
- **MLflow has two blocking issues in NeMo.**
  1. `exp_manager` calls `MLFlowLogger(run_name=version, **mlflow_kwargs)`, and the kwargs always contain a `run_name` key, so it raises `TypeError` ([exp_manager L1297][expman-logger]). The fix is still an open PR ([#15261][pr15261]).
  2. NeMo passes `tracking_uri=None` explicitly, so Lightning falls back to `file:./mlruns` and the `MLFLOW_TRACKING_URI` env var is ignored ([Lightning L119/L133][pl-mlflow-init]).
  - **Auth:** MLflow sends **Basic auth when both username and password are set; otherwise Bearer `MLFLOW_TRACKING_TOKEN`**. They are never sent together ([rest_utils L259-263][mlf-rest-auth]).
- **Checkpoints.** The official config sets `always_save_nemo: True` without `save_best_model`, so the `.nemo` is overwritten with the **latest** weights rather than the best. Set `save_best_model=true` ([checkpoint callback][ckpt-cb]).
- **Publishing.** Publish with `push_to_hf_hub()`. Anyone can then load the model with `ASRModel.from_pretrained("org/name")`, which looks for `name.nemo` at the repo root ([hf mixin][hf-mixin], [from_pretrained][hf-from-pretrained]).

---

## 1. Model identity

### 1.1 Exact match found

| Field | Value | Source |
|---|---|---|
| HF repo id | `nvidia/nemotron-3.5-asr-streaming-0.6b` | [HF API][hf-api] |
| Model version | `nemotron-3.5-asr-streaming-0.6b-v1` | [card L614-615][card-version] |
| Release date | HF 2026-06-04 (repo created 2026-05-15, last modified 2026-09-10, sha `ea30d66`) | [card L244-246][card-release], [HF API][hf-api] |
| NGC entry | UNVERIFIED: none found. The card points to NIM and build.nvidia.com only. | [card L341-354][card] |
| Predecessor | `nvidia/nemotron-speech-streaming-en-0.6b` ("Nemotron 3 ASR", English only). Not a candidate for Hinglish. | [card L233-234][card-intro], [en model][en-model] |
| License | OpenMDW-1.1. "This model is ready for commercial use." | [card L242, L733-735][card-license], [OpenMDW-1.1][openmdw] |
| NeMo class | `nemo.collections.asr.models.rnnt_bpe_models_prompt.EncDecRNNTBPEModelWithPrompt`, saved with `nemo_version: 2.8.0rc0` | `model_config.yaml` inside the [.nemo][hf-nemo] (end of file) |
| Architecture | Cache-aware FastConformer encoder (24 layers, d_model 1024, 8× subsampling, so 80 ms frames) and an RNNT decoder (2-layer pred net, 640 hidden; joint 640) | [card L283-305][card-arch], `.nemo` config L585-660 |
| Language conditioning | A 128-dim one-hot language vector is concatenated to the encoder output, then projected back to 1024 (`prompt_kernel`: Linear(1152→2048)→Linear(2048→1024)) | [card L293-298][card-arch], [mixins L955-1034][mixins] |
| Parameters | "600M". The safetensors hold 637,997,088 F32 params. The `.nemo` is 2.37 GB. | [card][card-intro], [HF API][hf-api] |
| Input | 16 kHz, **mono required**, 128 mel bins, 25 ms window / 10 ms hop | [card L564-572][card-io], [processor_config][hf-proc] |
| Output | Text with punctuation and capitalization. In `auto` mode a `<xx-XX>` language tag is appended after the terminal punctuation. | [card L274-279][card-langs] |
| Runtime | "NeMo 26.06". Hardware listed: Ampere, Blackwell, **Hopper**, Jetson, Lovelace, Turing, Volta | [card L590-601][card-sw] |
| Transformers port | `Nemotron3_5AsrForRNNT`, transformers ≥ 5.13.0 | [card L433-440][card-tf], [config.json][hf-config] |

### 1.2 Languages

- **Coverage:** 40 locales in three tiers ([card L260-274][card-langs]).
- **Transcription-ready (19):** includes **Hindi (hi-IN)** and **English (en-US, en-GB)**.
- **Adaptation-ready (8):** these need fine-tuning to be usable.
- **Hindi on FLEURS ([card L684][card-hi]):**
  - With the LangID prompt: **8.13% WER at 80 ms → 6.81% at 1.12 s**.
  - With auto-detect: 11.47% → 8.23%.
- **No Hinglish/code-mixed numbers are published:** UNVERIFIED.

### 1.3 Tokenizer

| Fact | Value | Source |
|---|---|---|
| Type | SentencePiece **unigram**, `nmt_nfkc` normalizer, `byte_fallback=False`, character_coverage 0.9995, `split_by_unicode_script=True` | **(measured)** from `tokenizer.model` inside the [.nemo][hf-nemo] |
| Name in config | `universal_merged_tokenizer_latest_40`, `type: bpe` (NeMo's "bpe" means SentencePiece) | `.nemo` `model_config.yaml` L560-564 |
| Vocab | 13,087 pieces plus the RNNT blank, so 13,088 outputs (`blank_token_id` 13087) | [config.json][hf-config], `.nemo` config L635/L648 |
| Language tags | 40 `<xx-XX>` pieces (for example `<hi-IN>` = id 3247) are ordinary vocabulary pieces | **(measured)** |
| Devanagari | 196 pieces contain Devanagari (79 word-initial). About 2,429 contain ASCII Latin letters. | **(measured)** |
| HF `tokenizer.json` | A *BPE* conversion that segments differently from NeMo's unigram model. Use the `.nemo` tokenizer for any token statistics. | **(measured)** ([tokenizer.json][hf-tok]) |

### 1.4 Latency settings (`att_context_size`, in 80 ms frames)

- **Card's list ([card L421-431][card-nemo]):**
  - `[56,0]` = 80 ms
  - `[56,1]` = 160 ms
  - `[56,3]` = 320 ms
  - `[56,6]` = 560 ms
  - `[56,13]` = 1.12 s
- **Trained list (from the `.nemo` encoder config):** `[[56,3],[56,0],[56,6],[56,13]]` (L602-610), with `chunked_limited` attention, `causal` convolutions and causal downsampling. The HF `config.json` also lists lookaheads `[3,0,6,13]` ([config.json][hf-config]).
- **Discrepancy:** `[56,1]` (160 ms) is **not** in the trained list. NeMo only logs a warning for untrained lookaheads ([encoder L993-1008][encoder-setctx]). The card still reports 160 ms WER numbers, so whether 160 ms is officially supported is UNVERIFIED.

### 1.5 License (OpenMDW-1.1)

- **Grant:** "permission is hereby granted, free of charge, to deal in the Model Materials without restriction" ([OpenMDW-1.1][openmdw]).
- **Redistribution:** you must keep a copy of the agreement and all copyright and origin notices.
- **No copyleft:** there is no same-license requirement for derivatives.
- **No restrictions on use or outputs.**
- **Patent/copyright-litigation termination clause.**
- **"As is":** no warranty.
- **Due diligence:** you are responsible for third-party rights and consents.
- **Derivatives:** the license defines only "Model Materials", not "derivative". That publishing a fine-tune is allowed is my reading of "deal in … without restriction" **(analysis)**; have it confirmed by legal.

---

## 2. Official fine-tuning approach

### 2.1 What NVIDIA prescribes

- **Model card:** points to the blog "how to fine-tune Nemotron 3.5 ASR" ([card L256][card-intro]).
- **The blog's five moves ([blog][blog], 2026-06-04, NVIDIA authors):**
  1. Use tarred NeMo/Lhotse data.
  2. Fine-tune from the base checkpoint with `init_from_nemo_model`, "using the same Cache-Aware FastConformer-RNNT recipe, conditioned on each clip's language tag".
  3. Evaluate on held-out data at deployment latency (e.g. `[56,0]`).
  4. Add data where the language is weak.
  5. Export and deploy.
- **Blog takeaways:** do a full fine-tune with a fixed step budget, get the language tag right, and "blend in a slice of the model's other languages ('replay')".
- **Worked example:** Greek and Bulgarian, about 290 h, later about 2,300 h. Base → fine-tuned WER at 80 ms: el 35→24, bg 22→15.
- **Inconsistency:** Step 1 of the blog says "~2000-hour mix", while Step 4 says the pool grew "from ~290 hours to ~2,300 hours".
- **Companion notebook ([Riva notebook][riva-nb], commit `83ca000`, 2026-06-02):** runs `examples/asr/speech_to_text_finetune.py` with
  - `--config-path=../asr/conf/fastconformer/cache_aware_streaming`
  - `--config-name=fastconformer_transducer_bpe_streaming_prompt.yaml`
  - `+init_from_nemo_model=<HF .nemo>`
  - It then evaluates with the cache-aware streaming script.
  - Its manifests carry `"lang"` and `"target_lang"` fields.
- **NeMo docs:** the fine-tuning page and the agent skill both route generic fine-tuning to the same script ([fine_tuning.rst][docs-ft], [skill SKILL.md L39-46][skill]).
- **Agent skill:** NeMo ships the `nemo-speech-asr-finetune` agent skill, linked from the docs ([fine_tuning.rst L22-25][docs-ft]).

### 2.2 What the fine-tune script actually takes from the YAML

`speech_to_text_finetune.py` restores the full model from the `.nemo` file, then applies **only** these YAML sections ([script L83-115, L179-224][ft-script]):
- `train_ds`, `validation_ds` and `test_ds`
- `optim`
- `spec_augment`
- an optional tokenizer swap

Consequences **(analysis)**:
- **Architecture settings come from the checkpoint.** The encoder (24 layers, multi-lookahead, causal), decoder, joint, `decoding` and `fuse_loss_wer: true` all come from the `.nemo`. YAML values such as `n_layers: 42`, `att_context_size: [70,6]` and `decoding.strip_lang_tags: true` are **ignored** ([yaml L239-308][prompt-yaml-model]).
- **Streaming settings are preserved automatically.**
- **The prompt dictionary comes from the YAML.** The dataloaders get it from the YAML's `model_defaults.prompt_dictionary`, which matches the model for `hi-IN`→6 and `auto`→101 ([yaml L24-144][prompt-yaml-defaults]). The YAML is missing `nn-NO` (104), which is present in the HF processor config ([processor_config][hf-proc]); this is irrelevant for Hinglish.

### 2.3 Key overrides (from the official notebook and config)

| Key | Notebook value | Config default ([yaml][prompt-yaml]) | Notes |
|---|---|---|---|
| `init_from_nemo_model` | HF `.nemo` path | — | `init_from_pretrained_model=nvidia/...` also works, but the script sleeps 60 s first, even on a single GPU ([script L94-112][ft-script]) |
| `model.train_ds.manifest_filepath` / `tarred_audio_filepaths` / `is_tarred` | manifest | `???` / null / `true` | the config assumes tarred Lhotse data ([yaml L146-160][prompt-yaml-train]) |
| `model.validation_ds.manifest_filepath` | manifest | `???` | non-tarred, static `batch_size: 2` |
| `model.train_ds.batch_duration` | 200 | 400 (`quadratic_duration` 15, `num_buckets` 30) | seconds of audio per batch |
| `model.train_ds.max_duration` | — | 39.99 | the `.nemo` pre-training used **20** (`.nemo` L145). Lhotse silently filters longer cuts. |
| `model.optim.name` / `lr` / `weight_decay` | adamw / 0.1 / 0.001 | adamw / 2.0 / 1e-3 | **Noam** scheduler multiplies `lr` by d_model^-0.5 |
| `model.optim.sched.warmup_steps` / `d_model` | 100 / 1024 | 10000 / 1024 | |
| `trainer.precision` | bf16 | bf16 | |
| `trainer.max_steps` / `limit_train_batches` | `max_epochs=1`, `limit_train_batches=60` | 500000 / 1000 | Lhotse with tarred data is infinite, so use `max_steps` ([dataloaders.rst L300-354][docs-dl]) |
| `trainer.use_distributed_sampler` | — | false | required for Lhotse |
| `exp_manager.checkpoint_callback_params` | — | `monitor: val_wer`, `mode: min`, `save_top_k: 5`, `always_save_nemo: True` | see §8 |

**Effective learning rate (analysis).** Noam gives lr × 1024^-0.5 × min(step^-0.5, step × warmup^-1.5) ([lr_scheduler L532, L568-577][lr-sched]). Peak values:

| Source | Peak LR |
|---|---|
| Notebook | **3.1e-4** at step 100 |
| Official YAML | **6.25e-4** at step 10k |
| Original pre-training (`lr: 0.5`, `.nemo` L13771) | **1.56e-4** |

NeMo's general guidance points the other way:
- **Fine-tuning docs:** typical fine-tuning LRs are 1e-4 to 1e-5. If the pretrained config uses Noam, switch to a constant or cosine schedule "to avoid the warmup phase resetting to a high LR" ([fine_tuning.rst L170-176][docs-ft]).
- **Agent skill:**
  - Use cosine annealing and warmup of 1–2% of `max_steps`.
  - Use `lr=1e-4` for large or mixed runs.
  - Use `3e-5` for small domain adaptation (under 20 h).
  - Use `bf16-true` ([skill training-evaluation L5-34][skill-train]).

These official sources conflict, so treat LR and schedule as a sweep: Noam peak about 3e-4 versus cosine 1e-4.

### 2.4 Required NeMo version and container

| Item | Value | Source |
|---|---|---|
| Prompt model added | 2026-05-29 (#15666). Not in `v2.7.3`; present in `v3.0.0`. | [commit][nemo-pr15666]; checked raw-file presence per tag |
| NeMo Speech 3.0 | Repo renamed to `NVIDIA-NeMo/Speech`; uv-first install; "Unified prompt-model support now covers multilingual ASR and streaming inference" | [release notes][nemo-rel3] |
| Model card install | `pip install git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]`; Python ≥ 3.11 | [card L384-390][card-nemo] |
| Docs | Python ≥ 3.12 (pyproject says ≥ 3.10); PyTorch ≥ 2.7; tested stack Python 3.13 + PyTorch 2.11/CUDA 12.9 or 2.12/CUDA 13.2 | [install.rst L9-25][docs-install], [pyproject L26][pyproject] |
| Container | `nvcr.io/nvidia/nemo-speech:26.07.00` | [install.rst L129-134][docs-install] |
| Container includes the prompt model? | UNVERIFIED. The card says "NeMo 26.06"; the agent skill defaults to the container. | [card L592][card-sw], [skill setup L1-16][skill-setup] |
| Lightning | pinned `>2.2.1,<=2.4.0` | [pyproject L83][pyproject] |

### 2.5 Alternative script

- **When it applies:** you need to replace the tokenizer **and** exclude decoder/joint from initialization.
- **How it works:** `examples/asr/asr_transducer/speech_to_text_rnnt_bpe_prompt.py` builds the model from the YAML, then calls `maybe_init_from_pretrained_checkpoint`. That supports `init_from_nemo_model: {model0: {path:…, include:[encoder], exclude:[decoder, joint]}}` ([script][rnnt-train-script], [modelPT L1263-1300][modelpt-init]).
- **Catch:** you must then copy the architecture into the YAML yourself, because its defaults (42 layers, `[70,6]`) do **not** match the checkpoint (24 layers, four lookaheads).
- **Doc bug:** [fine_tuning.rst L65-105][docs-ft] shows `init_from_nemo_model_include/_exclude` keys. Those keys do not exist in `modelPT`, which uses the dict form above **(analysis)**.

---

## 3. Hinglish: tokenizer and script

### 3.1 Does the base tokenizer cover Devanagari? Yes (measured)

These results come from running the published `tokenizer.model` from the `.nemo` through `sentencepiece` (see the Appendix):

| Input | Pieces | `<unk>` |
|---|---|---|
| `मुझे meeting के लिए late हो गया।` | 12: `▁मुझे ▁mee ti ng ▁के ▁लिए ▁ la te ▁हो ▁गया ।` | 0 |
| `मुझे मीटिंग के लिए लेट हो गया।` (English words in Devanagari) | 15: `मीटिंग` splits into single characters | 0 |
| `Aapka account number confirm kar dijiye.` (romanized) | 12 | 0 |
| `ज़रूर` with precomposed U+095B | `▁ ज ़ र ू र`: `nmt_nfkc` decomposes the nukta letter | 0 |
| `मेरा नंबर ९८७६ है` | `९८७६` → **`<unk>`** | **1** |
| `Thank you. <hi-IN>` | `<hi-IN>` is a single piece | 0 |

Character coverage (measured from the vocab):
- **Covered:** 67 Devanagari code points, including virama ्, nukta ़, anusvara ं, chandrabindu ँ, visarga ः, danda । and ॐ.
- **Missing:**
  - Devanagari digits ०-९
  - double danda ॥
  - avagraha ऽ
  - precomposed nukta letters U+0958–U+095F. These are fine after normalization, because they are Unicode composition exclusions and NFC/NFKC keep them decomposed ([CompositionExclusions][unicode-excl]).
  - ZWJ/ZWNJ

**Conclusion (analysis):** the base tokenizer can represent your mixed-script target text as long as you normalize digits and strip zero-width joiners. Devanagari has only about 196 pieces, against about 6,900 CJK and about 2,600 Latin, so rarer Hindi words fall back to character-level pieces. That lengthens target sequences but is not a blocker for RNNT, which can emit up to 10 symbols per frame (`max_symbols: 10`, `.nemo` decoding config).

### 3.2 Options compared

| Option | What happens to decoder/joint | Pros | Cons / risks | NVIDIA guidance |
|---|---|---|---|---|
| **A. Keep the base tokenizer (recommended start)** | Reused as-is | Keeps the pretrained Hindi and English output layers and the language-tag tokens. Zero-risk start. | Devanagari is token-starved; digits must be normalized. | Riva notebook: "If training dataset is not enough (<50 hrs) we recommend using tokenizers from pretrained model itself" ([notebook][riva-nb]). A NeMo maintainer gave the same advice for a code-switched Arabic case ([#15778][issue15778]). |
| **B. New SentencePiece model on the Hinglish train text, then `change_vocabulary`** | `change_vocabulary` **always deletes and re-creates** joint and decoder ([rnnt_bpe_models L271-386][change-vocab]). The fine-tune script then **reloads the old weights if the vocab size is unchanged**, which is wrong for a different tokenizer ([script L145-176][ft-script], [#15778][issue15778]). | Tokens fit your distribution. Feasible with thousands of hours. | Decoder/joint start from scratch, so this needs lots of data and an early validation check. You lose the 40 language tags unless you add them as `--spe_user_defined_symbols`, and you lose the out-of-the-box multilingual output. Use a **different vocab size** from 13,087, or exclude decoder/joint via the §2.5 route. | Docs: for a new tokenizer, "Exclude the decoder/joint from initialization" ([fine_tuning.rst L88-105][docs-ft]). Skill: "Replace or extend it when … the run is multilingual/code-switching" ([skill arch L43-70][skill-arch]). |
| **C. Aggregate tokenizer (`type: agg`, en and hi sub-tokenizers)** | Decoder/joint re-created | NeMo docs recommend `agg` for multilingual code-switched models ([configs.rst L344-357][docs-configs-cs]). | `AggregateTokenizer.text_to_ids(text, lang_id)` routes the **whole utterance** to one sub-tokenizer ([aggregate_tokenizer L121-127][agg-tok]). The prompt dataset passes one language per cut ([prompt dataset L150][prompt-ds-getitem]), so intra-sentence Latin+Devanagari text would hit the wrong sub-vocab **(analysis)**. NeMo's on-the-fly code-switching splices whole monolingual utterances, which does not match real Hinglish. Whether the prompt model supports `agg` is UNVERIFIED. | Not recommended for intra-sentential mixed-script text **(analysis)**. |
| **D. Keep the tokenizer, change only the language prompt** | Unchanged | No vocab change | — | See §3.3 |

**Tokenizer-training details, if you go with B.** `process_asr_text_tokenizer.py` **lowercases by default**, which uses the `nmt_nfkc_cf` rule. You must pass `--no_lower_case` to keep the cased, punctuated style. Use `--spe_character_coverage=1.0` (the default is 1.0). The optional `--spe_byte_fallback`, `--spe_user_defined_symbols` and `--spe_split_digits` flags exist ([script L112-175][tok-script], [sentencepiece_tokenizer L640-700][spm-train]). Train on training transcripts only ([SKILL L70][skill]).

### 3.3 Which language prompt to use for Hinglish

- **How prompts are chosen:** every clip needs a `target_lang` value that resolves in the prompt dictionary. `hi-IN`/`hi` map to index 6, `en-US` to 0 and `auto` to 101. Unknown values raise an error ([prompt_language_utils L33-80][prompt-utils], [yaml L24-144][prompt-yaml-defaults]).
- **Prompt modes** ([dataset L76-146][prompt-ds-modes], [configs.rst L886-896][docs-configs-prompt]):
  - `langID` always passes the clip's language.
  - `auto` always passes the language-agnostic prompt. The official YAML comments it as "**use for code-switching**".
  - `unified` (the training default) uses `auto` with probability `unified_auto_ratio` (0.5), otherwise the language ID.
  - Modes can be set per data source via Lhotse `input_cfg` `tags: {prompt_mode: …}`.
- **Candidates:**
  - (a) `target_lang=hi-IN` with `unified`, the default.
  - (b) `prompt_mode: auto` for the Hinglish source, following the YAML hint.
  - (c) A dedicated unused prompt index (for example a `hi-EN` key). The dictionary has 128 slots with many unused. This requires editing the model's `prompt_dictionary` and is UNVERIFIED as a supported path.
- **Inference must match:** transcription always uses `langID` with your `target_lang` ([prompt model L273-281][prompt-model-transcribe]).

### 3.4 Language-tag token in the targets (open question)

- **What the model does:** it emits `<xx-XX>` after terminal punctuation in `auto` mode ([card L278-279][card-langs]).
- **What the dataset does:** it tokenizes `text` verbatim and adds no tag ([dataset L148-165][prompt-ds-getitem]).
- **What NVIDIA's notebook does:** its manifests have no tag in `text` ([notebook][riva-nb]).
- **Open question:** whether NVIDIA's own training text ended with `<hi-IN>` is **UNVERIFIED**. If your targets omit the tag, the model will likely stop emitting it **(analysis)**.
- **Next step:** decide whether you need language tagging at all, and run a baseline (`target_lang=auto`, `strip_lang_tags=false`) on your data to see the current behavior.

### 3.5 Text normalization for mixed script

Recommendations, with their basis:
- **Unicode NFC everywhere** (train, val, test, inference post-processing). The tokenizer applies `nmt_nfkc` internally (measured). The nukta letters U+0958–095F always decompose to base + U+093C ([CompositionExclusions][unicode-excl]), so references must use the same form or WER will count false errors.
- **Strip ZWJ/ZWNJ (U+200C/U+200D).** They are not in the vocab (measured).
- **Digits:** Devanagari digits map to `<unk>` (measured). Convert them to ASCII digits or to spoken words. The card says training text was "normalized to have spoken forms in text with punctuation and capitalization" ([card L621][card-data]), which hints at spelled-out numbers. The exact number convention is UNVERIFIED; check baseline outputs.
- **Danda:** keep `।` (in vocab); map `॥` to `।`.
- **Casing and punctuation:** the base model outputs punctuation and capitalization. Pick one style and apply it consistently across all sources. NVIDIA's skill treats mixed style as "an experiment setup failure" ([skill data L17-59][skill-data]). Latin words keep case; Devanagari has none.
- **English words always in Latin; never transliterate.** Your convention already does this. Devanagari spellings of English words fragment into characters (measured) and make WER ambiguous.
- **Policy decisions, not normalization:** nukta-optional spellings (ज/ज़, फ/फ़) and chandrabindu versus anusvara. Do not silently fold them in training text. Consider folding them **only in scoring** (§9) **(analysis)**.

---

## 4. Data format and preparation

### 4.1 Manifest (JSONL)

- **Fields:** `audio_filepath`, `duration` and `text` are standard ([datasets.rst L14-27][docs-datasets]).
- **Language field:** the prompt model also needs **`target_lang`**. The config sets `lang_field: target_lang` for train, val and test ([yaml L167/L197/L212][prompt-yaml-train]).
- **Documented example:** `{"audio_filepath": "/path/to/audio.wav", "text": "transcription text", "duration": 10.5, "target_lang": "en-US"}` ([configs.rst L987-999][docs-configs-prompt]).
- **NVIDIA's notebook** also writes `"lang"` ([notebook][riva-nb]). `transcribe_speech.py` reads `lang` by default (`gt_lang_attr_name: "lang"`) ([transcribe L192-214][transcribe]). Carry both fields.
- **Per-source prompt mode** goes in Lhotse `input_cfg` tags: `tags: {prompt_mode: auto|langID|unified}` ([configs.rst L895-896][docs-configs-prompt], [dataloaders.rst L430-452][docs-dl]).

### 4.2 Audio requirements and 8 kHz telephony

| Topic | Finding | Source |
|---|---|---|
| Expected input | 16 kHz (`sample_rate: 16000`), mono required, 128-mel features | `.nemo` config L1, [card L572][card-io] |
| NVIDIA guidance on 8 kHz | "The first approach, **which is recommended**, is to finetune a pretrained 16 kHz model by upsampling all the data to 16 kHz. … upsampling offline before training is not necessary but recommended as online upsampling during training is very time consuming". The tutorial shows `sox input_8k.wav -r 16000 -o output_16k.wav`. | [telephony tutorial][tel-nb] |
| Mixed narrowband + wideband | NVIDIA's English models were trained "with mixed sample rate data" (Fisher and Switchboard at 8 kHz upsampled to 16 kHz), so "our models can be used to recognize both narrowband (8kHz) and wideband speech (16kHz)" | [telephony tutorial][tel-nb] |
| Does NeMo resample automatically? | **Yes.** The Lhotse dataloader does `cuts.map(partial(resample, sampling_rate=config.sample_rate))`, commented "Resample as a safeguard; it's a no-op when SR is already OK". The legacy `AudioSegment` path resamples with librosa when `target_sr != sample_rate`. `sample_rate` is the model's target rate, distinct from `input_sampling_rate`, which describes the source. | [dataloader L866-869][lhotse-dl-resample], [L186-191][lhotse-dl-cfg], [segment L183-188][segment] |
| Tarring | `convert_to_tarred_audio_dataset.py` has no resample option (only `--force_codec`), so resample **before** tarring | [tarred script][tarred-script] |
| Narrowband augmentation | NeMo 3.0 Lhotse adds "clipping, lowpass, lossy codec augmentation" | [v3.0.0 release notes][nemo-rel3] |

**Recommendation (analysis):**
- **Resample:** upsample 8 kHz audio to 16 kHz offline, once (sox or ffmpeg), and leave native 16 kHz audio alone.
- **Mix the bands:** train on both bands, as NVIDIA's guidance does.
- **Expect empty high bands:** upsampled narrowband audio has no energy above 4 kHz, so the upper mel bins will be empty for those clips. If production is 8 kHz only, weight narrowband higher or apply lowpass/codec augmentation to the 16 kHz data.
- **Evaluate per band:** keep separate 8 kHz-origin and 16 kHz-origin test sets.

### 4.3 Duration, segmentation and filters

- **Length limits:** the pre-training `train_ds` used `max_duration: 20` and `min_duration: 0.1` (`.nemo` L145-146). The fine-tune YAML uses `max_duration: 39.99` ([yaml L156][prompt-yaml-train]). The card says maximum length is "specific to GPU Memory" ([card L572][card-io]).
- **Silent filtering:** Lhotse drops out-of-range cuts without warning ([SKILL L71][skill]).
- **(analysis):** segment the telephony calls into utterances of roughly 20 s or less, matching pre-training, and log how many hours each filter removes.

### 4.4 Text normalization and style

- **Rules:** see §3.5.
- **Audit:** run NVIDIA's transcript-style preflight (casing rate, punctuation rate, number conventions, Unicode and script consistency per source) before training ([skill data L17-59][skill-data]).
- **Raw data:** keep the raw manifests and write derived `_style_normalized` manifests.

### 4.5 Tarred and Lhotse data at thousands of hours

- **Tar the training set** with `convert_to_tarred_audio_dataset.py` (`--num_shards`, `--max_duration`, `--min_duration`, `--shuffle`, `--sort_in_shards`, `--force_codec=flac`, `--workers=-1`) ([tarred script L15-80][tarred-script], [datasets.rst L108-186][docs-datasets]).
- **Shard counts:** make shards divisible by world size, with equal entries per shard ([datasets.rst L180-186][docs-datasets]).
- **Do not tar validation or test sets** ([datasets.rst L175-176][docs-datasets], [skill data L65-67][skill-data]).
- **Use Lhotse** for training (`use_lhotse: true`, dynamic bucketing, `input_cfg` blends with weights and tags). With tarred data the loader is infinite, so you need `trainer.max_steps`, `limit_train_batches` (a pseudo-epoch) and `use_distributed_sampler=false` ([dataloaders.rst L300-354][docs-dl]).
- **Validation:** Lhotse with a static `batch_size` and bucketing disabled ([skill data L141-152][skill-data]).
- **Small sets (≈100 h on one GPU):** NVIDIA suggests no bucketing. **That does not apply at your scale.**

### 4.6 Splits and replay

- **Splits:** use separate train, val and test manifests ([skill data L14-15][skill-data]). Never train the tokenizer on val or test text ([SKILL L70][skill]). Make the splits speaker- or call-disjoint, and keep a narrowband test set **(analysis)**.
- **Replay:** the blog advises blending in a slice of the model's other languages and re-checking them ([blog][blog]). For Hinglish, include monolingual Hindi (`hi-IN`) and English (`en-US`/`en-GB`) data as replay sources so the model does not regress on pure-language calls **(analysis)**.

---

## 5. Streaming specifics

### 5.1 How the checkpoint was trained for streaming

- **Encoder setup:**
  - `att_context_style: chunked_limited`
  - `conv_context_size: causal`
  - `causal_downsampling: true`
  - multi-lookahead `att_context_size: [[56,3],[56,0],[56,6],[56,13]]`
  - (`.nemo` config L595-617)
- **During training:** the encoder samples one context per step from that list, uniformly unless `att_context_probs` is set ([conformer_encoder L119-125, L665-672][encoder-train], [L943-991][encoder-ctx]).
- **During validation and inference:** it uses the **first** entry, `[56,3]` = 320 ms, unless changed. So in-training `val_wer` is measured at the 320 ms lookahead **(analysis from code)**.
- **Fine-tuning preserves all of this:** `speech_to_text_finetune.py` restores these settings from the `.nemo` (§2.2), and the blog says the fine-tuned model "is the same architecture as the base … you pick your latency/accuracy operating point at inference time via att_context_size" ([blog][blog]).
- **With the §2.5 alternative script** you must reproduce the settings exactly in YAML. Biasing `att_context_probs` toward your deployment latency is possible in principle, but no NVIDIA guidance exists for this model: UNVERIFIED.

### 5.2 True streaming evaluation

- **Script:** `examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py` ([docstring][stream-infer]; NeMo docs list it as the cache-aware simulation script, with multi-lookahead support through `att_context_size` lists, [featured_models.rst L60-90][docs-featured]) with `model_path`, `dataset_manifest`, `att_context_size="[56,0]"` (or your target), `target_lang=hi-IN|auto`, `strip_lang_tags=true|false`, `output_path`, `batch_size` ([card L399-431][card-nemo], [script config L160-205][stream-infer-cfg]). NVIDIA's notebook adds `decoder_type=rnnt pad_and_drop_preencoded=true` ([notebook][riva-nb]).
- **Precision:** the script defaults to `compute_dtype="float32"`, "since currently cache-aware models do not work with different dtype" ([L169-172][stream-infer-cfg]).
- **Offline equals streaming:** with `compare_vs_offline=true` the script runs both modes. Its code comments say the outputs "should be exactly the same" ([script][stream-infer-run]). The cache-aware design has no buffered-overlap approximation ([card L287][card-arch]). In other words, **for this model, offline inference at the same `att_context_size` reproduces streaming** **(analysis)**.
- **The real accuracy gap is between lookaheads.** For Hindi FLEURS with LangID, WER goes 8.13 / 7.97 / 7.41 / 7.05 / 6.81 at 80 / 160 / 320 / 560 / 1120 ms ([card L684][card-hi]).
- **Offline batch evaluation** (faster, same numbers at the same lookahead): `transcribe_speech.py` accepts `att_context_size` and `target_lang` ([transcribe L189-197, L321-322][transcribe]).
- **Scoring:** the streaming script's WER is **raw** (no normalization) and uses `word_error_rate` ([script L492-519][stream-infer-run]). Re-score its output JSON (`pred_text`/`text`) with `speech_to_text_eval.py only_score_manifest=true` and your normalization (§9).
- **Evaluate at the latency you will deploy** ([blog][blog]).

### 5.3 Inference prompt

- **What the script sets:** it calls `set_inference_prompt(target_lang or "auto")` and `set_strip_lang_tags(...)` ([script L383-387][stream-infer-run]). The tag-stripping regex is `\s*<[a-z]{2}-[A-Z]{2}>` ([rnnt_decoding L701-716][rnnt-decoding]).
- **Gap in training-time validation:** the restored `.nemo` decoding config has no `strip_lang_tags` (it defaults to False), so emitted tags would count as words in `val_wer` **(analysis)**.

---

## 6. Hardware and scale (single H200)

| Item | Finding | Source / basis |
|---|---|---|
| GPU family | Hopper is a listed supported microarchitecture | [card L594-601][card-sw] |
| Static memory | 638M FP32 params ≈ 2.55 GB weights. Adding gradients and AdamW m/v gives **≈ 10.2 GB** before activations. | **(analysis)**, param count from [HF API][hf-api] |
| Precision | Official config uses `trainer.precision: bf16` ([yaml L360][prompt-yaml-trainer]). NVIDIA's skill prefers `bf16-true` for memory, falling back if unstable ([skill train L32-34][skill-train]). | |
| Batch sizing | Duration-based: config `batch_duration: 400`, `quadratic_duration: 15`, `num_buckets: 30` ([yaml L150-152][prompt-yaml-train]); pre-training used 200 (`.nemo` L139). As a scale reference, docs suggest `batch_duration=1100`, `quadratic_duration=30` "for a hybrid RNN-T + CTC ASR model on a 32GB GPU" with about 15 s utterances ([dataloaders.rst L300-330][docs-dl]). | |
| Best practice | Run **OOMptimizer** (`scripts/speech_recognition/oomptimizer.py --pretrained-name … --buckets […]`) to get per-bucket `bucket_batch_size`. It uses up to 90% of GPU RAM by default; lower with `--memory-fraction` only after a real OOM ([dataloaders.rst L1372-1420][docs-dl], [skill data L80-99][skill-data]). Whether OOMptimizer handles the prompt model's extra language-index input is **UNVERIFIED**. | |
| Fused loss | The `.nemo` has `fuse_loss_wer: true`, `fused_batch_size: 2`. NVIDIA's skill says "Disable fused loss/WER" for RNNT fine-tuning ([skill arch L97-102][skill-arch]). `speech_to_text_finetune.py` does not apply YAML `joint` overrides (§2.2), so turning it off needs a code-level change: UNVERIFIED path. | |
| Throughput / epoch time | **No NVIDIA numbers for training throughput** of this model on any GPU: UNVERIFIED. The blog says only "an epoch is minutes, not hours" for its small set ([blog][blog]). The card's H100 numbers are *inference* concurrency (240 streams at 80 ms, 2,400 at 1.12 s) ([card L325-337][card]). | |
| Step budgeting | Audio hours per optimizer step ≈ `batch_duration × n_gpus × accumulate_grad_batches / 3600`. For example, 1,000 s/step on one GPU means one pass over 3,000 h takes about 10,800 steps. Measure real step time in a pilot and set `max_steps` from it. | **(analysis)** |
| Multi-GPU (more H200s) | The config already uses `DDPStrategy` with `devices: -1` ([yaml L348-369][prompt-yaml-trainer]). Lhotse requires `use_distributed_sampler=false` ([dataloaders.rst L347][docs-dl]). If launches hang, use `torchrun --nproc_per_node=N` with `trainer.devices=N` ([skill train L45-83][skill-train]). `batch_duration` is applied by each rank's sampler, so the global batch grows with GPU count **(analysis)**. NVIDIA gives no LR-scaling rule: UNVERIFIED. | |
| Full fine-tune vs light adaptation | NVIDIA's recipe is a "straightforward full fine-tune", and more in-language data "keeps helping" up to about 2,300 h ([blog][blog]). The skill suggests `lr=1e-4` for large or mixed runs and lower-LR refinement phases later ([skill train L5-13][skill-train], [skill refine L114-132][skill-refine]). With thousands of hours, a full fine-tune of all modules, including the encoder, is the documented path. Adapters/LoRA for this model are not documented: UNVERIFIED. | |
| 6 GB RTX 4050 laptop | Not viable for full fine-tuning: about 10 GB of static FP32 training state alone. Use it for data checks and small inference only. | **(analysis)** |

---

## 7. MLflow integration

### 7.1 How NeMo wires MLflow

- **Config fields:** `exp_manager.create_mlflow_logger` (default False) and `exp_manager.mlflow_logger_kwargs`, typed as `MLFlowParams` ([exp_manager L235-300][expman-params]).
- **`MLFlowParams` fields** ([mlflow_logger.py L20-37][mlflow-params]):
  - `experiment_name`
  - `run_name`
  - `tracking_uri`
  - `tags`
  - `save_dir` (`./mlruns`)
  - `prefix`
  - `artifact_location`
  - `run_id` (to resume)
  - `log_model: bool`
- **Logger class:** Lightning's own `lightning.pytorch.loggers.MLFlowLogger`, imported at exp_manager L39 and constructed as `MLFlowLogger(run_name=version, **mlflow_kwargs)` ([L1294-1300][expman-logger]).
- **Default experiment name:** if `experiment_name` is unset, it falls back to `exp_manager.name` with a warning ([L631-640][expman-name]).
- **Docs example:** [exp_manager.rst L117-137][docs-expman].

### 7.2 Blocking issues (verified in source; present in v3.0.0 and main)

1. **`TypeError` on startup.**
   - **Cause:** `mlflow_kwargs` is merged with the `MLFlowParams` schema, so it always contains `run_name`. Passing `run_name=version` explicitly as well raises "got multiple values for keyword argument 'run_name'".
   - **Status:** PR [#15261][pr15261] ("Resolve MLflow run_name duplicate argument error") is **open**. The same line is in the `v3.0.0` tag.
   - **Options:** patch NeMo locally per the PR; or disable `create_mlflow_logger` and attach a Lightning `MLFlowLogger` yourself in a thin training wrapper (future work, not written here).
2. **`MLFLOW_TRACKING_URI` is ignored.**
   - **Cause:** Lightning's signature defaults `tracking_uri=os.getenv("MLFLOW_TRACKING_URI")`, and `if not tracking_uri: tracking_uri = f"file:{save_dir}"` ([Lightning L115-152][pl-mlflow-init]). NeMo always passes `tracking_uri=None` from the schema, so the env default never applies and runs go to local `./mlruns` **(analysis from code)**.
   - **Fix:** set `exp_manager.mlflow_logger_kwargs.tracking_uri` explicitly. Hydra/OmegaConf `${oc.env:MLFLOW_TRACKING_URI}` interpolation works, because exp_manager resolves the config ([OmegaConf oc.env][omegaconf-env]).
3. **`log_model` cannot be `"all"` through NeMo.** It is typed `bool` in `MLFlowParams`, while Lightning accepts `True | False | "all"`. Expect OmegaConf to reject `"all"` (UNVERIFIED by running). `synchronous` is not exposed at all.
4. **MLflow is not installed by default.** It appears only in `[tool.uv] override-dependencies` (`mlflow>=3.11.1`), not as a NeMo dependency ([pyproject L439-470][pyproject]). Install it explicitly. Lightning raises `ModuleNotFoundError` without it ([Lightning][pl-mlflow-init]).

### 7.3 Auth environment variables (MLflow client, v3.17.2)

| Env var | Effect | Source |
|---|---|---|
| `MLFLOW_TRACKING_URI` | Server URL. With NeMo it must be set in the config; see 7.2. | [env vars L101-103][mlf-env] |
| `MLFLOW_TRACKING_USERNAME` + `MLFLOW_TRACKING_PASSWORD` | HTTP **Basic** auth. Both must be set. The env vars override `~/.mlflow/credentials` (INI section `[mlflow]`). | [credentials.py L36-73][mlf-creds], [basic-auth docs][mlf-basic-auth-doc] |
| `MLFLOW_TRACKING_TOKEN` | HTTP **Bearer** token | [rest_utils L262-263][mlf-rest-auth] |
| **Both set?** | Code: `if username and password: Basic … elif token: Bearer`, so **Basic wins and the token is not sent**. (The env-var docstring claims the token "takes precedence"; the code says otherwise.) | [rest_utils L259-263][mlf-rest-auth], [env vars L505-507][mlf-env] |
| `MLFLOW_TRACKING_AUTH` | Name of a registered auth-provider plugin (entry point `mlflow.request_auth_provider`). It is applied as a `requests` `auth=` object and overrides the Authorization header. | [rest_utils L292-295][mlf-rest-auth], [registry][mlf-auth-reg] |
| `MLFLOW_TRACKING_INSECURE_TLS` | Skip TLS verification. Cannot be combined with `MLFLOW_TRACKING_SERVER_CERT_PATH` (raises). | [rest_utils L765-812][mlf-rest] |
| `MLFLOW_TRACKING_SERVER_CERT_PATH` | CA bundle, used as `requests` `verify=` | [rest_utils L765-812][mlf-rest], [auth docs 3.3.0][mlf-auth-330] |
| `MLFLOW_TRACKING_CLIENT_CERT_PATH` | Client cert, used as `requests` `cert=` (mTLS) | [rest_utils L284-285][mlf-rest-auth] |
| `MLFLOW_HTTP_REQUEST_TIMEOUT` / `_MAX_RETRIES` | Default timeout 120 s | [env vars L233-257][mlf-env] |

**"Password + token" server.** The client sends exactly one `Authorization` header.
- If the server is MLflow's basic-auth app, use username and password.
- If it sits behind a proxy that expects a bearer token, set only the token.
- If it needs **both at once** (for example a gateway token *plus* MLflow basic auth), you would need a custom header plugin (`mlflow.request_header_provider` entry point, [registry][mlf-hdr-reg]) or an `MLFLOW_TRACKING_AUTH` provider. Ask the server admin which one is expected; this is UNVERIFIED for your server.
- **Process scope:** Lightning logs only from rank 0 (`rank_zero_only`), but export the env vars to every process ([Lightning L228-270][pl-mlflow-hparams]).

### 7.4 What gets logged

- **Params:** the flattened model config. NeMo calls `save_hyperparameters("cfg")` ([modelPT L131][modelpt-hparams]). Lightning truncates each value to 250 characters and sends batches of 100 ([Lightning L228-243][pl-mlflow-hparams]). Expect hundreds of params; the prompt dictionary and vocabulary list are each truncated.
- **Metrics:** `train_loss`, `learning_rate`, `global_step`, `training_batch_wer` and `val_wer` ([prompt model L460-580][prompt-model-logs]), plus step-timing metrics from exp_manager. `val_loss` appears only if `compute_eval_loss: true` (false in the `.nemo`).
- **Metric names:** characters outside `[a-zA-Z0-9_/. -]` are stripped ([Lightning][pl-mlflow-hparams]).

### 7.5 Artifacts and large files

- **`log_model=True` behavior:** at train end, Lightning uploads the **`.ckpt` files tracked by the checkpoint callback** (best-k plus last, or every save if `save_top_k == -1`) to `model/checkpoints/<stem>/`, together with `metadata.yaml` and `aliases.txt` ([Lightning L326-366][pl-mlflow-ckpt], [utilities L32-56][pl-utils]).
- **`.nemo` files are not uploaded** by `log_model` **(analysis from code)**.
- **Checkpoint size (analysis):** a `.ckpt` with optimizer state is roughly 7–8 GB (weights plus AdamW states), so `save_top_k: 5` means about 40 GB of uploads.
- **Upload path:** for a proxied artifact store (`mlflow-artifacts:`), the client does **one HTTP PUT per file** through the tracking server. Multipart applies only if `MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD=true` (default False) and the file is at least `MLFLOW_MULTIPART_UPLOAD_MINIMUM_FILE_SIZE` (500 MiB), in 10 MiB chunks, and the server's backend supports it ([http_artifact_repo L71-99][mlf-http-art], [env vars L471-477, L693-707][mlf-env]).
- **No hard limit in client code:** I found no client-side size limit. Reverse-proxy body-size limits on your server are UNVERIFIED.
- **Recommendation (analysis):** keep `log_model=false`. After training, upload only the chosen best `.nemo` (about 2.4 GB) plus eval reports as artifacts, and tag the run with the HF repo and commit.

---

## 8. Checkpointing and Hugging Face publishing

### 8.1 `exp_manager.checkpoint_callback_params`

- **Fields** ([exp_manager L145-176][expman-params]): `monitor`, `mode`, `save_top_k`, `save_last`, `always_save_nemo`, `save_best_model`, `save_nemo_on_train_end` (default True), `every_n_train_steps`, `filename`.
- **Official config:** `monitor: "val_wer"`, `mode: "min"`, `save_top_k: 5`, `always_save_nemo: True` ([yaml L372-388][prompt-yaml-trainer]).

| Setting combination | `.nemo` behavior | Source |
|---|---|---|
| `always_save_nemo=True`, `save_best_model=False` (**official config**) | On every checkpoint save, `pl_module.save_to(...)` writes the **current** weights, so the `.nemo` is the *latest* model, not the best | [checkpoint callback L212-253][ckpt-cb] |
| `always_save_nemo=True`, `save_best_model=True` | The `.nemo` is rewritten only when the best `.ckpt` changes | same |
| `save_best_model=True` at train end | Restores the best `.ckpt` and then writes the `.nemo` | [L255-306][ckpt-cb] |

- **NVIDIA's warning:** "Do not assume the latest `.nemo` is the best validation checkpoint … `always_save_nemo=true` can overwrite the `.nemo`". Evaluate the final `.nemo`, the best-checkpoint artifact and the averaged artifact with the same standalone command, and keep the winner ([skill train L113-124][skill-train]).
- **Averaging:** `scripts/checkpoint_averaging/checkpoint_averaging.py <best.nemo>` averages every non-`-last` `.ckpt` in that folder. On PyTorch 2.6 and later it may need `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`, for trusted files only ([skill train L88-111][skill-train]).
- **Stable validation:** the YAML's `validation_ds` does not set `default_prompt_mode`, and the dataset default is `unified`. Validation prompts are therefore random per run (50% `auto`), which makes `val_wer` noisy. The docs show `default_prompt_mode: langID` for validation; set it explicitly ([yaml L183-197][prompt-yaml-train], [dataset L84][prompt-ds-modes], [configs.rst L980-986][docs-configs-prompt]).

### 8.2 Push to Hugging Face Hub

- **`model.push_to_hf_hub(repo_id="org/name", pack_nemo_file=True, private=…, token=…, model_card=…, commit_message=…, branch=…)`** ([hf_io_mixin L119-235][hf-mixin]):
  - creates the repo (`exist_ok=True`)
  - saves `<name>.nemo`
  - writes `README.md` from your card or a generated template
  - calls `HfApi.upload_folder`
  - takes its token from the argument or from `huggingface_hub.get_token()` (the `HF_TOKEN` env var or a cached login)
- **Loading:** `ASRModel.from_pretrained("org/name")` looks for **`name.nemo` at the repo root**. If that file is absent, it downloads the whole repo as an unpacked NeMo directory ([common.py L1141-1200][hf-from-pretrained]). If you upload manually with `huggingface_hub`, name the file after the repo.
- **Transformers format:** NVIDIA's repo also ships Transformers weights (`model.safetensors`, `config.json`). A NeMo fine-tune produces only `.nemo`. A NeMo → `Nemotron3_5Asr` Transformers conversion script is UNVERIFIED (not located).

### 8.3 Model card and license obligations

- **Card metadata** ([HF model-card docs][hf-modelcards]):
  - `license: other`, `license_name: openmdw-1.1`, `license_link: https://openmdw.ai/license/1-1/` (the custom-license pattern; NVIDIA's card uses the same values, [card L1-6][card])
  - `base_model: nvidia/nemotron-3.5-asr-streaming-0.6b` and optionally `base_model_relation: finetune`
  - `library_name: nemo`, `language: [hi, en]`, `pipeline_tag: automatic-speech-recognition`
  - `model-index` eval results
- **Card body:** also document the transcript normalization, the expected `target_lang` and prompt mode, the recommended `att_context_size`, and how to strip tags.
- **OpenMDW-1.1 obligations:** ship a copy of the agreement and keep the copyright and origin notices ([OpenMDW-1.1][openmdw]). Put the license text in the repo.
- **Data rights:** OpenMDW makes you solely responsible for third-party rights and consents. Telephony recordings carry privacy and consent obligations, and a public repo exposes a model trained on them. This is a legal decision, not a technical one.

---

## 9. Evaluation for Hinglish

### 9.1 Metrics

| Metric | NeMo implementation | Hinglish notes |
|---|---|---|
| WER | `word_error_rate`: whitespace `split()` ([wer.py L36-70][wer]) | Primary metric. Sensitive to spelling variants and script choice for English words. |
| CER | `use_cer=True` uses `list(text)`: Unicode **code points**, including spaces ([wer.py L60-62][wer]) | Each Devanagari matra, virama and nukta counts as a character, so CER can't be compared with Latin-only CER. Report alongside WER. |
| MER (mixed error rate) | Not built into NeMo: UNVERIFIED | MER is mainly used for CJK-English mixing. For Hinglish, WER plus CER is standard **(analysis)**. |
| Per-script WER | Not built in | Recommended diagnostic **(analysis)**: WER over Devanagari words and over Latin words separately, and a "script-confusion" rate (correct word in the wrong script) |

### 9.2 NVIDIA's evaluation workflow

- **Default metric:** WER with capitalization and punctuation removed, scored from the saved prediction manifest ([skill eval-style][skill-evalstyle], [SKILL L72-74][skill]).
- **Step 1, transcribe:** `speech_to_text_eval.py model_path=… dataset_manifest=… output_filename=… amp=false compute_dtype=bfloat16`, optionally with `rnnt_decoding.greedy.use_cuda_graph_decoder=true` ([skill train L146-180][skill-train]).
- **Step 2, score:** `speech_to_text_eval.py dataset_manifest=<preds> only_score_manifest=true text_processing.do_lowercase=true text_processing.rm_punctuation=true use_cer=false` ([eval L90-185][eval]).
- **Streaming numbers:** use the cache-aware streaming script (§5.2), then score its output the same way.
- **Do not use AMP for evaluation** ([SKILL L72][skill]).
- **Compare models with the same standalone command:** baseline, fine-tuned and averaged ([skill refine L6-29][skill-refine]).

### 9.3 Normalization pitfalls for code-mixed text

- **Punctuation set:** `text_processing.punctuation_marks` defaults to `".,?"` ([eval L105-108][eval]). Add `।`, `!`, `-`, quotes and so on, or the danda counts as part of words. For broader Unicode punctuation, write a derived manifest with one documented normalizer applied to both `text` and `pred_text` ([skill eval-style L64-74][skill-evalstyle]).
- **`do_lowercase`** affects only Latin script.
- **Spelling variation** inflates WER: nukta presence (ज/ज़), chandrabindu/anusvara, and English loanwords written in Devanagari versus Latin (मीटिंग vs meeting). Fix the reference policy (§3.5). Consider a secondary "lenient" score that folds nukta and maps a curated loanword list **(analysis)**.
- **Numbers:** digits versus spoken words, and Devanagari versus ASCII digits.
- **Language tags:** strip `<xx-XX>` (`strip_lang_tags=true`) before scoring.
- **Normalizer quality:** NVIDIA itself notes that its WER normalization "is not perfect across all 40 language-locales" ([card L669][card-perf]).

### 9.4 Reference points (not targets)

- **FLEURS hi-IN** (Hindi only, not code-mixed): 8.13% WER at 80 ms with LangID, 11.47% with auto ([card L684][card-hi]).
- **Public Hindi-English code-switching corpus:** [arXiv 1810.00662][hi-en-corpus] (2018). Other Hinglish benchmarks (for example MUCS 2021) were not verified here: UNVERIFIED.
- **Your own held-out telephony test sets** (8 kHz-origin and 16 kHz-origin) are the only meaningful targets.

---

## 10. Pitfalls and open questions

### 10.1 Things that would block or change the plan

| # | Pitfall | Impact | Source |
|---|---|---|---|
| 1 | `exp_manager` MLflow `run_name` `TypeError` | MLflow logging crashes at startup; needs a patch or a custom logger | [exp_manager L1297][expman-logger], [PR #15261][pr15261] |
| 2 | `MLFLOW_TRACKING_URI` ignored by NeMo's logger | Runs silently land in local `./mlruns` | [Lightning L115-152][pl-mlflow-init], [MLFlowParams][mlflow-params] |
| 3 | Basic auth beats Bearer token when both are set | The token is never sent | [rest_utils L259-263][mlf-rest-auth] |
| 4 | `always_save_nemo` without `save_best_model` | The `.nemo` you publish may not be the best | [checkpoint callback][ckpt-cb] |
| 5 | Same-vocab-size tokenizer swap reloads old decoder/joint weights | WER stuck at 1.0 | [script L145-176][ft-script], [#15778][issue15778] |
| 6 | Validation prompt mode defaults to `unified` (random) | Noisy `val_wer` and checkpoint selection | [yaml][prompt-yaml-train], [dataset L84][prompt-ds-modes] |
| 7 | The fine-tune script ignores YAML encoder/joint/decoding | `fuse_loss_wer`, `strip_lang_tags` and lookaheads come from the `.nemo` | [script L197-224][ft-script] |
| 8 | Language-tag tokens in targets (append `<hi-IN>`?) | Tag emission may disappear or be mis-learned | UNVERIFIED; §3.4 |
| 9 | Devanagari digits map to `<unk>`; ZWJ/ZWNJ are absent from the vocab | Silent label corruption | (measured) §3.1 |
| 10 | NeMo version: the prompt model needs ≥ 3.0.0; container tag coverage UNVERIFIED; Python minimum differs (card 3.11, docs 3.12, pyproject 3.10) | Environment breakage | §2.4 |
| 11 | `[56,1]` (160 ms) is not among the trained lookaheads | Possible degradation at 160 ms | `.nemo` config, [encoder L993-1008][encoder-setctx] |
| 12 | Noam warmup with fine-tuning LRs; the official sources conflict | Risk of forgetting or instability | §2.3 |
| 13 | YAML `max_duration` 39.99 vs pre-training 20; Lhotse filters silently | Unexpected data loss or OOM | §4.3 |
| 14 | Online 8 kHz resampling | Slow dataloading | [telephony tutorial][tel-nb] |
| 15 | MLflow artifact sizes (`.ckpt` about 7–8 GB) | Slow or failed uploads | §7.5 |
| 16 | Data rights and privacy for publishing a telephony-trained model | Legal blocker for a public HF release | [OpenMDW-1.1][openmdw] |
| 17 | `speech_to_text_finetune.py` multi-GPU download guard multiplies `num_devices × num_devices` (a typo for nodes) and sleeps ≥ 60 s | Minor; prefer `init_from_nemo_model` with a pre-downloaded file | [script L94-112][ft-script] |

### 10.2 Decisions the user must make

1. **Transcript conventions:**
   - numbers (ASCII digits or spoken words)
   - punctuation and casing (keep PnC like the base model, or strip)
   - nukta and chandrabindu policy
   - an always-Latin rule for English words, including loanwords
   - whether targets end with a language tag
2. **Prompt strategy:** `target_lang=hi-IN` with `unified` (default), `prompt_mode: auto` for Hinglish sources (the YAML's code-switching hint), or a new prompt slot. Settle it with a small A/B test.
3. **Tokenizer:** keep the base tokenizer (recommended first), or train a new SentencePiece model (different size, decoder/joint re-initialized, add the lang tags as user symbols).
4. **Deployment latency:** the target `att_context_size` (80, 160, 320, 560 or 1120 ms), which drives the evaluation protocol.
5. **Data mix:** narrowband/wideband weighting, replay share of monolingual Hindi and English, segmentation rules, and speaker- or call-disjoint test sets per band.
6. **Learning-rate schedule:** NVIDIA notebook Noam (peak ≈ 3e-4) versus NeMo docs cosine at 1e-4 with 1–2% warmup.
7. **Compute:** a single H200 now versus multiple H200s later (DDP and LR policy), and the `max_steps` budget after a throughput pilot.
8. **NeMo pin:** `v3.0.0` tag, a specific `main` commit, or the `nemo-speech:26.07.00` container (verify that it contains `EncDecRNNTBPEModelWithPrompt`).
9. **MLflow:**
   - patch NeMo (PR #15261) or use a custom logger wrapper
   - which single auth scheme the server expects
   - artifact policy (no `.ckpt` uploads; final `.nemo` only)
10. **HF publishing:** public or private repo and organization; legal sign-off on data rights; whether a Transformers-format export is required.

---

## Recommended next steps

1. **Pin the environment.** NeMo Speech ≥ 3.0.0 (container or uv), `mlflow>=3.11.1`, and confirm that `nemo.collections.asr.models.EncDecRNNTBPEModelWithPrompt` imports.
2. **Run baselines** on a held-out Hinglish telephony set before any training:
   - offline at each `att_context_size`
   - with `target_lang ∈ {hi-IN, auto, en-US}` and `strip_lang_tags=false`
   - Inspect script choice, number style, PnC and tag emission. This settles decisions 1, 2 and 4.
3. **Prepare data:**
   - Freeze the normalization spec (§3.5).
   - Run the transcript-style audit.
   - Upsample 8 kHz to 16 kHz offline.
   - Segment to 20 s or less.
   - Tar the training set and leave val/test untarred.
   - Build a Lhotse `input_cfg` with Hinglish, monolingual Hindi and English replay, and per-source `prompt_mode` tags.
4. **Run a short pilot on the H200:**
   - base tokenizer
   - `init_from_nemo_model`
   - bf16
   - `validation_ds.default_prompt_mode=langID`
   - `save_best_model=true`
   - a small `max_steps`
   - Measure step time and memory, then size `batch_duration` or the OOMptimizer profile and set `max_steps`.
5. **Fix MLflow first.** Apply the #15261 change or plan a logger wrapper; set `tracking_uri` explicitly; choose Basic or Bearer; keep `log_model=false`.
6. **Run the full fine-tune** and sweep the LR schedule (Noam ≈ 3e-4 peak vs cosine 1e-4). Evaluate in streaming at the deployment `att_context_size` with the normalized scoring contract (§9).
7. **Select and average checkpoints** (keep the averaged model only if it is better). Publish to the HF Hub with an OpenMDW-1.1 card and `base_model`. Log the final `.nemo` and eval reports to MLflow.

---

## Appendix: local measurements (no model executed)

- **Read the `.nemo` header without downloading the checkpoint:** range-downloaded the first 2 MB of `nemotron-3.5-asr-streaming-0.6b.nemo` (a gzip tar) and extracted `model_config.yaml`, `tokenizer.model` and `vocab.txt`. The 2.55 GB `model_weights.ckpt` was not downloaded.
- **Inspected `tokenizer.model`** with `sentencepiece` (data parsing only). Findings:
  - piece counts by script and type
  - normalizer `nmt_nfkc`
  - model type UNIGRAM
  - no byte fallback
  - sample segmentations shown in §3.1
- **Inspected the HF `tokenizer.json`** with `tokenizers`. It is a BPE conversion and segments differently.
- **Scratch files:** all downloads and probe scripts live in the session scratchpad, not in this project.

---

## Sources

Primary sources (all accessed 2026-10-27):

- Model card: [`nvidia/nemotron-3.5-asr-streaming-0.6b` README @ea30d66][card]
- Model repo files: [HF API metadata][hf-api], [config.json][hf-config], [processor_config.json][hf-proc], [tokenizer.json][hf-tok], [.nemo checkpoint][hf-nemo]. Its `model_config.yaml` and `tokenizer.model` were read by range-downloading the first 2 MB of the tarball.
- NVIDIA blog: [How to Fine-Tune Nemotron 3.5 ASR for Your Language, Domain, or Accent (2026-06-04)][blog]
- NVIDIA Riva tutorial: [asr-finetune-nemotron-3.5-asr-streaming-prompt.ipynb @83ca000][riva-nb]
- License: [OpenMDW-1.1][openmdw]
- NeMo Speech: [v3.0.0 release notes][nemo-rel3], [prompt-model commit #15666][nemo-pr15666], plus source files at `main@2bf0867` linked inline
- NeMo agent skill: [nemo-speech-asr-finetune][skill]
- NeMo issues and PRs: [#15261 MLflow run_name fix (open)][pr15261], [#15778 tokenizer swap → WER 1.0][issue15778]
- NeMo telephony tutorial: [ASR_for_telephony_speech.ipynb (v2.7.3; also present on main)][tel-nb]
- Lightning 2.4.0: [loggers/mlflow.py][pl-mlflow], [loggers/utilities.py][pl-utils]
- MLflow v3.17.2 source and docs: [credentials.py][mlf-creds], [rest_utils.py][mlf-rest], [environment_variables.py][mlf-env], [http_artifact_repo.py][mlf-http-art], [request_auth registry][mlf-auth-reg], [request_header registry][mlf-hdr-reg], [basic-auth docs][mlf-basic-auth-doc], [auth docs 3.3.0][mlf-auth-330]
- Hugging Face Hub: [model cards / metadata][hf-modelcards]
- Unicode: [CompositionExclusions.txt][unicode-excl]
- OmegaConf: [`oc.env` resolver][omegaconf-env]
- Reference corpus: [Hindi-English Code-Switching Speech Corpus (arXiv 1810.00662)][hi-en-corpus]

[card]: https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/blob/ea30d66debe3740a08b573244286791d423d6b3e/README.md
[card-intro]: https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/blob/ea30d66debe3740a08b573244286791d423d6b3e/README.md#L233-L256
[card-release]: https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/blob/ea30d66debe3740a08b573244286791d423d6b3e/README.md#L244-L246
[card-langs]: https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/blob/ea30d66debe3740a08b573244286791d423d6b3e/README.md#L260-L279
[card-arch]: https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/blob/ea30d66debe3740a08b573244286791d423d6b3e/README.md#L283-L305
[card-nemo]: https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/blob/ea30d66debe3740a08b573244286791d423d6b3e/README.md#L382-L431
[card-tf]: https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/blob/ea30d66debe3740a08b573244286791d423d6b3e/README.md#L433-L440
[card-io]: https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/blob/ea30d66debe3740a08b573244286791d423d6b3e/README.md#L564-L586
[card-sw]: https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/blob/ea30d66debe3740a08b573244286791d423d6b3e/README.md#L590-L607
[card-version]: https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/blob/ea30d66debe3740a08b573244286791d423d6b3e/README.md#L614-L615
[card-data]: https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/blob/ea30d66debe3740a08b573244286791d423d6b3e/README.md#L617-L640
[card-perf]: https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/blob/ea30d66debe3740a08b573244286791d423d6b3e/README.md#L662-L698
[card-hi]: https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/blob/ea30d66debe3740a08b573244286791d423d6b3e/README.md#L684
[card-license]: https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/blob/ea30d66debe3740a08b573244286791d423d6b3e/README.md#L733-L735
[hf-api]: https://huggingface.co/api/models/nvidia/nemotron-3.5-asr-streaming-0.6b
[hf-config]: https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/blob/ea30d66debe3740a08b573244286791d423d6b3e/config.json
[hf-proc]: https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/blob/ea30d66debe3740a08b573244286791d423d6b3e/processor_config.json
[hf-tok]: https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/blob/ea30d66debe3740a08b573244286791d423d6b3e/tokenizer.json
[hf-nemo]: https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/blob/ea30d66debe3740a08b573244286791d423d6b3e/nemotron-3.5-asr-streaming-0.6b.nemo
[en-model]: https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b
[blog]: https://huggingface.co/blog/nvidia/fine-tuning-nemotron-35-asr
[riva-nb]: https://github.com/nvidia-riva/tutorials/blob/83ca0000ed1b9c906f7ab0a9060a6ef3456466b6/asr-finetune-nemotron-3.5-asr-streaming-prompt.ipynb
[openmdw]: https://openmdw.ai/license/1-1/
[nemo-rel3]: https://github.com/NVIDIA-NeMo/Speech/releases/tag/v3.0.0
[nemo-pr15666]: https://github.com/NVIDIA-NeMo/Speech/commit/6967f48fda2a776a68bbaed6a71d7fea78ccc3f6
[ft-script]: https://github.com/NVIDIA-NeMo/Speech/blob/2bf0867c24e5283389175065dba0c01246458844/examples/asr/speech_to_text_finetune.py#L83-L224
[prompt-yaml]: https://github.com/NVIDIA-NeMo/Speech/blob/2bf0867c24e5283389175065dba0c01246458844/examples/asr/conf/fastconformer/cache_aware_streaming/fastconformer_transducer_bpe_streaming_prompt.yaml
[prompt-yaml-defaults]: https://github.com/NVIDIA-NeMo/Speech/blob/2bf0867c24e5283389175065dba0c01246458844/examples/asr/conf/fastconformer/cache_aware_streaming/fastconformer_transducer_bpe_streaming_prompt.yaml#L17-L144
[prompt-yaml-train]: https://github.com/NVIDIA-NeMo/Speech/blob/2bf0867c24e5283389175065dba0c01246458844/examples/asr/conf/fastconformer/cache_aware_streaming/fastconformer_transducer_bpe_streaming_prompt.yaml#L146-L213
[prompt-yaml-model]: https://github.com/NVIDIA-NeMo/Speech/blob/2bf0867c24e5283389175065dba0c01246458844/examples/asr/conf/fastconformer/cache_aware_streaming/fastconformer_transducer_bpe_streaming_prompt.yaml#L239-L308
[prompt-yaml-trainer]: https://github.com/NVIDIA-NeMo/Speech/blob/2bf0867c24e5283389175065dba0c01246458844/examples/asr/conf/fastconformer/cache_aware_streaming/fastconformer_transducer_bpe_streaming_prompt.yaml#L335-L388
[prompt-model]: https://github.com/NVIDIA-NeMo/Speech/blob/2bf0867c24e5283389175065dba0c01246458844/nemo/collections/asr/models/rnnt_bpe_models_prompt.py
[prompt-model-transcribe]: https://github.com/NVIDIA-NeMo/Speech/blob/2bf0867c24e5283389175065dba0c01246458844/nemo/collections/asr/models/rnnt_bpe_models_prompt.py#L253-L294
[prompt-model-logs]: https://github.com/NVIDIA-NeMo/Speech/blob/2bf0867c24e5283389175065dba0c01246458844/nemo/collections/asr/models/rnnt_bpe_models_prompt.py#L460-L580
[prompt-ds-modes]: https://github.com/NVIDIA-NeMo/Speech/blob/2bf0867c24e5283389175065dba0c01246458844/nemo/collections/asr/data/audio_to_text_lhotse_prompt_index.py#L76-L146
[prompt-ds-getitem]: https://github.com/NVIDIA-NeMo/Speech/blob/2bf0867c24e5283389175065dba0c01246458844/nemo/collections/asr/data/audio_to_text_lhotse_prompt_index.py#L148-L165
[prompt-utils]: https://github.com/NVIDIA-NeMo/Speech/blob/2bf0867c24e5283389175065dba0c01246458844/nemo/collections/asr/parts/utils/prompt_language_utils.py#L33-L80
[mixins]: https://github.com/NVIDIA-NeMo/Speech/blob/2bf0867c24e5283389175065dba0c01246458844/nemo/collections/asr/parts/mixins/mixins.py#L938-L1034
[rnnt-train-script]: https://github.com/NVIDIA-NeMo/Speech/blob/2bf0867c24e5283389175065dba0c01246458844/examples/asr/asr_transducer/speech_to_text_rnnt_bpe_prompt.py
[modelpt-init]: https://github.com/NVIDIA-NeMo/Speech/blob/2bf0867c24e5283389175065dba0c01246458844/nemo/core/classes/modelPT.py#L1263-L1300
[modelpt-hparams]: https://github.com/NVIDIA-NeMo/Speech/blob/2bf0867c24e5283389175065dba0c01246458844/nemo/core/classes/modelPT.py#L131
[change-vocab]: https://github.com/NVIDIA-NeMo/Speech/blob/2bf0867c24e5283389175065dba0c01246458844/nemo/collections/asr/models/rnnt_bpe_models.py#L271-L386
[agg-tok]: https://github.com/NVIDIA-NeMo/Speech/blob/2bf0867c24e5283389175065dba0c01246458844/nemo/collections/common/tokenizers/aggregate_tokenizer.py#L43-L127
[spm-train]: https://github.com/NVIDIA-NeMo/Speech/blob/2bf0867c24e5283389175065dba0c01246458844/nemo/collections/common/tokenizers/sentencepiece_tokenizer.py#L640-L700
[tok-script]: https://github.com/NVIDIA-NeMo/Speech/blob/2bf0867c24e5283389175065dba0c01246458844/scripts/tokenizers/process_asr_text_tokenizer.py#L112-L175
[encoder-train]: https://github.com/NVIDIA-NeMo/Speech/blob/2bf0867c24e5283389175065dba0c01246458844/nemo/collections/asr/modules/conformer_encoder.py#L665-L672
[encoder-ctx]: https://github.com/NVIDIA-NeMo/Speech/blob/2bf0867c24e5283389175065dba0c01246458844/nemo/collections/asr/modules/conformer_encoder.py#L943-L991
[encoder-setctx]: https://github.com/NVIDIA-NeMo/Speech/blob/2bf0867c24e5283389175065dba0c01246458844/nemo/collections/asr/modules/conformer_encoder.py#L993-L1008
[stream-infer]: https://github.com/NVIDIA-NeMo/Speech/blob/2bf0867c24e5283389175065dba0c01246458844/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py#L15-L110
[stream-infer-cfg]: https://github.com/NVIDIA-NeMo/Speech/blob/2bf0867c24e5283389175065dba0c01246458844/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py#L160-L205
[stream-infer-run]: https://github.com/NVIDIA-NeMo/Speech/blob/2bf0867c24e5283389175065dba0c01246458844/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py#L383-L519
[transcribe]: https://github.com/NVIDIA-NeMo/Speech/blob/2bf0867c24e5283389175065dba0c01246458844/examples/asr/transcribe_speech.py#L185-L322
[eval]: https://github.com/NVIDIA-NeMo/Speech/blob/2bf0867c24e5283389175065dba0c01246458844/examples/asr/speech_to_text_eval.py#L90-L185
[wer]: https://github.com/NVIDIA-NeMo/Speech/blob/2bf0867c24e5283389175065dba0c01246458844/nemo/collections/asr/metrics/wer.py#L36-L70
[rnnt-decoding]: https://github.com/NVIDIA-NeMo/Speech/blob/2bf0867c24e5283389175065dba0c01246458844/nemo/collections/asr/parts/submodules/rnnt_decoding.py#L701-L716
[lhotse-dl-resample]: https://github.com/NVIDIA-NeMo/Speech/blob/2bf0867c24e5283389175065dba0c01246458844/nemo/collections/common/data/lhotse/dataloader.py#L866-L869
[lhotse-dl-cfg]: https://github.com/NVIDIA-NeMo/Speech/blob/2bf0867c24e5283389175065dba0c01246458844/nemo/collections/common/data/lhotse/dataloader.py#L186-L191
[segment]: https://github.com/NVIDIA-NeMo/Speech/blob/2bf0867c24e5283389175065dba0c01246458844/nemo/collections/asr/parts/preprocessing/segment.py#L183-L188
[tarred-script]: https://github.com/NVIDIA-NeMo/Speech/blob/2bf0867c24e5283389175065dba0c01246458844/scripts/speech_recognition/convert_to_tarred_audio_dataset.py#L15-L80
[docs-ft]: https://github.com/NVIDIA-NeMo/Speech/blob/2bf0867c24e5283389175065dba0c01246458844/docs/source/asr/fine_tuning.rst
[docs-datasets]: https://github.com/NVIDIA-NeMo/Speech/blob/2bf0867c24e5283389175065dba0c01246458844/docs/source/asr/datasets.rst
[docs-dl]: https://github.com/NVIDIA-NeMo/Speech/blob/2bf0867c24e5283389175065dba0c01246458844/docs/source/dataloaders.rst
[docs-configs-cs]: https://github.com/NVIDIA-NeMo/Speech/blob/2bf0867c24e5283389175065dba0c01246458844/docs/source/asr/configs.rst#L344-L380
[docs-configs-prompt]: https://github.com/NVIDIA-NeMo/Speech/blob/2bf0867c24e5283389175065dba0c01246458844/docs/source/asr/configs.rst#L870-L1030
[docs-featured]: https://github.com/NVIDIA-NeMo/Speech/blob/2bf0867c24e5283389175065dba0c01246458844/docs/source/asr/featured_models.rst#L60-L90
[docs-install]: https://github.com/NVIDIA-NeMo/Speech/blob/2bf0867c24e5283389175065dba0c01246458844/docs/source/starthere/install.rst
[docs-expman]: https://github.com/NVIDIA-NeMo/Speech/blob/2bf0867c24e5283389175065dba0c01246458844/docs/source/core/exp_manager.rst#L93-L137
[pyproject]: https://github.com/NVIDIA-NeMo/Speech/blob/2bf0867c24e5283389175065dba0c01246458844/pyproject.toml
[skill]: https://github.com/NVIDIA-NeMo/Speech/blob/2bf0867c24e5283389175065dba0c01246458844/.claude/skills/nemo-speech-asr-finetune/SKILL.md
[skill-setup]: https://github.com/NVIDIA-NeMo/Speech/blob/2bf0867c24e5283389175065dba0c01246458844/.claude/skills/nemo-speech-asr-finetune/references/setup-checkpoints.md
[skill-data]: https://github.com/NVIDIA-NeMo/Speech/blob/2bf0867c24e5283389175065dba0c01246458844/.claude/skills/nemo-speech-asr-finetune/references/data-lhotse.md
[skill-arch]: https://github.com/NVIDIA-NeMo/Speech/blob/2bf0867c24e5283389175065dba0c01246458844/.claude/skills/nemo-speech-asr-finetune/references/architecture-tokenizer-metrics.md
[skill-train]: https://github.com/NVIDIA-NeMo/Speech/blob/2bf0867c24e5283389175065dba0c01246458844/.claude/skills/nemo-speech-asr-finetune/references/training-evaluation.md
[skill-evalstyle]: https://github.com/NVIDIA-NeMo/Speech/blob/2bf0867c24e5283389175065dba0c01246458844/.claude/skills/nemo-speech-asr-finetune/references/evaluation-style-contract.md
[skill-refine]: https://github.com/NVIDIA-NeMo/Speech/blob/2bf0867c24e5283389175065dba0c01246458844/.claude/skills/nemo-speech-asr-finetune/references/refinement-iteration.md
[expman-params]: https://github.com/NVIDIA-NeMo/Speech/blob/2bf0867c24e5283389175065dba0c01246458844/nemo/utils/exp_manager.py#L145-L300
[expman-name]: https://github.com/NVIDIA-NeMo/Speech/blob/2bf0867c24e5283389175065dba0c01246458844/nemo/utils/exp_manager.py#L631-L640
[expman-logger]: https://github.com/NVIDIA-NeMo/Speech/blob/2bf0867c24e5283389175065dba0c01246458844/nemo/utils/exp_manager.py#L1294-L1300
[mlflow-params]: https://github.com/NVIDIA-NeMo/Speech/blob/2bf0867c24e5283389175065dba0c01246458844/nemo/utils/loggers/mlflow_logger.py#L20-L37
[ckpt-cb]: https://github.com/NVIDIA-NeMo/Speech/blob/2bf0867c24e5283389175065dba0c01246458844/nemo/utils/callbacks/nemo_model_checkpoint.py#L212-L306
[hf-mixin]: https://github.com/NVIDIA-NeMo/Speech/blob/2bf0867c24e5283389175065dba0c01246458844/nemo/core/classes/mixins/hf_io_mixin.py#L119-L235
[hf-from-pretrained]: https://github.com/NVIDIA-NeMo/Speech/blob/2bf0867c24e5283389175065dba0c01246458844/nemo/core/classes/common.py#L1141-L1200
[lr-sched]: https://github.com/NVIDIA-NeMo/Speech/blob/2bf0867c24e5283389175065dba0c01246458844/nemo/core/optim/lr_scheduler.py#L526-L577
[pr15261]: https://github.com/NVIDIA-NeMo/Speech/pull/15261
[issue15778]: https://github.com/NVIDIA-NeMo/Speech/issues/15778
[tel-nb]: https://github.com/NVIDIA-NeMo/Speech/blob/v2.7.3/tutorials/asr/ASR_for_telephony_speech.ipynb
[pl-mlflow]: https://github.com/Lightning-AI/pytorch-lightning/blob/2.4.0/src/lightning/pytorch/loggers/mlflow.py
[pl-mlflow-init]: https://github.com/Lightning-AI/pytorch-lightning/blob/2.4.0/src/lightning/pytorch/loggers/mlflow.py#L115-L152
[pl-mlflow-ckpt]: https://github.com/Lightning-AI/pytorch-lightning/blob/2.4.0/src/lightning/pytorch/loggers/mlflow.py#L326-L366
[pl-mlflow-hparams]: https://github.com/Lightning-AI/pytorch-lightning/blob/2.4.0/src/lightning/pytorch/loggers/mlflow.py#L228-L270
[pl-utils]: https://github.com/Lightning-AI/pytorch-lightning/blob/2.4.0/src/lightning/pytorch/loggers/utilities.py#L32-L56
[mlf-creds]: https://github.com/mlflow/mlflow/blob/v3.17.2/mlflow/utils/credentials.py#L36-L73
[mlf-rest]: https://github.com/mlflow/mlflow/blob/v3.17.2/mlflow/utils/rest_utils.py#L765-L812
[mlf-rest-auth]: https://github.com/mlflow/mlflow/blob/v3.17.2/mlflow/utils/rest_utils.py#L256-L295
[mlf-env]: https://github.com/mlflow/mlflow/blob/v3.17.2/mlflow/environment_variables.py
[mlf-http-art]: https://github.com/mlflow/mlflow/blob/v3.17.2/mlflow/store/artifact/http_artifact_repo.py#L71-L99
[mlf-auth-reg]: https://github.com/mlflow/mlflow/blob/v3.17.2/mlflow/tracking/request_auth/registry.py
[mlf-hdr-reg]: https://github.com/mlflow/mlflow/blob/v3.17.2/mlflow/tracking/request_header/registry.py
[mlf-basic-auth-doc]: https://mlflow.org/docs/latest/self-hosting/security/basic-http-auth/
[mlf-auth-330]: https://www.mlflow.org/docs/3.3.0/ml/auth/
[hf-modelcards]: https://huggingface.co/docs/hub/model-cards
[unicode-excl]: https://www.unicode.org/Public/UCD/latest/ucd/CompositionExclusions.txt
[omegaconf-env]: https://omegaconf.readthedocs.io/en/2.3_branch/custom_resolvers.html#oc-env
[hi-en-corpus]: https://arxiv.org/abs/1810.00662
