"""Fine-tune the prompt model on the converted calls: pilot, LR sweep, full run.

    hx() { local m=$1; shift; uv run --extra nemo --extra mlflow python -m "hinglish_asr.$m" "$@"; }

    # 1. pilot: ~300 steps per batch size -> batch_duration, speed, max_steps
    hx train pilot --config configs/train.yaml --arm cosine --batch-durations 300 400 500 600 --passes 3

    # 2. sweep: equal-length runs, one per arm, each on the first steps of the
    #    full run's LR schedule (length from runs/pilot.json); compare best val_wer
    hx train sweep --config configs/train.yaml --arms noam cosine --set model.train_ds.batch_duration=500

    # 3. the full run with the sweep's arm and the pilot's max_steps
    hx train run --config configs/train.yaml --arm cosine --name calls-v1 \\
        --set model.train_ds.batch_duration=500 --set trainer.max_steps=60000

Follows NeMo's speech_to_text_finetune.py (restore the .nemo; set data,
optimizer and SpecAugment from the config; fit) and adds what it leaves out:

- validation at the deployment lookahead (80 ms) with the `hi-IN` prompt, and
  val_wer on the benchmark's normalization (tags, case, punctuation);
- `att_context_probs`: how often each trained lookahead is used per batch;
- validation loss, top-k + last checkpoints by val_wer, one .nemo at the end;
- an MLflow logger with an explicit tracking URI (`tracking`);
- a throughput callback (s/step, audio s/step, peak GPU memory).

Runs live in <run_root>/<name>/ (config.yaml, checkpoints/, mlflow_run_id,
result.json); pilot and sweep write each child's output to
<run_root>/logs/<name>.log. Rerunning the same name resumes from its last
checkpoint.

With `checkpoint_hub_repo` set, top-k checkpoints are never written locally:
each goes straight to runs/<name>/ in that private HF model repo as weights
only, and the repo keeps the k best (`hub`). Only the -last checkpoint, with
optimizer state for resuming, stays on disk. NeMo's end-of-training .nemo
(which reloads the best checkpoint from disk) is off then; the benchmark
writes winner.nemo instead.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

SAMPLE_RATE = 16000


# ---------- config ----------


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def apply_overrides(cfg: dict, sets: list[str]) -> dict:
    """`a.b.c=value` strings; value is parsed as YAML (numbers, lists, null)."""
    cfg = copy.deepcopy(cfg)
    for spec in sets:
        if "=" not in spec:
            raise ValueError(f"--set wants KEY=VALUE, got {spec!r}")
        key, value = spec.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)
    return cfg


def load_config(path: str | Path, arm: str | None = None, sets: list[str] = ()) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    arms = cfg.pop("arms", {})
    if arm is not None:
        if arm not in arms:
            raise ValueError(f"arm {arm!r} not in {sorted(arms)}")
        cfg = deep_merge(cfg, arms[arm])
    cfg["arm"] = arm
    return apply_overrides(cfg, list(sets))


def resolve_optim(optim: dict) -> dict:
    """NeMo optim config; `peak_lr` becomes the `lr` that peaks there.

    NoamAnnealing scales lr by d_model^-0.5 * min(step^-0.5, step * warmup^-1.5),
    which peaks at step = warmup with lr / sqrt(d_model * warmup).
    """
    optim = copy.deepcopy(optim)
    peak = optim.pop("peak_lr", None)
    if peak is not None:
        sched = optim.get("sched", {})
        if sched.get("name") == "NoamAnnealing":
            optim["lr"] = peak * math.sqrt(sched["d_model"]) * math.sqrt(sched["warmup_steps"])
        else:
            optim["lr"] = peak
    if "lr" not in optim:
        raise ValueError("optim needs lr or peak_lr (pick an --arm)")
    return optim


def context_probs(all_contexts: list[list[int]], probs: dict[str, float]) -> list[float]:
    """{"56,0": p, ...} -> probabilities in the encoder's att_context_size order."""
    keys = [f"{left},{right}" for left, right in all_contexts]
    unknown = set(probs) - set(keys)
    if unknown:
        raise ValueError(f"att_context_probs keys {sorted(unknown)} not among the model's lookaheads {keys}")
    out = [float(probs.get(k, 0.0)) for k in keys]
    if abs(sum(out) - 1.0) > 1e-6:
        raise ValueError(f"att_context_probs must sum to 1, got {sum(out)}")
    return out


def steps_for(train_hours: float, passes: float, audio_s_per_step: float) -> int:
    return math.ceil(passes * train_hours * 3600 / audio_s_per_step)


def pick_batch_duration(results: list[dict], memory_fraction: float = 0.85) -> dict | None:
    """Fastest pilot (audio seconds per wall second) that kept peak memory under the cap."""
    ok = [
        r for r in results  # an ok run that measured no steps has no throughput numbers
        if r.get("status") == "ok" and "peak_reserved_gb" in r
        and r["peak_reserved_gb"] <= memory_fraction * r["total_memory_gb"]
    ]
    return max(ok, key=lambda r: r["audio_s_per_s"], default=None)


# ---------- training (needs the nemo extra) ----------


def _throughput_callback(out_path: Path, skip_steps: int = 20):
    import lightning.pytorch as pl
    import torch

    class Throughput(pl.Callback):
        """Wall time between consecutive steps (data loading included), minus validation."""

        def __init__(self):
            self.last = None
            self.step_s: list[float] = []
            self.audio_s: list[float] = []

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            now = time.perf_counter()
            if self.last is not None and trainer.global_step > skip_steps:
                self.step_s.append(now - self.last)
                self.audio_s.append(float(batch[1].sum()) / SAMPLE_RATE)
            self.last = now

        def on_validation_start(self, trainer, pl_module):
            self.last = None

        def summary(self) -> dict:
            if not self.step_s:
                return {}
            props = torch.cuda.get_device_properties(0)
            s, a = sum(self.step_s), sum(self.audio_s)
            return {
                "steps_measured": len(self.step_s),
                "s_per_step": s / len(self.step_s),
                "audio_s_per_step": a / len(self.audio_s),
                "audio_s_per_s": a / s,
                "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
                "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
                "total_memory_gb": props.total_memory / 1e9,
                "gpu": props.name,
            }

        def on_train_end(self, trainer, pl_module):
            out_path.write_text(json.dumps(self.summary(), indent=1) + "\n")

    return Throughput()


def _notify_callback(chat, name: str, stage: str, cfg: dict, check_loss_every: int = 50):
    import lightning.pytorch as pl
    import torch

    from . import notify

    every = cfg.get("notify", {}).get("every_n_steps") or 0

    class Notify(pl.Callback):
        """Google Chat: start, progress every n steps, each epoch's validation, a non-finite loss."""

        def __init__(self):
            self.created = time.perf_counter()
            self.step = None  # last completed step; None until training starts
            self.bad_loss = False

        def _reset_window(self, step):
            self.mark, self.mark_step, self.audio_s, self.val_s = time.perf_counter(), step, 0.0, 0.0

        def on_train_start(self, trainer, pl_module):
            self.step = self.first_step = trainer.global_step  # > 0 when resuming
            self.started = time.perf_counter()
            self.best = best_val_wer(trainer)  # checkpoint state is restored by now: a resumed run keeps its best
            self._reset_window(self.step)
            gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU"
            batch_duration = cfg["model"]["train_ds"].get("batch_duration")
            chat.send(notify.started_text(name, stage, cfg.get("arm"), self.step, trainer.max_steps, batch_duration, gpu))

        def on_validation_start(self, trainer, pl_module):
            self.val_t0 = time.perf_counter()

        def on_validation_end(self, trainer, pl_module):
            if not trainer.sanity_checking:
                self.val_s += time.perf_counter() - self.val_t0

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            step = trainer.global_step
            self.audio_s += float(batch[1].sum()) / SAMPLE_RATE
            if step == self.step:  # gradient accumulation: same optimizer step
                return
            self.step = step
            report = every and step % every == 0
            loss = None
            if report or step % check_loss_every == 0:
                loss = float(outputs["loss"]) if isinstance(outputs, dict) else float(outputs)
                if not math.isfinite(loss) and not self.bad_loss:
                    chat.send(notify.bad_loss_text(name, step, loss))
                self.bad_loss = not math.isfinite(loss)
            if not report:
                return
            now = time.perf_counter()
            steps, train_s = step - self.mark_step, now - self.mark - self.val_s
            done = step - self.first_step  # ETA from the whole run so far, validation included
            eta = (trainer.max_steps - step) * (now - self.started) / done if trainer.max_steps > 0 and done else None
            chat.send(notify.progress_text(
                name, step, trainer.max_steps, loss, trainer.optimizers[0].param_groups[0]["lr"],
                train_s / steps if steps else None, self.audio_s / train_s if train_s > 0 else None, eta,
            ))
            self._reset_window(step)

        def on_train_epoch_end(self, trainer, pl_module):
            metrics = trainer.callback_metrics
            value = lambda k: float(metrics[k]) if k in metrics else None  # noqa: E731
            val_wer, best = value("val_wer"), self.best
            if val_wer is not None and (best is None or val_wer < best):
                self.best = val_wer
            limit = trainer.limit_train_batches
            epochs = math.ceil(trainer.max_steps / limit) if isinstance(limit, int) and trainer.max_steps > 0 else None
            chat.send(notify.epoch_text(name, trainer.current_epoch + 1, epochs, trainer.global_step,
                                        val_wer, value("val_loss"), best))

    return Notify()


def _hub_checkpoint_io(hub, tmp_dir: Path):
    import torch
    from lightning.pytorch.plugins.io import TorchCheckpointIO

    from .hub import VAL_WER

    class HubCheckpointIO(TorchCheckpointIO):
        """Top-k checkpoints go to the hub as weights only; everything else is saved as usual."""

        def save_checkpoint(self, checkpoint, path, storage_options=None):
            path = Path(path)
            if path.stem.endswith("-last") or not VAL_WER.search(path.stem):
                return super().save_checkpoint(checkpoint, path, storage_options)
            tmp_dir.mkdir(parents=True, exist_ok=True)
            tmp = tmp_dir / path.name
            torch.save({"state_dict": checkpoint["state_dict"], "global_step": checkpoint.get("global_step")}, tmp)
            try:
                uploaded = hub.add(tmp, path.name)
            finally:
                tmp.unlink(missing_ok=True)
            print(f"{path.name}: {'uploaded to ' + hub.repo if uploaded else 'not in the hub top-k, skipped'}",
                  file=sys.stderr)

        def remove_checkpoint(self, path):
            # Lightning drops a checkpoint that fell out of its top-k; the hub
            # trims itself on upload, so only a local file (if any) goes.
            super().remove_checkpoint(path)

    io = HubCheckpointIO()
    io.hub = hub
    return io


def best_val_wer(trainer) -> float | None:
    """Best val_wer of the run so far.

    With hub checkpoints, NeMo's checkpoint callback forgets the top k on
    resume (it drops entries with no local file), so the hub's ranking is
    the record.
    """
    hub = getattr(trainer.strategy.checkpoint_io, "hub", None)
    if hub is not None:
        ranked = hub.list()
        return ranked[0][1] if ranked else None
    ckpt = trainer.checkpoint_callback
    return float(ckpt.best_model_score) if ckpt is not None and ckpt.best_model_score is not None else None


def normalize_val_wer(wer) -> None:
    """Score NeMo's val_wer on text normalized as the benchmark scores it.

    NeMo compares raw strings, so case and punctuation count as errors and
    the top-k kept by val_wer need not be the checkpoints the benchmark ranks
    best. References and hypotheses both go through
    `scoring.normalize_for_scoring` (tags, case, punctuation) instead.
    """
    from .scoring import normalize_for_scoring

    class References:
        def __init__(self, decoding):
            self._decoding = decoding

        def decode_ids_to_str(self, ids):
            return normalize_for_scoring(self._decoding.decode_ids_to_str(ids))

        def __getattr__(self, name):
            return getattr(self._decoding, name)

    decode = wer.decode

    def decode_normalized(*args):
        hyps = decode(*args)
        for h in hyps:
            for one in h if isinstance(h, list) else [h]:
                one.text = normalize_for_scoring(one.text)
        return hyps

    wer.decoding = References(wer.decoding)
    wer.decode = decode_normalized


def setup_optimization(model, optim: dict, schedule_steps: int | None) -> None:
    """model.setup_optimization, with the LR schedule spanning `schedule_steps`.

    NeMo sizes the schedule by trainer.max_steps. A sweep run stops early but
    should follow the start of the full run's schedule, not a squeezed copy
    that anneals within the sweep (and so flatters the arms that anneal).
    """
    from omegaconf import OmegaConf

    loop = model.trainer.fit_loop.epoch_loop
    stop = loop.max_steps
    if schedule_steps:
        loop.max_steps = schedule_steps
    try:
        model.setup_optimization(OmegaConf.create(optim))
    finally:
        loop.max_steps = stop


def _setup_model(model, cfg: dict) -> dict:
    from nemo.collections.asr.models import ASRModel
    from omegaconf import OmegaConf, open_dict

    m = cfg["model"]
    enc = model.encoder
    applied = {"att_context_size_all": enc.att_context_size_all}
    if m.get("att_context_probs"):
        probs = context_probs(enc.att_context_size_all, m["att_context_probs"])
        enc.att_context_probs = probs
        with open_dict(model.cfg):
            model.cfg.encoder.att_context_probs = probs
        applied["att_context_probs"] = probs
    # Eval (validation) uses the default lookahead; training samples by probs.
    enc.set_default_att_context_size(m["att_context_size"])
    model.compute_eval_loss = m["compute_eval_loss"]
    with open_dict(model.cfg):
        model.cfg.compute_eval_loss = m["compute_eval_loss"]
    model.decoding.set_strip_lang_tags(True)
    normalize_val_wer(model.wer)
    if m.get("fused_batch_size"):
        model.joint.set_fused_batch_size(m["fused_batch_size"])
    applied["fused_batch_size"] = model.joint.fused_batch_size
    applied["loss"] = type(model.loss._loss).__name__ if hasattr(model.loss, "_loss") else type(model.loss).__name__

    train_ds = OmegaConf.to_container(model.cfg.train_ds, resolve=True)
    train_ds.update(m["train_ds"])
    train_ds.update(input_cfg=cfg["data"]["input_cfg"], manifest_filepath=None, tarred_audio_filepaths=None)
    val_ds = OmegaConf.to_container(model.cfg.validation_ds, resolve=True)
    val_ds.update(m["validation_ds"])
    val_ds["manifest_filepath"] = cfg["data"]["val_manifest"]
    with open_dict(model.cfg):
        model.cfg.train_ds = train_ds
        model.cfg.validation_ds = val_ds
    model.setup_training_data(OmegaConf.create(train_ds))
    model.setup_multiple_validation_data(OmegaConf.create(val_ds))
    optim = dict(m["optim"])
    schedule_steps = optim.pop("schedule_steps", None)
    setup_optimization(model, resolve_optim(optim), schedule_steps)
    applied["schedule_steps"] = schedule_steps or model.trainer.max_steps
    model.spec_augment = ASRModel.from_config_dict(OmegaConf.create(m["spec_augment"]))
    return applied


def fit(cfg: dict, name: str, stage: str, parent_run_id: str | None = None) -> dict:
    """One run; with GCHAT_WEBHOOK_URL set, its progress and any error go to Google Chat (pilot runs stay quiet)."""
    from . import notify

    chat = notify.GChat.from_env(thread=name)
    if stage == "pilot" or not chat.enabled:
        return _fit(cfg, name, stage, parent_run_id, [])
    progress = _notify_callback(chat, name, stage, cfg)
    try:
        result = _fit(cfg, name, stage, parent_run_id, [progress])
    except BaseException as e:
        chat.send(notify.failed_text(name, e, progress.step))
        raise
    chat.send(notify.finished_text(name, progress.step, result["best_val_wer"], time.perf_counter() - progress.created))
    return result


def big_tmpdir(run_root: Path) -> Path:
    """Temp files go to $TMPDIR, or <run_root>/tmp when it is unset; created if missing.

    Lightning writes each checkpoint to a temp file before moving it into place
    and NeMo unpacks a .nemo into one: a checkpoint with optimizer state is
    ~8 GB, more than a small /tmp holds (ENOSPC at the first -last save).
    """
    tmp = Path(os.environ.get("TMPDIR") or (run_root / "tmp")).resolve()
    tmp.mkdir(parents=True, exist_ok=True)  # a missing TMPDIR would silently fall back to /tmp
    os.environ["TMPDIR"] = str(tmp)  # dataloader workers and other children too
    tempfile.tempdir = None  # gettempdir() caches its first answer
    return tmp


def _fit(cfg: dict, name: str, stage: str, parent_run_id: str | None, callbacks: list) -> dict:
    import lightning.pytorch as pl
    from nemo.collections.asr.models import ASRModel
    from nemo.utils.exp_manager import exp_manager
    from omegaconf import OmegaConf

    from . import tracking

    run_dir = Path(cfg["run_root"]) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    big_tmpdir(Path(cfg["run_root"]))
    # exp_manager moves every loose file in run_dir into run_<n>/ (keeping the
    # previous attempt's logs), so this run's files are read before it and
    # written after it.
    id_file = run_dir / "mlflow_run_id"
    # Continue the MLflow run only when training continues too: without a
    # -last checkpoint the run restarts at step 0 and gets a fresh MLflow run.
    resumable = any((run_dir / "checkpoints").glob("*-last.ckpt"))
    mlflow_run_id = id_file.read_text().strip() if id_file.exists() and resumable else None
    throughput = _throughput_callback(run_dir / "throughput.json")
    ckpt_params, plugins = dict(cfg["checkpoint"]), []
    hub_repo = cfg.get("checkpoint_hub_repo")
    if hub_repo and stage != "pilot":
        from .hub import HubCheckpoints

        hub = HubCheckpoints(hub_repo, name, keep=ckpt_params["save_top_k"])
        plugins.append(_hub_checkpoint_io(hub, run_dir / "upload-tmp"))
        ckpt_params.update(save_best_model=False, save_nemo_on_train_end=False)
    trainer = pl.Trainer(
        **cfg["trainer"], logger=False, enable_checkpointing=False, callbacks=[throughput, *callbacks], plugins=plugins
    )
    # exp_manager refuses a trainer that already has a logger alongside
    # explicit_log_dir, so MLflow is attached after it.
    exp_manager(
        trainer,
        OmegaConf.create(
            {
                "explicit_log_dir": str(run_dir),
                "create_tensorboard_logger": False,
                "create_mlflow_logger": False,
                "create_wandb_logger": False,
                "create_checkpoint_callback": stage != "pilot",
                "checkpoint_callback_params": ckpt_params,
                "resume_if_exists": True,
                "resume_ignore_no_checkpoint": True,
            }
        ),
    )
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    if plugins:
        (run_dir / "checkpoint_hub.json").write_text(json.dumps({"repo": hub_repo, "run": name}) + "\n")
    tags = {"stage": stage, "arm": str(cfg.get("arm")), "run_dir": str(run_dir.resolve())}
    if parent_run_id:
        tags[tracking.PARENT_TAG] = parent_run_id
    resuming = mlflow_run_id is not None
    logger = tracking.lightning_logger(cfg["mlflow"]["experiment"], name, tags=tags, run_id=mlflow_run_id)
    trainer.loggers = [logger]
    if resuming:
        id_file.write_text(logger.run_id + "\n")

    model = ASRModel.restore_from(cfg["base_model"], map_location="cpu")
    model.set_trainer(trainer)
    applied = _setup_model(model, cfg)
    if not resuming:  # MLflow params are write-once; config.yaml records any change
        logger.log_hyperparams({**tracking.flatten(cfg), **tracking.flatten({"applied": applied})})
        # Recorded only once the params are in: a run that failed during setup starts a fresh MLflow run.
        id_file.write_text(logger.run_id + "\n")
    trainer.fit(model)

    ckpt = trainer.checkpoint_callback
    result = {
        "name": name,
        "stage": stage,
        "arm": cfg.get("arm"),
        "mlflow_run_id": logger.run_id,
        "best_val_wer": best_val_wer(trainer),
        "best_model_path": getattr(ckpt, "best_model_path", None) if ckpt is not None else None,
        "checkpoint_hub_repo": hub_repo if stage != "pilot" else None,
        "applied": applied,
        "throughput": throughput.summary(),
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=1, default=str) + "\n")
    return result


# ---------- drivers ----------


def _child(args, name: str, stage: str, arm: str | None, sets: list[str], parent: str | None) -> dict:
    """One training run in its own process, so GPU memory is fully released between runs."""
    cmd = [sys.executable, "-m", "hinglish_asr.train", "run", "--config", args.config, "--name", name, "--stage", stage]
    if arm:
        cmd += ["--arm", arm]
    if parent:
        cmd += ["--parent-run-id", parent]
    for s in sets:
        cmd += ["--set", s]
    run_root = Path(load_config(args.config, arm, sets)["run_root"])
    run_dir = run_root / name
    # Outside run_dir: exp_manager moves every loose file there into run_<n>/ at start.
    log = run_root / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"{name}: running, log in {log}", file=sys.stderr)
    started = time.time()
    with log.open("w") as fh:
        returncode = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
    exit_text = f"killed by {signal.Signals(-returncode).name}" if returncode < 0 else f"exit code {returncode}"
    result = run_dir / "result.json"
    # result.json is the run's last step: a fresh one means it finished, even if
    # the process then crashed on the way out (CUDA graphs, dataloader workers).
    if result.exists() and result.stat().st_mtime >= started:
        if returncode != 0:
            print(f"{name}: finished, then the process ended with {exit_text}; keeping its result\n  full log: {log}",
                  file=sys.stderr)
        return {"status": "ok", **json.loads(result.read_text()), "returncode": returncode}
    if returncode < 0 and stage != "pilot":  # killed by a signal: the run could not post its own error
        from . import notify

        notify.GChat.from_env(thread=name).send(notify.killed_text(name, signal.Signals(-returncode).name))
    tail = log.read_text(errors="replace")[-4000:]
    status = "oom" if "out of memory" in tail.lower() else "failed"
    error = last_error(tail)
    print(f"{name}: {status} ({exit_text}): {error}\n  full log: {log}", file=sys.stderr)
    return {"status": status, "name": name, "returncode": returncode, "error": error, "log": str(log),
            "log_tail": tail[-500:]}


_ERROR_LINE = re.compile(r"\b\w*(Error|Exception|Interrupt)\b.*:|^error\b", re.IGNORECASE)


def last_error(log_tail: str) -> str:
    """The line naming what went wrong at the end of a log ("KeyError: 'x'"), else its last line."""
    lines = [line.strip() for line in log_tail.splitlines() if line.strip()]
    for line in reversed(lines):
        if _ERROR_LINE.search(line):
            return line[:300]
    return lines[-1][:300] if lines else "(empty log)"


def pilot(args) -> dict:
    cfg = load_config(args.config, args.arm, args.set)
    train_hours = args.train_hours
    if train_hours is None:
        report = json.loads(Path(cfg["data"]["report"]).read_text())
        train_hours = report["splits"]["train"]["hours"]
    results = []
    for bd in args.batch_durations:
        sets = [
            *args.set,
            f"model.train_ds.batch_duration={bd}",
            f"trainer.max_steps={args.steps}",
            "trainer.limit_val_batches=0",
            "trainer.num_sanity_val_steps=0",
        ]
        r = _child(args, f"pilot-bd{bd:g}", "pilot", args.arm, sets, None)
        row = {"batch_duration": bd, "status": r["status"], **r.get("throughput", {})}
        if r["status"] != "ok":
            row.update(returncode=r["returncode"], error=r["error"], log=r["log"])
        results.append(row)
        print(json.dumps(row), file=sys.stderr)
    best = pick_batch_duration(results, args.memory_fraction)
    out = {"train_hours": train_hours, "passes": args.passes, "results": results, "chosen": best}
    if best:
        out["max_steps"] = steps_for(train_hours, args.passes, best["audio_s_per_step"])
        out["est_hours"] = out["max_steps"] * best["s_per_step"] / 3600
    path = Path(cfg["run_root"]) / "pilot.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {path}", file=sys.stderr)
    if best:
        print(f"batch_duration={best['batch_duration']}  max_steps={out['max_steps']}  "
              f"(~{out['est_hours']:.1f} h for {args.passes} passes over {train_hours:.0f} h)", file=sys.stderr)
    from . import notify

    notify.GChat.from_env(thread="pilot").send(pilot_text(out))
    return out


def pilot_text(out: dict) -> str:
    lines = ["*pilot* done"]
    for r in out["results"]:
        if r["status"] == "ok":
            lines.append(f"• batch {r['batch_duration']:g} s: {r['audio_s_per_s']:,.0f}× realtime, "
                         f"{r['s_per_step']:.2f} s/step, peak {r['peak_reserved_gb']:.0f} GB")
        else:
            lines.append(f"• batch {r['batch_duration']:g} s: {r['status']}")
    if out.get("chosen"):
        lines.append(f"chosen: batch {out['chosen']['batch_duration']:g} s → max_steps {out['max_steps']:,}, "
                     f"~{out['est_hours']:.1f} h for {out['passes']:g} passes")
    else:
        lines.append("🔴 no batch size worked")
    return "\n".join(lines)


def sweep(args) -> dict:
    from . import tracking

    cfg = load_config(args.config, None, args.set)
    schedule_steps = args.schedule_steps
    pilot_file = Path(cfg["run_root"]) / "pilot.json"
    if schedule_steps is None and pilot_file.exists():
        schedule_steps = json.loads(pilot_file.read_text()).get("max_steps")
    if schedule_steps is None:
        sys.exit(f"sweep needs the full run's length for the LR schedule: --schedule-steps N, or run pilot first ({pilot_file})")
    sets = [*args.set, f"model.optim.schedule_steps={schedule_steps}"]
    if args.steps:
        sets.append(f"trainer.max_steps={args.steps}")
    with tracking.run(cfg["mlflow"]["experiment"], args.name, stage="sweep") as parent:
        tracking.log_params({"arms": args.arms, "overrides": args.set, "steps": args.steps, "schedule_steps": schedule_steps})
        results = [_child(args, f"{args.name}-{arm}", "sweep", arm, sets, parent.info.run_id) for arm in args.arms]
        done = [r for r in results if r["status"] == "ok" and r.get("best_val_wer") is not None]
        winner = min(done, key=lambda r: r["best_val_wer"], default=None)
        out = {"results": results, "winner": winner and winner["arm"]}
        for r in done:
            tracking.log_metrics({f"best_val_wer/{r['arm']}": r["best_val_wer"]})
    path = Path(cfg["run_root"]) / f"{args.name}.json"
    path.write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(json.dumps({r.get("arm") or r["name"]: r.get("best_val_wer", r["status"]) for r in results}), file=sys.stderr)
    from . import notify

    notify.GChat.from_env(thread=args.name).send(sweep_text(args.name, out))
    return out


def sweep_text(name: str, out: dict) -> str:
    lines = [f"*{name}* done"]
    for r in out["results"]:
        label = r.get("arm") or r["name"]
        wer = r.get("best_val_wer")
        lines.append(f"• {label}: best val_wer {wer:.2%}" if wer is not None else f"• {label}: {r['status']}")
    lines.append(f"winner: {out['winner']}" if out["winner"] else "🔴 no arm finished")
    return "\n".join(lines)


def main(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--config", default="configs/train.yaml")
        sp.add_argument("--set", action="append", default=[], help="KEY=VALUE override, e.g. trainer.max_steps=20000")

    r = sub.add_parser("run", help="one training run")
    common(r)
    r.add_argument("--name", required=True)
    r.add_argument("--arm")
    r.add_argument("--stage", default="train", choices=["train", "sweep", "pilot"])
    r.add_argument("--parent-run-id")

    pl_ = sub.add_parser("pilot", help="short runs per batch_duration: speed, memory, max_steps")
    common(pl_)
    pl_.add_argument("--arm", default="cosine", help="any arm; the schedule barely matters for speed")
    pl_.add_argument("--batch-durations", type=float, nargs="+", default=[300, 400, 500, 600])
    pl_.add_argument("--steps", type=int, default=300)
    pl_.add_argument("--passes", type=float, default=3.0, help="passes over the training hours for max_steps")
    pl_.add_argument("--train-hours", type=float, help="default: data.report's train hours")
    pl_.add_argument("--memory-fraction", type=float, default=0.85)

    s = sub.add_parser("sweep", help="equal-length runs, one per arm")
    common(s)
    s.add_argument("--arms", nargs="+", default=["noam", "cosine"])
    s.add_argument("--steps", type=int, help="default: trainer.max_steps in the config")
    s.add_argument("--schedule-steps", type=int,
                   help="length of the full run's LR schedule, which each arm follows for --steps (default: pilot.json)")
    s.add_argument("--name", default="sweep")

    args = p.parse_args(argv)
    if args.cmd == "run":
        return fit(load_config(args.config, args.arm, args.set), args.name, args.stage, args.parent_run_id)
    if args.cmd == "pilot":
        return pilot(args)
    return sweep(args)


if __name__ == "__main__":
    main()
