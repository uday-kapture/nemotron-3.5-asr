import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from hinglish_asr import train
from hinglish_asr.train import (
    _child,
    big_tmpdir,
    last_error,
    apply_overrides,
    best_val_wer,
    context_probs,
    load_config,
    normalize_val_wer,
    pick_batch_duration,
    resolve_optim,
    setup_optimization,
    steps_for,
)

CONFIG = Path(__file__).parent.parent / "configs" / "train.yaml"
TRAINED = [[56, 3], [56, 0], [56, 6], [56, 13]]  # the .nemo's att_context_size order


def test_repo_config_resolves_for_every_arm():
    for arm in ("noam", "cosine"):
        cfg = load_config(CONFIG, arm)
        assert cfg["arm"] == arm
        assert cfg["model"]["att_context_size"] == [56, 0]
        assert cfg["model"]["validation_ds"]["default_prompt_mode"] == "langID"  # the rows' hi-IN
        assert cfg["model"]["train_ds"]["default_prompt_mode"] == "langID"
        assert resolve_optim(cfg["model"]["optim"])["lr"] > 0
        assert sum(context_probs(TRAINED, cfg["model"]["att_context_probs"])) == pytest.approx(1.0)
        assert "arms" not in cfg


def test_arm_merges_without_dropping_siblings():
    cfg = load_config(CONFIG, "cosine")
    assert cfg["model"]["optim"]["betas"] == [0.9, 0.98]  # from the base section
    assert cfg["model"]["optim"]["sched"]["name"] == "CosineAnnealing"


def test_unknown_arm_is_an_error():
    with pytest.raises(ValueError, match="arm"):
        load_config(CONFIG, "adam")


def test_overrides_parse_yaml_values():
    cfg = apply_overrides({"trainer": {"max_steps": 1}}, ["trainer.max_steps=60000", "model.x=[1, 2]", "a=null"])
    assert cfg == {"trainer": {"max_steps": 60000}, "model": {"x": [1, 2]}, "a": None}
    with pytest.raises(ValueError):
        apply_overrides({}, ["no-equals"])


def test_noam_peak_lr_is_where_the_schedule_peaks():
    optim = resolve_optim({"peak_lr": 3e-4, "sched": {"name": "NoamAnnealing", "d_model": 1024, "warmup_steps": 1000}})
    # NeMo's Noam multiplier at step == warmup: d_model^-0.5 * warmup^-0.5
    assert optim["lr"] * 1024**-0.5 * 1000**-0.5 == pytest.approx(3e-4)
    assert "peak_lr" not in optim
    assert resolve_optim({"peak_lr": 1e-4, "sched": {"name": "CosineAnnealing"}})["lr"] == 1e-4
    with pytest.raises(ValueError):
        resolve_optim({"name": "adamw"})


def test_context_probs_follow_the_models_order():
    assert context_probs(TRAINED, {"56,0": 0.7, "56,3": 0.3}) == [0.3, 0.7, 0.0, 0.0]
    with pytest.raises(ValueError, match="lookaheads"):
        context_probs(TRAINED, {"70,6": 1.0})
    with pytest.raises(ValueError, match="sum to 1"):
        context_probs(TRAINED, {"56,0": 0.5})


def test_steps_for_budget():
    assert steps_for(train_hours=2300, passes=3, audio_s_per_step=400) == 62100


def test_pick_batch_duration_takes_fastest_under_memory_cap():
    common = {"status": "ok", "total_memory_gb": 140.0}
    results = [
        {**common, "batch_duration": 300, "audio_s_per_s": 900.0, "peak_reserved_gb": 70.0},
        {**common, "batch_duration": 500, "audio_s_per_s": 1100.0, "peak_reserved_gb": 110.0},
        {**common, "batch_duration": 600, "audio_s_per_s": 1200.0, "peak_reserved_gb": 130.0},  # over 85%
        {"status": "oom", "batch_duration": 800},
    ]
    assert pick_batch_duration(results)["batch_duration"] == 500
    assert pick_batch_duration([{"status": "oom", "batch_duration": 300}]) is None
    assert pick_batch_duration([{"status": "ok", "batch_duration": 300}]) is None  # no steps measured


def test_child_reads_its_log_after_exp_manager_moves_the_run_dir_files(tmp_path, monkeypatch):
    cfg = tmp_path / "train.yaml"
    cfg.write_text(f"run_root: {tmp_path / 'runs'}\n")
    run_dir = tmp_path / "runs" / "pilot-bd600"

    def child(cmd, stdout, stderr):
        # What exp_manager's check_resume does to loose files, then an OOM.
        (run_dir / "run_0").mkdir(parents=True)
        for f in run_dir.iterdir():
            if f.is_file():
                shutil.move(str(f), str(run_dir / "run_0"))
        stdout.write("torch.OutOfMemoryError: CUDA out of memory\n")
        return subprocess.CompletedProcess(cmd, 1)

    monkeypatch.setattr(train.subprocess, "run", child)
    args = argparse.Namespace(config=str(cfg))
    r = _child(args, "pilot-bd600", "pilot", None, ["model.x=1"], None)
    assert r["status"] == "oom" and "out of memory" in r["log_tail"]
    assert r["error"] == "torch.OutOfMemoryError: CUDA out of memory"
    assert r["log"] == str(tmp_path / "runs" / "logs" / "pilot-bd600.log")
    assert (tmp_path / "runs" / "logs" / "pilot-bd600.log").exists()


def _fake_child(tmp_path, monkeypatch, returncode, write_result, stale=None):
    cfg = tmp_path / "train.yaml"
    cfg.write_text(f"run_root: {tmp_path / 'runs'}\n")
    run_dir = tmp_path / "runs" / "pilot-bd300"
    run_dir.mkdir(parents=True, exist_ok=True)
    if stale is not None:  # left by an earlier attempt
        (run_dir / "result.json").write_text(json.dumps(stale))
        os.utime(run_dir / "result.json", (0, 0))

    def child(cmd, stdout, stderr):
        stdout.write("Trainer.fit stopped: max_steps=300 reached.\n")
        if write_result:
            (run_dir / "result.json").write_text(json.dumps({"name": "pilot-bd300", "throughput": {"s_per_step": 0.2}}))
        return subprocess.CompletedProcess(cmd, returncode)

    monkeypatch.setattr(train.subprocess, "run", child)
    return _child(argparse.Namespace(config=str(cfg)), "pilot-bd300", "pilot", None, [], None)


def test_a_run_that_wrote_its_result_counts_even_if_it_crashed_on_exit(tmp_path, monkeypatch):
    r = _fake_child(tmp_path, monkeypatch, returncode=-11, write_result=True)
    assert r["status"] == "ok" and r["throughput"] == {"s_per_step": 0.2} and r["returncode"] == -11


def test_a_stale_result_from_an_earlier_attempt_does_not_count(tmp_path, monkeypatch):
    r = _fake_child(tmp_path, monkeypatch, returncode=1, write_result=False, stale={"name": "old"})
    assert r["status"] == "failed" and r["returncode"] == 1


@pytest.mark.parametrize(
    "tail,line",
    [
        ("Traceback (most recent call last):\n  File \"x.py\", line 1\nKeyError: 'val_wer'\n", "KeyError: 'val_wer'"),
        ("x\nhinglish_asr.tracking.TrackingError: MLFLOW_TRACKING_URI is not set\n[NeMo W] done\n",
         "hinglish_asr.tracking.TrackingError: MLFLOW_TRACKING_URI is not set"),
        ("nothing named here\nlast words\n", "last words"),
        ("", "(empty log)"),
    ],
)
def test_last_error_finds_the_line_naming_the_failure(tail, line):
    assert last_error(tail) == line


class _Hub:
    def __init__(self, ranked):
        self.ranked = ranked

    def list(self):
        return self.ranked


def _trainer(io, best):
    return SimpleNamespace(strategy=SimpleNamespace(checkpoint_io=io),
                           checkpoint_callback=SimpleNamespace(best_model_score=best))


def test_best_val_wer_comes_from_the_hub_when_checkpoints_live_there():
    # After a resume NeMo has dropped the hub checkpoints from its top k: best_model_score is None.
    assert best_val_wer(_trainer(SimpleNamespace(hub=_Hub([("a", 0.18), ("b", 0.2)])), None)) == 0.18
    assert best_val_wer(_trainer(SimpleNamespace(hub=_Hub([])), None)) is None
    assert best_val_wer(_trainer(SimpleNamespace(), 0.25)) == 0.25


def test_val_wer_scores_normalized_text():
    decoding = SimpleNamespace(decode_ids_to_str=lambda ids: "Order cancel करो।", other="kept")
    hyp = SimpleNamespace(text="order, Cancel करो <hi-IN>")
    wer = SimpleNamespace(decoding=decoding, decode=lambda *a: [hyp, [SimpleNamespace(text="OK!")]])
    normalize_val_wer(wer)
    assert wer.decoding.decode_ids_to_str([1]) == "order cancel करो"
    hyps = wer.decode(None, None, None, None)
    assert hyps[0].text == "order cancel करो" and hyps[1][0].text == "ok"
    assert wer.decoding.other == "kept"


def test_schedule_spans_schedule_steps_but_training_stops_at_max_steps():
    seen = []
    loop = SimpleNamespace(max_steps=4000)
    model = SimpleNamespace(trainer=SimpleNamespace(fit_loop=SimpleNamespace(epoch_loop=loop)),
                            setup_optimization=lambda optim: seen.append((loop.max_steps, optim["lr"])))
    setup_optimization(model, {"lr": 1e-4}, 60000)
    setup_optimization(model, {"lr": 1e-4}, None)
    assert seen == [(60000, 1e-4), (4000, 1e-4)]
    assert loop.max_steps == 4000


def test_temp_files_go_next_to_the_runs_unless_tmpdir_is_set(tmp_path, monkeypatch):
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", None)
    monkeypatch.delenv("TMPDIR", raising=False)
    assert big_tmpdir(tmp_path / "runs") == (tmp_path / "runs" / "tmp").resolve()
    assert tempfile.gettempdir() == str((tmp_path / "runs" / "tmp").resolve())
    monkeypatch.setenv("TMPDIR", str(tmp_path / "big"))  # set but missing: created, not ignored
    assert big_tmpdir(tmp_path / "runs") == (tmp_path / "big").resolve() and (tmp_path / "big").is_dir()
