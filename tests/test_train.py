from pathlib import Path

import pytest

from hinglish_asr.train import (
    apply_overrides,
    context_probs,
    load_config,
    pick_batch_duration,
    resolve_optim,
    steps_for,
)

CONFIG = Path(__file__).parent.parent / "configs" / "train.yaml"
TRAINED = [[56, 3], [56, 0], [56, 6], [56, 13]]  # the .nemo's att_context_size order


def test_repo_config_resolves_for_every_arm():
    for arm in ("noam", "cosine"):
        cfg = load_config(CONFIG, arm)
        assert cfg["arm"] == arm
        assert cfg["model"]["att_context_size"] == [56, 0]
        assert cfg["model"]["validation_ds"]["default_prompt_mode"] == "auto"
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
