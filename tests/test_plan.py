import json

import yaml

from hinglish_asr.plan import config_sha, main, plan_files


def _stats():
    def shard(dg_hours, other_hours=1.0):
        vendors = {"assemblyai_universal-3-5-pro": {"rows": 1, "seconds": other_hours * 3600}}
        if dg_hours:
            vendors["deepgram_nova-3"] = {"rows": 1, "seconds": dg_hours * 3600}
        return {"rows": 2, "seconds": (dg_hours + other_hours) * 3600, "vendors": vendors}

    return {
        **{f"data/combined/c{i}.parquet": shard(1.0) for i in range(10)},
        "data/combined/no_dg.parquet": shard(0),
        "data/inhouse-asr-2/d.parquet": shard(0.5, 0),
        "data/vaani/v.parquet": shard(2.0),
    }


def test_plan_keeps_vendor_files_of_chosen_subsets_numbered_in_order():
    files = plan_files(_stats(), subsets=["combined", "inhouse-asr-2"], vendor="deepgram_nova-3", seed="s")
    paths = {f["path"] for f in files}
    assert len(files) == 11
    assert "data/combined/no_dg.parquet" not in paths and "data/vaani/v.parquet" not in paths
    assert [f["index"] for f in files] == list(range(11))


def test_plan_order_is_seeded_and_budget_cuts_a_prefix():
    kw = dict(subsets=["combined"], vendor="deepgram_nova-3")
    a = plan_files(_stats(), seed="s", **kw)
    assert a == plan_files(_stats(), seed="s", **kw)
    assert [f["path"] for f in a] != sorted(f["path"] for f in a)  # shuffled
    cut = plan_files(_stats(), seed="s", max_vendor_hours=3.0, **kw)
    assert [f["path"] for f in cut] == [f["path"] for f in a[:3]]


def test_plan_skips_files_missing_at_the_revision_and_records_sizes():
    sizes = {f"data/combined/c{i}.parquet": 100 + i for i in range(10) if i != 4}
    files = plan_files(_stats(), subsets=["combined"], vendor="deepgram_nova-3", seed="s", sizes=sizes)
    assert len(files) == 9
    assert all(f["bytes"] == sizes[f["path"]] for f in files)
    assert [f["index"] for f in files] == list(range(9))


def test_cli_embeds_config(tmp_path):
    stats = tmp_path / "stats"
    stats.mkdir()
    (stats / "host.json").write_text(json.dumps({"shards": _stats()}))
    cfg = {
        "dataset": "org/ds", "revision": "a" * 40, "subsets": ["combined"], "vendor": "deepgram_nova-3",
        "max_vendor_hours": None, "plan_seed": "s",
    }
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    plan = main(["--config", str(tmp_path / "cfg.yaml"), "--out", str(tmp_path / "plan.json"),
                 "--stats-dir", str(stats), "--no-sizes"])
    saved = json.loads((tmp_path / "plan.json").read_text())
    assert saved == plan
    assert saved["config"] == cfg
    assert saved["totals"]["files"] == 10 and saved["totals"]["vendor_hours"] == 10.0


def test_raising_the_hours_cap_keeps_the_config_hash_and_extends_the_plan():
    kw = dict(subsets=["combined"], vendor="deepgram_nova-3", seed="s")
    small, big = plan_files(_stats(), max_vendor_hours=3.0, **kw), plan_files(_stats(), **kw)
    assert big[: len(small)] == small
    cfg = {"plan_seed": "s", "min_confidence": 0.0, "max_vendor_hours": 800}
    assert config_sha(cfg) == config_sha({**cfg, "max_vendor_hours": None})
    assert config_sha(cfg) != config_sha({**cfg, "min_confidence": 0.5})
