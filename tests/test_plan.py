import json

import pytest
import yaml

from hinglish_asr.plan import config_sha, main, plan_files, usable_hours


def _shard(hi=1.0, en=0.0, ta=0.0, test_share=0.0):
    langs = {k: {"rows": 1, "seconds": h * 3600} for k, h in (("hi", hi), ("en", en), ("ta", ta)) if h}
    seconds = (hi + en + ta) * 3600
    splits = {"train": {"rows": 1, "seconds": seconds * (1 - test_share)}}
    if test_share:
        splits["test"] = {"rows": 1, "seconds": seconds * test_share}
    return {"rows": 2, "seconds": seconds, "languages": langs, "splits": splits}


def _stats():
    return {
        **{f"data/combined/c{i}.parquet": _shard(1.0, 0.5, 1.0) for i in range(10)},
        "data/combined/tamil.parquet": _shard(0, 0, 3.0),
        **{f"data/spring-inx/s{i}.parquet": _shard(2.0, test_share=0.5) for i in range(4)},
        "data/mucs-hinglish/m.parquet": _shard(5.0, test_share=1.0),
        "data/vaani/v.parquet": _shard(2.0),
    }


KW = dict(subsets={"combined": 0.75, "spring-inx": 0.25}, languages=["hi", "en"], seed="s")


def test_usable_hours_count_chosen_languages_outside_test_splits():
    assert usable_hours(_shard(1.0, 0.5, 1.0), ["hi", "en"]) == pytest.approx(1.5)
    assert usable_hours(_shard(2.0, test_share=0.5), ["hi"]) == pytest.approx(1.0)
    assert usable_hours(_shard(5.0, test_share=1.0), ["hi"]) == 0
    assert usable_hours({"seconds": 3600, "languages": {"hi": {"seconds": 3600}}}, ["hi"]) == 1.0  # no split stats


def test_plan_keeps_files_of_chosen_subsets_with_usable_hours_numbered_in_order():
    files = plan_files(_stats(), **KW)
    paths = {f["path"] for f in files}
    assert len(files) == 14
    assert "data/combined/tamil.parquet" not in paths and "data/vaani/v.parquet" not in paths
    assert [f["index"] for f in files] == list(range(14))
    assert {f["subset"] for f in files} == {"combined", "spring-inx"}


def test_plan_mixes_subsets_by_share():
    files = plan_files(_stats(), max_hours=8.0, **KW)
    hours = {s: sum(f["hours"] for f in files if f["subset"] == s) for s in KW["subsets"]}
    assert hours["combined"] == pytest.approx(6.0) and hours["spring-inx"] == pytest.approx(2.0)


def test_a_small_share_waits_until_its_file_fits():
    files = plan_files(_stats(), max_hours=3.0, **KW)
    assert {f["subset"] for f in files} == {"combined"}  # a 1 h spring-inx file would be a third of 3 h


def test_plan_order_is_seeded_and_a_subset_that_runs_out_stops():
    a = plan_files(_stats(), **KW)
    assert a == plan_files(_stats(), **KW)
    combined = [f["path"] for f in a if f["subset"] == "combined"]
    assert combined != sorted(combined)  # shuffled
    assert sum(f["subset"] == "spring-inx" for f in a) == 4


def test_plan_skips_files_missing_at_the_revision_and_records_sizes():
    sizes = {f"data/combined/c{i}.parquet": 100 + i for i in range(10) if i != 4}
    files = plan_files(_stats(), subsets={"combined": 1.0}, languages=["hi"], seed="s", sizes=sizes)
    assert len(files) == 9
    assert all(f["bytes"] == sizes[f["path"]] for f in files)
    assert [f["index"] for f in files] == list(range(9))


def test_cli_embeds_config(tmp_path):
    stats = tmp_path / "stats"
    stats.mkdir()
    (stats / "host.json").write_text(json.dumps({"shards": _stats()}))
    cfg = {
        "dataset": "org/ds", "revision": "a" * 40, "subsets": {"combined": 0.75, "spring-inx": 0.25},
        "languages": ["hi", "en"], "max_hours": None, "plan_seed": "s",
    }
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    plan = main(["--config", str(tmp_path / "cfg.yaml"), "--out", str(tmp_path / "plan.json"),
                 "--stats-dir", str(stats), "--no-sizes"])
    saved = json.loads((tmp_path / "plan.json").read_text())
    assert saved == plan
    assert saved["config"] == cfg
    assert saved["totals"]["files"] == 14 and saved["totals"]["hours"] == 19.0
    assert saved["totals"]["hours_by_subset"] == {"combined": 15.0, "spring-inx": 4.0}


def test_cli_refuses_a_subset_without_a_share(tmp_path):
    cfg = {"dataset": "org/ds", "revision": "a" * 40, "subsets": {"combined": 1.0, "vaani": 0},
           "languages": ["hi"], "plan_seed": "s"}
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    with pytest.raises(SystemExit, match="positive share"):
        main(["--config", str(tmp_path / "cfg.yaml"), "--out", str(tmp_path / "plan.json"),
              "--stats-dir", str(tmp_path), "--no-sizes"])


def test_raising_the_hours_cap_keeps_the_config_hash_and_extends_the_plan():
    small, big = plan_files(_stats(), max_hours=5.0, **KW), plan_files(_stats(), **KW)
    assert big[: len(small)] == small
    cfg = {"plan_seed": "s", "min_confidence": 0.0, "max_hours": 800}
    assert config_sha(cfg) == config_sha({**cfg, "max_hours": None})
    assert config_sha(cfg) != config_sha({**cfg, "min_confidence": 0.5})
