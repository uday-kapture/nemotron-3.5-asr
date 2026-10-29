import pytest
import yaml

from hinglish_asr.input_cfg import InputCfgError, build_entry, main


def _tarred(d, n):
    (d / "sharded_manifests").mkdir(parents=True)
    for i in range(n):
        (d / f"audio_{i}.tar").write_bytes(b"")
        (d / "sharded_manifests" / f"manifest_{i}.json").write_text("")
    return d


def test_tarred_dir_becomes_nemo_tarred_entry_with_brace_ranges(tmp_path):
    d = _tarred(tmp_path / "hinglish", 4)
    e = build_entry(str(d), weight=0.7, prompt_mode="auto")
    assert e == {
        "type": "nemo_tarred",
        "manifest_filepath": f"{d}/sharded_manifests/manifest__OP_0..3_CL_.json",
        "tarred_audio_filepath": f"{d}/audio__OP_0..3_CL_.tar",
        "weight": 0.7,
        "tags": {"prompt_mode": "auto"},
    }


def test_missing_shard_is_an_error(tmp_path):
    d = _tarred(tmp_path / "x", 3)
    (d / "audio_1.tar").unlink()
    with pytest.raises(InputCfgError, match="audio_1.tar"):
        build_entry(str(d), weight=1.0)


def test_missing_shard_manifest_is_an_error(tmp_path):
    d = _tarred(tmp_path / "x", 2)
    (d / "sharded_manifests" / "manifest_1.json").unlink()
    with pytest.raises(InputCfgError, match="manifest_1.json"):
        build_entry(str(d), weight=1.0)


def test_plain_manifest_becomes_nemo_entry_without_tags_when_no_mode(tmp_path):
    m = tmp_path / "val_manifest.jsonl"
    m.write_text("")
    assert build_entry(str(m), weight=1.0) == {"type": "nemo", "manifest_filepath": str(m), "weight": 1.0}


def test_unknown_prompt_mode_is_rejected(tmp_path):
    m = tmp_path / "m.jsonl"
    m.write_text("")
    with pytest.raises(InputCfgError, match="prompt_mode"):
        build_entry(str(m), weight=1.0, prompt_mode="codeswitch")


def test_cli_writes_yaml_list(tmp_path):
    a = _tarred(tmp_path / "hinglish", 2)
    b = tmp_path / "hindi_replay.jsonl"
    b.write_text("")
    out = tmp_path / "input_cfg.yaml"
    main(["--entry", f"{a}:0.8:auto", "--entry", f"{b}:0.2:langID", "--out", str(out)])
    cfg = yaml.safe_load(out.read_text())
    assert [(e["type"], e["weight"], e["tags"]["prompt_mode"]) for e in cfg] == [
        ("nemo_tarred", 0.8, "auto"),
        ("nemo", 0.2, "langID"),
    ]
