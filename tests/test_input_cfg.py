import json

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
        "tarred_audio_filepaths": f"{d}/audio__OP_0..3_CL_.tar",
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


def _remote(d, n, missing=()):
    d.mkdir(parents=True)
    (d / "sharded_manifests").mkdir()
    for i in range(n):
        (d / "sharded_manifests" / f"manifest_{i}.json").write_text("")
    (d / "remote.json").write_text(json.dumps({
        "repo": "org/shards", "revision": "abc", "prefix": "train", "shards": n,
        "missing_count": len(missing), "missing": list(missing),
    }))
    return d


def test_remote_shards_stream_with_local_manifests(tmp_path):
    d = _remote(tmp_path / "train", 3)
    e = build_entry(str(d), 1.0, "auto")
    assert e["manifest_filepath"] == f"{d}/sharded_manifests/manifest__OP_0..2_CL_.json"
    assert e["tarred_audio_filepaths"] == (
        "pipe:curl -sfL --retry 10 --retry-delay 5 --retry-all-errors "
        "https://huggingface.co/datasets/org/shards/resolve/abc/train/audio__OP_0..2_CL_.tar"
    )


def test_remote_shards_must_all_be_uploaded(tmp_path):
    with pytest.raises(InputCfgError, match="not uploaded"):
        build_entry(str(_remote(tmp_path / "train", 3, missing=[2])), 1.0)


def test_header_path_must_survive_nemo_brace_rewriting(tmp_path):
    with pytest.raises(InputCfgError, match="contains"):
        build_entry(str(_remote(tmp_path / "train", 1)), 1.0, auth_header="/tmp/hdr(1)")


def test_cli_writes_a_private_header_file_for_remote_entries(tmp_path, monkeypatch):
    monkeypatch.setattr("huggingface_hub.get_token", lambda: "hf_secret")
    d = _remote(tmp_path / "train", 2)
    header = tmp_path / "hdr" / "auth"
    main(["--entry", f"{d}:1.0:auto", "--out", str(tmp_path / "cfg.yaml"), "--auth-header", str(header)])
    assert header.read_text() == "Authorization: Bearer hf_secret\n"
    assert header.stat().st_mode & 0o777 == 0o600
    cfg = (tmp_path / "cfg.yaml").read_text()
    assert f"-H @{header}" in cfg and "hf_secret" not in cfg


def test_nemo_reads_the_tarred_entry_from_the_tars(tmp_path):
    """The entry goes through NeMo's own reader: audio comes out of the tars, not the filesystem."""
    cutset = pytest.importorskip("nemo.collections.common.data.lhotse.cutset")
    import io
    import itertools
    import tarfile

    import numpy as np
    import soundfile as sf

    d = tmp_path / "train"
    (d / "sharded_manifests").mkdir(parents=True)
    for k in range(2):
        rows = []
        with tarfile.open(d / f"audio_{k}.tar", "w") as tar:
            for j in range(3):
                name = f"chunk-{k}-{j}.flac"
                buf = io.BytesIO()
                sf.write(buf, np.zeros(8000, dtype=np.float32), 16000, format="FLAC")
                info = tarfile.TarInfo(name)
                info.size = len(buf.getvalue())
                buf.seek(0)
                tar.addfile(info, buf)
                rows.append({"audio_filepath": name, "duration": 0.5, "text": "ok", "shard_id": k})
        (d / "sharded_manifests" / f"manifest_{k}.json").write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "input_cfg.yaml"
    main(["--entry", f"{d}:1.0:auto", "--out", str(out)])

    cuts, is_tarred = cutset.read_cutset_from_config({"input_cfg": str(out), "shuffle": False, "seed": 0})
    cuts = list(itertools.islice(cuts, 6))  # tarred cut sets cycle through the shards forever
    assert is_tarred
    assert sorted(c.id for c in cuts) == [f"chunk-{k}-{j}.flac" for k in range(2) for j in range(3)]
    assert cuts[0].load_audio().shape == (1, 8000)
