import io
import json
import tarfile
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf

from hinglish_asr.convert import convert, pick_val_small
from hinglish_asr.input_cfg import build_entry

CFG = {
    "dataset": "org/ds",
    "revision": "a" * 40,
    "provider": "deepgram",
    "languages": ["hi", "en"],
    "min_duration": 0.1,
    "max_duration": 20.0,
    "min_confidence": 0.0,
    "channels": None,
    "val_frac": 0.25,
    "test_frac": 0.25,
    "split_seed": "t",
    "val_small_hours": 0.0005,
}


def _flac(seconds, rate=16000, channels=1):
    n = int(seconds * rate)
    pcm = (np.sin(np.arange(n) / 5.0) * 2000).astype("<i2")
    buf = io.BytesIO()
    sf.write(buf, pcm if channels == 1 else np.stack([pcm] * channels, 1), rate, format="FLAC", subtype="PCM_16")
    return buf.getvalue()


def _parquet(path, rows):
    cols = {k: [r.get(k) for r in rows] for k in rows[0]}
    audio = pa.array(cols.pop("audio"), type=pa.struct([("bytes", pa.binary()), ("path", pa.string())]))
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"audio": audio, **cols}), path, row_group_size=7)


def _row(i, customer, **kw):
    seconds = 1.0 + (i % 40) / 10
    row = {
        "audio": {"bytes": _flac(seconds), "path": "x.flac"},
        "text": f"order {i} cancel कर दो",
        "provider": "deepgram",
        "language_code": "hi",
        "is_unintelligible": False,
        "confidence": 0.9,
        "duration_s": seconds,
        "source_collection": "conversation_history",
        "call_direction": "Outbound",
        "to_phone_hash": customer,
        "from_phone_hash": "company",
        "call_key": f"call-{customer}",
        "chunk_id": f"chunk_{i:04d}",
        "channel": "right",
        "client_id": "c1",
    }
    row.update(kw)
    return row


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    rows0 = [_row(i, f"cust{i % 12}") for i in range(48)]
    rows0 += [
        _row(100, "x", provider="assemblyai"),
        _row(101, "x", text="", is_unintelligible=True),
        _row(102, "x", source_collection="test"),
        _row(103, "x", language_code="ta"),
        _row(104, "x", audio={"bytes": _flac(1.0, rate=8000), "path": "x"}),
        _row(105, "x", audio={"bytes": b"not audio", "path": "x"}),
        _row(106, "x", text="​"),
    ]
    _parquet(root / "data/combined/f0.parquet", rows0)
    _parquet(root / "data/combined/f1.parquet", [_row(200 + i, f"cust{i % 12}") for i in range(20)])
    plan = {
        "config": CFG,
        "config_sha": "abc",
        "files": [{"index": 0, "path": "data/combined/f0.parquet"}, {"index": 1, "path": "data/combined/f1.parquet"}],
    }
    return root, plan


def _jsonl(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_convert_writes_matching_tar_shards_and_heldout_manifests(tmp_path, repo):
    root, plan = repo
    out = tmp_path / "out"
    report = convert(plan, out, workers=1, local_root=str(root))

    train_rows = []
    for k in (0, 1):
        manifest = _jsonl(out / "train/sharded_manifests" / f"manifest_{k}.json")
        with tarfile.open(out / "train" / f"audio_{k}.tar") as tar:
            members = {m.name: tar.extractfile(m).read() for m in tar.getmembers()}
        # NeMo's tarred reader looks every tar member up by audio_filepath.
        assert sorted(members) == sorted(r["audio_filepath"] for r in manifest)
        for r in manifest:
            assert r["shard_id"] == k
            assert sf.info(io.BytesIO(members[r["audio_filepath"]])).duration == pytest.approx(r["duration"], abs=1e-3)
        train_rows += manifest

    val, test = _jsonl(out / "val_manifest.jsonl"), _jsonl(out / "test_manifest.jsonl")
    for r in val + test:
        assert sf.info(r["audio_filepath"]).samplerate == 16000
    groups = {s: {r["group"] for r in rows} for s, rows in (("train", train_rows), ("val", val), ("test", test))}
    assert not (groups["train"] & groups["val"] or groups["train"] & groups["test"] or groups["val"] & groups["test"])
    assert len(train_rows) + len(val) + len(test) == 68
    assert all(r["target_lang"] == "hi-IN" and r["group"].startswith("phone:") for r in train_rows + val + test)

    assert report["dropped"] == {
        "other_vendor": 1, "unintelligible": 1, "test_collection": 1, "language": 1, "bad_audio": 2, "empty_text": 1,
    }
    assert report["files"] == {"planned": 2, "done": 2, "missing": 0, "missing_indices": []}
    assert report["splits"]["train"]["latin_word_share"] == pytest.approx(2 / 5)  # order, cancel of: order N cancel कर दो
    # Shards are numbered 0..N-1, so input_cfg accepts the directory as is.
    assert build_entry(str(out / "train"), 1.0, "auto")["tarred_audio_filepath"].endswith("audio__OP_0..1_CL_.tar")


def test_rerun_skips_done_files_and_keeps_output(tmp_path, repo):
    root, plan = repo
    out = tmp_path / "out"
    convert(plan, out, workers=1, local_root=str(root), limit_files=1)
    assert not (out / "train/audio_1.tar").exists()
    first = (out / "train/audio_0.tar").stat().st_mtime_ns
    report = convert(plan, out, workers=1, local_root=str(root))
    assert (out / "train/audio_0.tar").stat().st_mtime_ns == first
    assert report["files"]["done"] == 2


def test_failed_file_is_reported_and_retried(tmp_path, repo):
    root, plan = repo
    plan["files"].append({"index": 2, "path": "data/combined/missing.parquet"})
    report = convert(plan, tmp_path / "out", workers=1, local_root=str(root))
    assert [e["index"] for e in report["errors"]] == [2]
    assert report["files"]["missing_indices"] == [2]
    assert not (tmp_path / "out/done/2.json").exists()


def test_parallel_matches_serial(tmp_path, repo):
    root, plan = repo
    convert(plan, tmp_path / "s", workers=1, local_root=str(root))
    convert(plan, tmp_path / "p", workers=2, local_root=str(root))
    for name in ("train/sharded_manifests/manifest_0.json", "val_manifest.jsonl", "test_manifest.jsonl"):
        strip = lambda rows: [{k: v for k, v in r.items() if k != "audio_filepath"} for r in rows]
        assert strip(_jsonl(tmp_path / "s" / name)) == strip(_jsonl(tmp_path / "p" / name))


def test_val_small_takes_whole_customers_up_to_the_budget():
    rows = [{"id": f"u{i}", "group": f"g{i % 5}", "duration": 360.0} for i in range(20)]
    small = pick_val_small(rows, hours=0.3, seed="s")
    groups = {r["group"] for r in small}
    assert all(sum(r["group"] == g for r in small) == 4 for g in groups)  # whole customers only
    assert len(groups) == 1  # 4 x 6 min = 0.4 h >= 0.3 h after one customer
    assert small == pick_val_small(list(reversed(rows)), hours=0.3, seed="s")


class FakeHub:
    def __init__(self, private=True):
        self.private = private
        self.commits = []

    def create_repo(self, repo, repo_type, private, exist_ok):
        assert private and repo_type == "dataset"

    def repo_info(self, repo, repo_type):
        return SimpleNamespace(private=self.private, sha=f"sha{len(self.commits)}")

    def create_commit(self, repo, operations, repo_type, commit_message):
        self.commits.append(sorted(op.path_in_repo for op in operations))
        return SimpleNamespace(oid=f"c{len(self.commits)}")


def test_upload_pushes_shards_in_batches_and_frees_local_disk(tmp_path, repo):
    root, plan = repo
    out, hub = tmp_path / "out", FakeHub()
    report = convert(plan, out, workers=1, local_root=str(root), upload_to="org/shards", upload_batch=1, api=hub)
    assert hub.commits == [
        ["train/audio_0.tar", "train/sharded_manifests/manifest_0.json"],
        ["train/audio_1.tar", "train/sharded_manifests/manifest_1.json"],
    ]
    assert not list((out / "train").glob("*.tar"))
    assert (out / "train/sharded_manifests/manifest_1.json").exists()  # NeMo reads manifests locally
    remote = json.loads((out / "train/remote.json").read_text())
    assert remote == report["shard_repo"]
    assert remote["revision"] == "sha2" and remote["shards"] == 2 and remote["missing_count"] == 0
    entry = build_entry(str(out / "train"), 1.0, "auto", auth_header="/home/u/.hf-header")
    assert entry["tarred_audio_filepath"].endswith(
        "-H @/home/u/.hf-header https://huggingface.co/datasets/org/shards/resolve/sha2/train/audio__OP_0..1_CL_.tar"
    )
    assert entry["tarred_audio_filepath"].startswith("pipe:curl -sfL --retry 10")


def test_upload_resumes_tars_left_by_a_local_run(tmp_path, repo):
    root, plan = repo
    out, hub = tmp_path / "out", FakeHub()
    convert(plan, out, workers=1, local_root=str(root))
    convert(plan, out, workers=1, local_root=str(root), upload_to="org/shards", api=hub)
    assert len(hub.commits) == 1 and len(hub.commits[0]) == 4
    assert not list((out / "train").glob("*.tar"))
    convert(plan, out, workers=1, local_root=str(root), upload_to="org/shards", api=hub)
    assert len(hub.commits) == 1  # nothing left to upload


def test_upload_refuses_public_repos_and_the_source_dataset(tmp_path, repo):
    root, plan = repo
    with pytest.raises(ValueError, match="public"):
        convert(plan, tmp_path / "a", workers=1, local_root=str(root), upload_to="org/shards", api=FakeHub(private=False))
    with pytest.raises(ValueError, match="refusing"):
        convert(plan, tmp_path / "b", workers=1, local_root=str(root), upload_to=CFG["dataset"], api=FakeHub())
