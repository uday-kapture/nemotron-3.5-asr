import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf

from hinglish_asr.prepare import main, prepare
from hinglish_asr.sources import Record


def _wav(path, seconds, rate=16000, channels=1):
    n = int(seconds * rate)
    pcm = (np.sin(np.arange(n) / 7.0) * 3000).astype("<i2")
    sf.write(path, pcm if channels == 1 else np.stack([pcm] * channels, 1), rate, subtype="PCM_16", format="WAV")
    return str(path)


def _read(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]


def test_prepare_writes_16k_audio_and_nemo_manifest_rows(tmp_path):
    recs = [
        Record(id="a", audio=_wav(tmp_path / "a.wav", 2.0, rate=8000), text="मेरा नंबर ९८ है", group="g1"),
        Record(id="b", audio=_wav(tmp_path / "b.wav", 1.0), text="ok जी", group="g2"),
    ]
    report = prepare(recs, tmp_path / "out", single_split="test")
    rows = _read(tmp_path / "out" / "test_manifest.jsonl")
    assert [r["id"] for r in rows] == ["a", "b"]
    a = rows[0]
    assert a["text"] == "मेरा नंबर 98 है"  # training normalization applied
    assert (a["target_lang"], a["lang"]) == ("hi-IN", "hi-IN")
    assert a["duration"] == pytest.approx(2.0, abs=1e-3)
    assert a["orig_sr"] == 8000
    assert sf.info(a["audio_filepath"]).samplerate == 16000
    assert report["utterances"] == {"test": 2}
    assert report["hours_by_orig_sr"]["8000"] == pytest.approx(2.0 / 3600, rel=1e-3)


def test_utterances_of_one_group_never_straddle_splits(tmp_path):
    recs = [
        Record(id=f"u{i}", audio=_wav(tmp_path / f"u{i}.wav", 0.5), text="हाँ", group=f"spk{i % 7}")
        for i in range(42)
    ]
    prepare(recs, tmp_path / "out", val_frac=0.3, test_frac=0.3)
    split_of_group = {}
    for split in ("train", "val", "test"):
        path = tmp_path / "out" / f"{split}_manifest.jsonl"
        for row in _read(path) if path.exists() else []:
            assert split_of_group.setdefault(row["group"], split) == split


def test_bad_rows_are_dropped_with_reasons_and_leave_no_audio(tmp_path):
    recs = [
        Record(id="ok", audio=_wav(tmp_path / "ok.wav", 1.0), text="ठीक है", group="g"),
        Record(id="empty", audio=_wav(tmp_path / "e.wav", 1.0), text="​ ", group="g"),
        Record(id="short", audio=_wav(tmp_path / "s.wav", 0.05), text="हाँ", group="g"),
        Record(id="long", audio=_wav(tmp_path / "l.wav", 3.0), text="लंबा", group="g"),
        Record(id="stereo", audio=_wav(tmp_path / "st.wav", 1.0, channels=2), text="दो", group="g"),
    ]
    report = prepare(recs, tmp_path / "out", single_split="train", min_duration=0.1, max_duration=2.0)
    assert [r["id"] for r in _read(tmp_path / "out" / "train_manifest.jsonl")] == ["ok"]
    assert report["dropped"] == {"empty_text": 1, "too_short": 1, "too_long": 1, "bad_audio": 1}
    assert {p.stem for p in (tmp_path / "out" / "audio").iterdir()} == {"ok"}


def test_rerun_reuses_converted_audio_and_reproduces_manifest(tmp_path):
    recs = [Record(id="a", audio=_wav(tmp_path / "a.wav", 1.0, rate=8000), text="हाँ", group="g")]
    prepare(recs, tmp_path / "out", single_split="test")
    wav = tmp_path / "out" / "audio" / "a.wav"
    first_mtime = wav.stat().st_mtime_ns
    first = (tmp_path / "out" / "test_manifest.jsonl").read_text()
    prepare(recs, tmp_path / "out", single_split="test")
    assert wav.stat().st_mtime_ns == first_mtime
    assert (tmp_path / "out" / "test_manifest.jsonl").read_text() == first


def test_rerun_does_not_need_the_source_audio_once_converted(tmp_path):
    # A resumed HF run skips downloading audio for finished rows (audio=None);
    # the original sample rate must come from what the first run recorded.
    src = _wav(tmp_path / "a.wav", 1.0, rate=8000)
    prepare([Record(id="a", audio=src, text="हाँ", group="g")], tmp_path / "out", single_split="test")
    prepare([Record(id="a", audio=None, text="हाँ जी", group="g")], tmp_path / "out", single_split="test")
    [row] = _read(tmp_path / "out" / "test_manifest.jsonl")
    assert row["orig_sr"] == 8000
    assert row["text"] == "हाँ जी"  # text is re-normalized from this run's record


def test_parallel_workers_give_the_same_manifest_as_serial(tmp_path):
    recs = [
        Record(id=f"u{i}", audio=_wav(tmp_path / f"u{i}.wav", 0.3 + i / 100), text=f"वाक्य {i}", group=f"g{i}")
        for i in range(12)
    ]
    prepare(recs, tmp_path / "serial", val_frac=0.2, test_frac=0.2, workers=1)
    prepare(recs, tmp_path / "parallel", val_frac=0.2, test_frac=0.2, workers=2)
    for split in ("train", "val", "test"):
        s = tmp_path / "serial" / f"{split}_manifest.jsonl"
        p = tmp_path / "parallel" / f"{split}_manifest.jsonl"
        strip = lambda path: [{k: v for k, v in r.items() if k != "audio_filepath"} for r in _read(path)]
        assert s.exists() == p.exists()
        if s.exists():
            assert strip(s) == strip(p)


def test_cli_hush_eval_source_uses_manifest_rows_and_references(tmp_path):
    # A local parquet stands in for the HF dataset (same schema), via --parquet.
    audio = []
    for i in range(6):
        p = _wav(tmp_path / f"r{i}.wav", 0.5)
        audio.append({"bytes": open(p, "rb").read(), "path": f"r{i}.wav"})
    table = pa.table(
        {
            "audio": pa.array(audio, type=pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
            "text": [f"original {i}" for i in range(6)],
        }
    )
    pq.write_table(table, tmp_path / "data.parquet", row_group_size=2)
    manifest = tmp_path / "test.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "source": {"type": "huggingface", "dataset": "org/ds", "revision": "a" * 40, "split": "train"},
                "samples": [
                    {"id": "hindi_x", "row_index": 1, "reference": "audited one"},
                    {"id": "hindi_y", "row_index": 4, "reference": "audited four"},
                ],
            }
        ),
        encoding="utf-8",
    )
    main(
        [
            "--source", "hush-eval", "--input", str(manifest),
            "--parquet", str(tmp_path / "data.parquet"),
            "--out", str(tmp_path / "out"), "--single-split", "test",
        ]
    )
    rows = _read(tmp_path / "out" / "test_manifest.jsonl")
    assert [(r["id"], r["text"], r["row"]) for r in rows] == [("hindi_x", "audited one", 1), ("hindi_y", "audited four", 4)]


def test_cli_exclude_hush_eval_rows_from_a_training_source(tmp_path):
    audio = [{"bytes": open(_wav(tmp_path / f"r{i}.wav", 0.5), "rb").read(), "path": f"r{i}.wav"} for i in range(4)]
    table = pa.table(
        {
            "audio": pa.array(audio, type=pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
            "text": [f"t{i}" for i in range(4)],
        }
    )
    pq.write_table(table, tmp_path / "data.parquet")
    held_out = tmp_path / "test.json"
    held_out.write_text(json.dumps({"version": 1, "samples": [{"id": "h", "row_index": 2, "reference": "x"}]}))
    main(
        [
            "--source", "hf", "--parquet", str(tmp_path / "data.parquet"), "--id-prefix", "hindi",
            "--exclude-hush-eval", str(held_out),
            "--out", str(tmp_path / "out"), "--single-split", "train",
        ]
    )
    assert [r["row"] for r in _read(tmp_path / "out" / "train_manifest.jsonl")] == [0, 1, 3]
