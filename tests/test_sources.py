import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hinglish_asr.sources import ParquetSource, SourceError, read_hush_eval_manifest, read_jsonl_source


def _parquet(path, rows, row_group_size=3):
    """Same schema as kapturecx/Hindi-Transcription-DS: audio{bytes,path}, text."""
    table = pa.table(
        {
            "audio": pa.array(
                [{"bytes": f"wav{i}".encode(), "path": p} for i, (p, _) in enumerate(rows)],
                type=pa.struct([("bytes", pa.binary()), ("path", pa.string())]),
            ),
            "text": [t for _, t in rows],
        }
    )
    pq.write_table(table, path, row_group_size=row_group_size)
    return str(path)


ROWS = [(f"IISc_VaaniProject_M_Bihar_Araria_Spk{i % 3}_x_GENERIC.wav", f"वाक्य {i}") for i in range(8)]


def test_parquet_records_carry_bytes_text_and_global_row_ids(tmp_path):
    src = ParquetSource([_parquet(tmp_path / "a.parquet", ROWS)])
    recs = list(src.records(id_prefix="hindi"))
    assert [r.row for r in recs] == list(range(8))
    assert recs[5].id == "hindi_5"
    assert recs[5].audio == b"wav5"
    assert recs[5].text == "वाक्य 5"


def test_parquet_row_group_bounds_are_global_across_files(tmp_path):
    a = _parquet(tmp_path / "a.parquet", ROWS[:5])  # groups [0,3) [3,5)
    b = _parquet(tmp_path / "b.parquet", ROWS[5:])  # groups [5,8)
    src = ParquetSource([a, b])
    assert src.row_group_bounds() == [(0, 3), (3, 5), (5, 8)]
    assert [r.row for r in src.records()] == list(range(8))


def test_parquet_selected_rows_only(tmp_path):
    src = ParquetSource([_parquet(tmp_path / "a.parquet", ROWS)])
    assert [r.row for r in src.records(rows=[6, 1, 2])] == [1, 2, 6]


def test_parquet_skips_audio_for_row_groups_already_converted(tmp_path):
    # Groups [0,3) [3,6) [6,8). Rows 0-2 are all done; row 3 is done but 4-5 are not.
    src = ParquetSource([_parquet(tmp_path / "a.parquet", ROWS)])
    done = {"r_0", "r_1", "r_2", "r_3"}
    recs = {r.row: r for r in src.records(id_prefix="r", converted=done.__contains__)}
    assert [recs[i].audio for i in (0, 1, 2)] == [None, None, None]
    assert recs[1].text == "वाक्य 1"  # text still comes from the dataset
    assert recs[3].audio == b"wav3"  # its group is read anyway for rows 4-5
    assert recs[4].audio == b"wav4"


def test_parquet_excluded_rows_are_skipped(tmp_path):
    src = ParquetSource([_parquet(tmp_path / "a.parquet", ROWS)])
    assert [r.row for r in src.records(exclude={0, 4, 7})] == [1, 2, 3, 5, 6]


def test_parquet_references_override_text_and_id(tmp_path):
    src = ParquetSource([_parquet(tmp_path / "a.parquet", ROWS)])
    refs = {3: {"id": "hindi_abc", "reference": "audited text"}}
    [rec] = src.records(rows=[3], references=refs)
    assert (rec.id, rec.text) == ("hindi_abc", "audited text")


def test_parquet_group_comes_from_pattern_on_audio_path(tmp_path):
    src = ParquetSource([_parquet(tmp_path / "a.parquet", ROWS)])
    recs = list(src.records(group_pattern=r"^IISc_VaaniProject_[A-Z]_[^_]+_[^_]+_([^_]+)_"))
    assert [r.group for r in recs[:4]] == ["Spk0", "Spk1", "Spk2", "Spk0"]


def test_parquet_without_pattern_groups_by_row(tmp_path):
    src = ParquetSource([_parquet(tmp_path / "a.parquet", ROWS)])
    assert next(iter(src.records(id_prefix="x"))).group == "x_0"


def test_jsonl_source_resolves_relative_audio_and_defaults(tmp_path):
    (tmp_path / "calls").mkdir()
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(
        json.dumps({"audio_filepath": "calls/c1.wav", "text": "हाँ जी", "offset": 1.5, "duration": 2.0, "group": "agent7"})
        + "\n"
        + json.dumps({"audio_filepath": "/abs/c2.wav", "text": "ok"})
        + "\n",
        encoding="utf-8",
    )
    a, b = read_jsonl_source(manifest)
    assert a.audio == str(tmp_path / "calls" / "c1.wav")
    assert (a.offset, a.duration, a.group) == (1.5, 2.0, "agent7")
    assert b.audio == "/abs/c2.wav"
    assert b.offset == 0.0 and b.duration is None
    assert a.id != b.id


def test_jsonl_ids_differ_for_segments_of_the_same_file(tmp_path):
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps({"audio_filepath": "/a/call.wav", "text": "x", "offset": o, "duration": 1.0}) for o in (0.0, 5.0)
        ),
        encoding="utf-8",
    )
    a, b = read_jsonl_source(manifest)
    assert a.id != b.id
    assert a.group == b.group  # segments of one call stay together by default


def test_jsonl_row_without_text_is_an_error_naming_the_line(tmp_path):
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(json.dumps({"audio_filepath": "/a.wav"}) + "\n", encoding="utf-8")
    with pytest.raises(SourceError, match="line 1"):
        list(read_jsonl_source(manifest))


def test_hush_eval_manifest_is_read(tmp_path):
    path = tmp_path / "test.json"
    doc = {
        "version": 1,
        "name": "hindi_test",
        "source": {"type": "huggingface", "dataset": "org/ds", "revision": "a" * 40, "split": "train"},
        "samples": [{"id": "hindi_1", "row_index": 75, "reference": "और स्कूल"}],
    }
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    source, samples = read_hush_eval_manifest(path)
    assert source["dataset"] == "org/ds"
    assert samples[0]["row_index"] == 75


def test_hush_eval_manifest_with_unknown_version_is_rejected(tmp_path):
    path = tmp_path / "test.json"
    path.write_text(json.dumps({"version": 2, "samples": []}), encoding="utf-8")
    with pytest.raises(SourceError, match="version"):
        read_hush_eval_manifest(path)
