import json

import pytest

from hinglish_asr.bolai import RowFilter, customer_key, load_stats, subset_of


def _row(**kw):
    row = {
        "provider": "deepgram",
        "text": "मेरा order cancel करो",
        "is_unintelligible": False,
        "source_collection": "conversation_history",
        "language_code": "hi",
        "channel": "right",
        "duration_s": 2.4,
        "confidence": 0.9,
    }
    row.update(kw)
    return row


@pytest.mark.parametrize(
    "change,reason",
    [
        ({}, None),
        ({"provider": "assemblyai"}, "other_vendor"),
        ({"is_unintelligible": True, "text": ""}, "unintelligible"),
        ({"text": "  "}, "unintelligible"),
        ({"source_collection": "Test"}, "test_collection"),
        ({"language_code": "ta"}, "language"),
        ({"language_code": None}, "language"),
        ({"duration_s": 0.05}, "too_short"),
        ({"duration_s": 24.9}, "too_long"),
    ],
)
def test_row_filter_reasons(change, reason):
    assert RowFilter().reason(_row(**change)) == reason


def test_row_filter_confidence_and_channel_are_opt_in():
    assert RowFilter().reason(_row(confidence=0.51, channel="left")) is None
    assert RowFilter(min_confidence=0.6).reason(_row(confidence=0.51)) == "low_confidence"
    assert RowFilter(channels=("right",)).reason(_row(channel="left")) == "channel"


def test_customer_key_follows_call_direction():
    base = {"to_phone_hash": "T", "from_phone_hash": "F", "call_key": "C"}
    assert customer_key({**base, "call_direction": "Outbound"}) == "phone:T"
    assert customer_key({**base, "call_direction": "Inbound"}) == "phone:F"
    assert customer_key({**base, "call_direction": None}) == "call:C"
    assert customer_key({"call_direction": "Outbound", "to_phone_hash": None, "chunk_id": "X"}) == "call:X"


def test_load_stats_applies_aliases_and_merges_hosts(tmp_path):
    shard = {"rows": 1, "seconds": 2.0, "vendors": {"deepgram_nova-3": {"rows": 1, "seconds": 2.0}}}
    (tmp_path / "a.json").write_text(json.dumps({"shards": {"data/deepgram_nova-3/x.parquet": shard}}))
    (tmp_path / "b.json").write_text(json.dumps({"shards": {"data/combined/y.parquet": shard}}))
    (tmp_path / "config-path-aliases.json").write_text(
        json.dumps({"path_aliases": {"data/deepgram_nova-3/x.parquet": "data/inhouse-asr-2/x.parquet"}})
    )
    assert sorted(load_stats(tmp_path)) == ["data/combined/y.parquet", "data/inhouse-asr-2/x.parquet"]


def test_subset_of():
    assert subset_of("data/combined/combined-1.parquet") == "combined"
    assert subset_of("stats/x.json") == ""
