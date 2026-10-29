import pytest

from hinglish_asr.baseline import latency_ms, output_stats, parse_att_context, summarize


def test_parse_att_context():
    assert parse_att_context("56,13") == [56, 13]


@pytest.mark.parametrize("bad", ["56", "a,b", "56,1,2"])
def test_parse_att_context_rejects_malformed(bad):
    with pytest.raises(ValueError):
        parse_att_context(bad)


@pytest.mark.parametrize("ctx,ms", [([56, 0], 80), ([56, 3], 320), ([56, 6], 560), ([56, 13], 1120)])
def test_latency_matches_model_card(ctx, ms):
    # 8x subsampling of 10 ms frames: each lookahead step is 80 ms on top of the current chunk.
    assert latency_ms(ctx) == ms


def test_output_stats_counts_tags_empties_and_script_share():
    s = output_stats(["हाँ जी। <hi-IN>", "ok <en-US>", "नहीं", ""])
    assert s["tag_rate"] == pytest.approx(0.5)
    assert s["tags"] == {"<hi-IN>": 1, "<en-US>": 1}
    assert s["empty_rate"] == pytest.approx(0.25)
    assert s["latin_word_share"] == pytest.approx(1 / 4)  # हाँ जी ok नहीं


def test_summarize_combines_scores_stats_and_speed():
    row = summarize(
        {"target_lang": "hi-IN", "att_context": [56, 0], "mode": "offline"},
        refs=["मुझे meeting के लिए late हो गया।"],
        hyps=["मुझे मीटिंग के लिए late हो गया। <hi-IN>"],
        equivalents=None,
        audio_seconds=100.0,
        wall_seconds=10.0,
    )
    assert row["latency_ms"] == 80
    assert row["wer"] == pytest.approx(1 / 7)
    assert row["wer_latin"] == pytest.approx(1 / 2)
    assert row["tag_rate"] == pytest.approx(1.0)
    assert row["rtfx"] == pytest.approx(10.0)
    assert row["ref_latin_word_share"] == pytest.approx(2 / 7)
