from hinglish_asr.verify_empty import split_verified, summary


def test_only_clips_the_model_also_leaves_empty_are_kept_in_their_split():
    rows = [
        {"id": "a", "for_split": "train", "duration": 3600.0, "target_lang": "hi-IN", "text": ""},
        {"id": "b", "for_split": "train", "duration": 1.0, "target_lang": "hi-IN", "text": ""},
        {"id": "c", "for_split": "val", "duration": 1.0, "target_lang": "en-US", "text": ""},
    ]
    kept, rejected = split_verified(rows, ["<hi-IN> ", "हाँ जी", " . "])  # tags and punctuation are not words
    assert [r["id"] for r in kept["train"]] == ["a"] and [r["id"] for r in kept["val"]] == ["c"]
    assert "for_split" not in kept["train"][0]
    assert [(r["id"], h) for r, h in rejected] == [("b", "हाँ जी")]
    s = summary(rows, kept, rejected)
    assert s["splits"]["train"] == {"candidates": 2, "verified": 1, "verified_share": 0.5, "verified_hours": 1.0}
    assert s["rejected_top_words"] == [["हाँ", 1], ["जी", 1]] or s["rejected_top_words"] == [("हाँ", 1), ("जी", 1)]
