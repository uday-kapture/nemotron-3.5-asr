import json

import pytest

from hinglish_asr.scoring import load_equivalents, normalize_for_scoring, score_corpus


def test_scoring_normalizer_drops_punctuation_and_case_but_keeps_matras():
    # `[^\w\s]`-style stripping would shred मुझे into म झ; category-based must not.
    assert normalize_for_scoring("मुझे Meeting के लिए, LATE हो गया।") == "मुझे meeting के लिए late हो गया"


def test_language_tags_are_removed_before_punctuation_stripping():
    # Without this, "<hi-IN>" would survive as the two words "hi in".
    assert normalize_for_scoring("हाँ जी। <hi-IN>") == "हाँ जी"


def test_equivalents_map_canonicalizes_tokens():
    eq = {"मैडम": "madam", "mam": "madam", "madam": "madam"}
    assert normalize_for_scoring("Mam मैडम सुनिए", eq) == "madam madam सुनिए"


def test_load_equivalents_skips_groups_pending_review(tmp_path):
    path = tmp_path / "equivalents.json"
    path.write_text(
        json.dumps(
            {
                "groups": [
                    {"canonical": "madam", "variants": ["मैडम", "Mam"], "source": "reviewed"},
                    {"canonical": "hello", "variants": ["हेलो"], "source": "auto — REVIEW ME"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    eq = load_equivalents(path)
    assert eq == {"madam": "madam", "मैडम": "madam", "mam": "madam"}


def test_substitutions_are_attributed_to_the_reference_words_script():
    # meeting->मीटिंग and late->लेट: 2 errors, both on Latin reference words.
    s = score_corpus(["मुझे meeting के लिए late हो गया।"], ["मुझे मीटिंग के लिए लेट हो गया"])
    assert s["ref_words"] == 7
    assert s["substitutions"] == 2
    assert s["wer"] == pytest.approx(2 / 7)
    assert s["latin_ref_words"] == 2
    assert s["wer_latin"] == pytest.approx(1.0)
    assert s["devanagari_ref_words"] == 5
    assert s["wer_devanagari"] == pytest.approx(0.0)


def test_insertions_count_against_the_inserted_words_script_and_deletions_against_the_reference():
    s = score_corpus(["ठीक है", "हाँ जी"], ["ठीक है sir", "हाँ"])
    assert (s["insertions"], s["deletions"]) == (1, 1)
    assert s["wer"] == pytest.approx(2 / 4)
    assert s["latin_errors"] == 1  # "sir" inserted
    assert s["devanagari_errors"] == 1  # "जी" deleted


def test_equivalents_remove_cross_script_substitutions_from_the_score():
    eq = {"मीटिंग": "meeting", "meeting": "meeting"}
    s = score_corpus(["मुझे meeting के लिए late हो गया।"], ["मुझे मीटिंग के लिए लेट हो गया"], eq)
    assert s["wer"] == pytest.approx(1 / 7)


def test_pairs_with_empty_normalized_reference_are_skipped_and_counted():
    s = score_corpus(["।", "ठीक है"], ["कुछ", "ठीक है"])
    assert s["skipped_empty_refs"] == 1
    assert s["utterances"] == 1
    assert s["wer"] == pytest.approx(0.0)
