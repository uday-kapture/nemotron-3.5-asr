import pytest

from hinglish_asr.semantic import canonical_tokens, dev_fold, lenient_scores, same_word_across_scripts
from hinglish_asr.scoring import score_corpus


@pytest.mark.parametrize(
    "latin,dev",
    [
        ("mobile", "मोबाइल"), ("cancel", "कैंसल"), ("manager", "मैनेजर"), ("service", "सर्विस"),
        ("station", "स्टेशन"), ("booking", "बुकिंग"), ("account", "अकाउंट"), ("payment", "पेमेंट"),
        ("number", "नंबर"), ("center", "सेन्टर"), ("quality", "क्वालिटी"), ("order", "ऑर्डर"), ("ok", "ओके"),
    ],
)
def test_same_word_in_both_scripts(latin, dev):
    assert same_word_across_scripts(latin, dev)
    assert same_word_across_scripts(dev, latin)


@pytest.mark.parametrize("latin,dev", [("is", "से"), ("call", "कम"), ("order", "ऑफर"), ("payment", "पैसा")])
def test_different_words_do_not_match(latin, dev):
    assert not same_word_across_scripts(latin, dev)


def test_same_script_is_never_a_cross_script_match():
    assert not same_word_across_scripts("mobile", "mobile")


@pytest.mark.parametrize("a,b", [("है", "हैं"), ("गये", "गए"), ("कीजिये", "कीजिए"), ("नयी", "नई"), ("हाँ", "हां")])
def test_devanagari_variants_fold_together(a, b):
    assert dev_fold(a) == dev_fold(b)


def test_devanagari_vowel_quality_still_differs():
    assert dev_fold("काम") != dev_fold("कम")


@pytest.mark.parametrize(
    "words,value",
    [
        ("दो सौ पचास", "#250"), ("250", "#250"), ("एक लाख", "#100000"), ("1 लाख", "#100000"),
        ("साढ़े तीन सौ", "#350"), ("डेढ़ सौ", "#150"), ("सवा लाख", "#125000"), ("दो लाख पचास हज़ार", "#250000"),
        ("नौ आठ सात", "#987"), ("9 8 7", "#987"), ("twenty five", "#25"), ("पाँच", "#5"), ("पांच", "#5"),
    ],
)
def test_number_runs_get_one_canonical_value(words, value):
    [(canon, surface)] = canonical_tokens(words.split())
    assert canon == value and surface == words


def test_number_runs_stop_at_words():
    assert [c for c, _ in canonical_tokens("मुझे दो सौ रुपये दो".split())] == ["मुझे", "#200", "रुपये", "#2"]


def test_lenient_wer_forgives_meaning_preserving_differences():
    refs = ["मेरा order cancel करो दो सौ रुपये है", "मिडकैप ठीक है"]
    hyps = ["मेरा ऑर्डर कैंसल करो 200 रुपये हैं", "मिड कैप ठीक है"]
    out = lenient_scores(refs, hyps)
    assert out["lenient_wer"] == 0.0
    assert out["matches_by_rule"] == {"cross_script": 2, "number": 1, "devanagari_variant": 1, "spacing": 1}
    assert dict(out["cross_script_pairs"]) == {("order", "ऑर्डर"): 1, ("cancel", "कैंसल"): 1}
    assert score_corpus(refs, hyps)["wer"] > 0.5


def test_lenient_wer_still_counts_real_errors():
    out = lenient_scores(["पैसे नहीं आए", "order कब आएगा"], ["पैसे आए", "offer कब आएगा"])
    assert out["per_utterance"] == [(1, 3), (1, 3)]
    assert out["lenient_wer"] == pytest.approx(2 / 6)


def test_lenient_skips_empty_references_like_score_corpus():
    out = lenient_scores(["", "हाँ"], ["कुछ", "हां"])
    assert out["per_utterance"] == [None, (0, 1)]


def test_a_wrong_or_missing_number_costs_every_word():
    ref = "मेरा नंबर नौ आठ सात छह पांच चार तीन दो एक शून्य है"
    wrong = lenient_scores([ref], ["मेरा नंबर 1 2 3 4 5 6 7 8 9 0 है"])
    assert wrong["per_utterance"] == [(10, 13)]
    assert lenient_scores([ref], ["मेरा नंबर है"])["per_utterance"] == [(10, 13)]
    assert lenient_scores([ref], ["मेरा नंबर 9876543210 है"])["per_utterance"] == [(0, 13)]
    # दो सौ vs 300: two reference words wrong, as in plain WER
    assert lenient_scores(["दो सौ रुपये"], ["300 रुपये"])["per_utterance"] == [(2, 3)]


@pytest.mark.parametrize(
    "words,value",
    [("शून्य नौ आठ", "#098"), ("0987", "#0987"), ("1 50 000", "#150000"), ("98765 43210", "#9876543210"),
     ("1 लाख 50000", "#150000"), ("2 लाख 50 हज़ार", "#250000")],
)
def test_digit_strings_keep_leading_zeros_and_join_groups(words, value):
    [(canon, _)] = canonical_tokens(words.split())
    assert canon == value


def test_switching_between_digits_and_number_words_starts_a_new_number():
    assert [c for c, _ in canonical_tokens("1 000 दो".split())] == ["#1000", "#2"]
    # after a multiplier the number goes on
    assert [c for c, _ in canonical_tokens("दो लाख 50 हज़ार".split())] == ["#250000"]


@pytest.mark.parametrize(
    "ref,hyp,errors",
    [("नंबर शून्य नौ आठ सात है", "नंबर 0987 है", 0), ("₹1,000 दो", "एक हज़ार रुपये दो", 1),
     ("1,50,000 रुपये", "डेढ़ लाख रुपये", 0), ("10,000", "10000", 0)],
)
def test_formatted_numbers_match_their_spoken_form(ref, hyp, errors):
    assert lenient_scores([ref], [hyp])["per_utterance"][0][0] == errors


def test_lenient_never_counts_more_than_plain_wer():
    # पांच सौ दो is 502, or 500 + "give": reading it as one number would cost 4.
    assert lenient_scores(["पांच सौ दो"], ["500 दो"])["per_utterance"] == [(2, 3)]
