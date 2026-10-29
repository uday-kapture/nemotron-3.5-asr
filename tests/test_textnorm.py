from hinglish_asr.textnorm import normalize_target


def test_devanagari_digits_become_ascii():
    # The base tokenizer has no pieces for ०-९; left alone they train as <unk>.
    assert normalize_target("मेरा नंबर ९८७६ है") == "मेरा नंबर 9876 है"


def test_zero_width_characters_are_removed():
    # ZWNJ/ZWJ are absent from the vocab; ZWSP and BOM sneak in from copy-paste.
    assert normalize_target("क्‍ष ‌है​﻿") == "क्ष है"


def test_precomposed_nukta_letter_is_decomposed():
    # U+095B (ज़) is a composition exclusion: NFC turns it into ज + ़, which is
    # the form the tokenizer emits. Mixed forms would count as WER errors.
    assert normalize_target("ज़रूर") == "ज़रूर"


def test_double_danda_becomes_single_danda():
    assert normalize_target("राम राम॥") == "राम राम।"


def test_whitespace_is_collapsed_including_nbsp():
    assert normalize_target("  मुझे  meeting\tके  लिए \n") == "मुझे meeting के लिए"


def test_case_and_punctuation_are_kept_for_training_targets():
    # The base model is trained to emit punctuation and capitalization; the
    # scoring normalizer strips them, the training normalizer must not.
    assert normalize_target("मुझे Meeting के लिए late हो गया।") == "मुझे Meeting के लिए late हो गया।"


def test_brace_gloss_replaces_the_devanagari_spelling_before_it():
    # Vaani transcripts write "रेस्टोरेंट {restaurant}": the speaker said one
    # English word; the convention here is English words in Latin script.
    assert normalize_target("यह रेस्टोरेंट {restaurant} है, स्टूल {stool} भी") == "यह restaurant है, stool भी"


def test_empty_brace_gloss_is_dropped_and_the_word_kept():
    assert normalize_target("नीला स्टूल {} है") == "नीला स्टूल है"


def test_brace_gloss_at_the_start_is_unwrapped():
    assert normalize_target("{bulb} जल रहा है") == "bulb जल रहा है"


def test_text_that_is_only_invisible_characters_normalizes_to_empty():
    assert normalize_target("​ ‍ ") == ""
