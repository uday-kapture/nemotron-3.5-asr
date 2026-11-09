import pytest

from hinglish_asr.textnorm import spoken_target
from hinglish_asr.verbalize import cardinal, verbalize_numbers


@pytest.mark.parametrize(
    "n,hi,en",
    [
        (0, "शून्य", "zero"),
        (25, "पच्चीस", "twenty five"),
        (100, "एक सौ", "one hundred"),
        (2978, "दो हज़ार नौ सौ अठहत्तर", "two thousand nine hundred seventy eight"),
        (150000, "एक लाख पचास हज़ार", "one lakh fifty thousand"),
        (12000000, "एक करोड़ बीस लाख", "one crore twenty lakh"),
    ],
)
def test_cardinals_use_the_indian_system_in_both_languages(n, hi, en):
    assert " ".join(cardinal(n, "hi")) == hi
    assert " ".join(cardinal(n, "en")) == en


@pytest.mark.parametrize(
    "text,lang,expected",
    [
        ("मैडम ₹2,000 दे दिए", "hi", "मैडम दो हज़ार रुपये दे दिए"),
        ("04:00 बजे करेंगे", "hi", "चार बजे करेंगे"),
        ("4:30 बजे", "hi", "साढ़े चार बजे"),
        ("3:45", "hi", "पौने चार"),
        ("call at 4:05", "en", "call at four oh five"),
        ("50% discount", "hi", "पचास प्रतिशत discount"),
        ("1,50,000 रुपये", "hi", "एक लाख पचास हज़ार रुपये"),
        ("2.5 लाख", "hi", "ढाई लाख"),
        ("2.5 lakh", "en", "two point five lakh"),
        ("नंबर 9876543210 है", "hi", "नंबर नौ आठ सात छह पांच चार तीन दो एक शून्य है"),
        ("OTP 0798", "en", "OTP zero seven nine eight"),
        ("50000 रुपये", "hi", "पचास हज़ार रुपये"),  # a round amount, not an id
        ("2024 में", "hi", "दो हज़ार चौबीस में"),
        ("3rd floor", "en", "third floor"),
        ("4G network", "hi", "4G network"),  # glued to letters: left as written
    ],
)
def test_digits_become_spoken_words_in_the_rows_language(text, lang, expected):
    assert verbalize_numbers(text, lang) == expected


def test_spoken_target_is_lowercase_without_punctuation():
    assert spoken_target("Hello जो, वह जो ₹500 बंद हो गया।", "hi") == "hello जो वह जो पांच सौ रुपये बंद हो गया"
    assert spoken_target("No, uh, we're fine. Follow-up?", "en") == "no uh we're fine follow up"
    assert spoken_target("'OK' sir", "en") == "ok sir"


def test_ordinals_are_english_words_in_hindi_rows_too():
    assert verbalize_numbers("हां 12th होता है", "hi") == "हां twelfth होता है"
    assert verbalize_numbers("1st time", "hi") == "first time"
