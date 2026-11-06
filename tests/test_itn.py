import pytest

from hinglish_asr.itn import numbers_to_digits


@pytest.mark.parametrize(
    "text,expected",
    [
        ("दो सौ पचास रुपये", "250 रुपये"),
        ("मेरा नंबर नौ आठ सात छह पांच चार तीन दो एक शून्य है।", "मेरा नंबर 9876543210 है।"),
        ("अट्ठानवे सत्तासी छिहत्तर पचपन चौवालीस", "9887765544"),  # spoken in pairs
        ("OTP है शून्य नौ आठ सात", "OTP है 0987"),
        ("दो लाख 50 हज़ार", "250000"),
        ("सवा लाख", "125000"),
        ("साढ़े तीन सौ", "350"),
        ("पौने चार सौ", "375"),
        ("5 हज़ार", "5000"),
        ("दो हज़ार चौबीस में", "2024 में"),
        ("पच्चीस तारीख को", "25 तारीख को"),
        ("सौ रुपये", "100 रुपये"),
        ("twenty five rupees", "25 rupees"),
        ("One hundred percent", "100 percent"),
        ("पांच सौ दो रुपये", "502 रुपये"),
        ("(दो सौ)", "(200)"),
        ("हाँ जी, दो सौ, पचास", "हाँ जी, 200, 50"),
        ("pin number है पचासी तैंतालीस सत्ताईस", "pin number है 854327"),  # spoken in pairs
        ("एक सौ हज़ार", "100000"),
        ("twenty five thousand", "25000"),
        ("one hundred twenty", "120"),
        ("नौ आठ जीरो सात", "9807"),
    ],
)
def test_number_words_become_digits(text, expected):
    assert numbers_to_digits(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "एक बार", "दो दिन", "कर दो", "पैसे दो",  # lone numbers below 10: दो is also "give", एक "a"
        "एक दो दिन", "दस बारह दिन", "दो तीन सौ रुपये", "शून्य नौ आठ",  # ranges, too short for digit by digit
        "डेढ़ घंटा", "साढ़े तीन बजे", "पौने",  # fractions and bare modifiers
        "1 2 3 4", "500 दो", "₹1,000 दो",  # already digits
        # several whole numbers: place values only fall within one number
        "दो हज़ार तीन हज़ार नहीं", "चार सौ बीस पांच सौ", "सौ सौ रुपये", "बीस पांच",
    ],
)
def test_unsure_numbers_stay_as_written(text):
    assert numbers_to_digits(text) == text


def test_do_after_a_multiplier_at_sentence_end_is_the_verb():
    assert numbers_to_digits("मुझे पांच सौ दो") == "मुझे 500 दो"
    assert numbers_to_digits("मुझे पांच सौ दो। ठीक है") == "मुझे 500 दो। ठीक है"
