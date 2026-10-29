import pytest

from hinglish_asr.split import assign_split, group_key, sample_rows_by_row_group


def test_split_fractions_are_respected_over_many_groups():
    splits = [assign_split(f"speaker{i}", val_frac=0.05, test_frac=0.10) for i in range(20000)]
    assert splits.count("test") / len(splits) == pytest.approx(0.10, abs=0.01)
    assert splits.count("val") / len(splits) == pytest.approx(0.05, abs=0.01)


def test_zero_fractions_put_everything_in_train():
    assert {assign_split(f"g{i}", 0.0, 0.0) for i in range(1000)} == {"train"}


def test_split_is_stable_across_calls_for_the_same_group():
    first = [assign_split(f"g{i}", 0.1, 0.1) for i in range(200)]
    assert first == [assign_split(f"g{i}", 0.1, 0.1) for i in range(200)]


def test_different_seed_reshuffles_groups():
    a = [assign_split(f"g{i}", 0.5, 0.0, seed="a") for i in range(200)]
    b = [assign_split(f"g{i}", 0.5, 0.0, seed="b") for i in range(200)]
    assert a != b


VAANI = r"^IISc_VaaniProject_[A-Z]_[^_]+_[^_]+_([^_]+)_"


def test_group_key_extracts_speaker_from_path():
    path = "IISc_VaaniProject_M_Bihar_Araria_Juga65357_0445450000_GENERIC_0975_446_2464.wav"
    assert group_key(path, VAANI, fallback="row_1") == "Juga65357"


def test_group_key_falls_back_when_pattern_does_not_match():
    assert group_key("sarvam_v2v_contextual_hi-IN_000180.flac", VAANI, fallback="row_9") == "row_9"


def test_group_key_without_pattern_uses_fallback():
    assert group_key("anything.wav", None, fallback="row_3") == "row_3"


def test_row_group_sampling_takes_whole_groups_until_enough_rows():
    # Row groups: [0,10) [10,20) [20,30) [30,40). Wanted rows sit in all four.
    bounds = [(0, 10), (10, 20), (20, 30), (30, 40)]
    wanted = [1, 2, 11, 12, 13, 25, 31, 32]
    picked = sample_rows_by_row_group(wanted, bounds, n=3, seed=0)
    # Exactly n rows, all from the fewest groups possible, sorted for streaming.
    assert len(picked) == 3
    assert picked == sorted(picked)
    groups_touched = {next(i for i, (s, e) in enumerate(bounds) if s <= r < e) for r in picked}
    assert len(groups_touched) <= 2


def test_row_group_sampling_returns_all_rows_when_n_exceeds_available():
    bounds = [(0, 10), (10, 20)]
    assert sample_rows_by_row_group([3, 15], bounds, n=100, seed=0) == [3, 15]
