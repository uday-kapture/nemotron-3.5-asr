import pytest

torch = pytest.importorskip("torch")

from hinglish_asr.blend import blend, main  # noqa: E402


def _sd(w, steps):
    return {"w": torch.tensor(w), "bn.num_batches_tracked": torch.tensor(steps)}


def test_blend_moves_each_weight_part_of_the_way_to_the_fine_tune():
    base, tuned = _sd([0.0, 2.0], 10), _sd([1.0, 4.0], 99)
    assert torch.equal(blend(base, tuned, 0.0)["w"], base["w"])
    assert torch.equal(blend(base, tuned, 1.0)["w"], tuned["w"])
    half = blend(base, tuned, 0.5)
    assert torch.allclose(half["w"], torch.tensor([0.5, 3.0]))
    assert half["bn.num_batches_tracked"].item() == 99  # integer buffers are not averaged


def test_blend_refuses_models_that_do_not_line_up():
    with pytest.raises(ValueError, match="different layers"):
        blend({"a": torch.zeros(1)}, {"b": torch.zeros(1)}, 0.5)
    with pytest.raises(ValueError, match="shape"):
        blend({"a": torch.zeros(2)}, {"a": torch.zeros(3)}, 0.5)


def test_alphas_outside_0_to_1_are_refused():
    with pytest.raises(SystemExit):
        main(["--tuned", "x.nemo", "--out-dir", "out", "--alphas", "1.5"])
