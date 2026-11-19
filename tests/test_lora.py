import copy

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from hinglish_asr import lora  # noqa: E402


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.linear_q = nn.Linear(8, 8)
        self.self_attn.linear_pos = nn.Linear(8, 8, bias=False)  # not chosen
        self.feed_forward1 = nn.Module()
        self.feed_forward1.linear1 = nn.Linear(8, 16)
        self.feed_forward1.linear2 = nn.Linear(16, 8)
        self.norm = nn.LayerNorm(8)

    def forward(self, x):
        x = self.norm(x + self.self_attn.linear_q(x) + self.self_attn.linear_pos(x))
        return x + self.feed_forward1.linear2(torch.relu(self.feed_forward1.linear1(x)))


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList([_Layer(), _Layer()])
        self.joint = nn.Module()
        self.joint.enc = nn.Linear(8, 4)
        self.joint.joint_net = nn.Sequential(nn.ReLU(), nn.Linear(4, 5))  # output layer: not chosen

    def forward(self, x):
        for layer in self.encoder.layers:
            x = layer(x)
        return self.joint.joint_net(self.joint.enc(x))


def test_add_wraps_the_chosen_layers_and_trains_only_the_lora():
    torch.manual_seed(0)
    model, x = _Model(), torch.randn(3, 8)
    before = model(x)
    names = lora.add(model, rank=2, alpha=4)
    assert names == [
        "encoder.layers.0.self_attn.linear_q", "encoder.layers.0.feed_forward1.linear1",
        "encoder.layers.0.feed_forward1.linear2", "encoder.layers.1.self_attn.linear_q",
        "encoder.layers.1.feed_forward1.linear1", "encoder.layers.1.feed_forward1.linear2", "joint.enc",
    ]
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == {f"{n}.lora_{m}" for n in names for m in "AB"}
    assert torch.allclose(model(x), before)  # B starts at zero: the same model


def test_merged_weights_load_into_the_plain_model_and_give_the_same_output():
    torch.manual_seed(0)
    plain = _Model()
    model, x = copy.deepcopy(plain), torch.randn(3, 8)
    lora.add(model, rank=2, alpha=4)
    for n, p in model.named_parameters():
        if n.endswith("lora_B"):
            p.data.normal_()  # as if trained
    merged = lora.merge(model.state_dict())
    assert merged.keys() == plain.state_dict().keys()
    plain.load_state_dict(merged)  # strict: the base model's own keys
    assert torch.allclose(plain(x), model(x), atol=1e-5)


def test_batchnorm_running_stats_stay_frozen_in_train_mode():
    model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))
    lora.add(model, layers=r"0", rank=2, alpha=2)
    before = model[1].running_mean.clone()
    model.train()(torch.randn(16, 4) + 5)
    assert torch.equal(model[1].running_mean, before)


def test_merge_leaves_a_plain_state_dict_alone_and_add_needs_a_match():
    sd = _Model().state_dict()
    assert lora.merge(sd) is sd
    with pytest.raises(ValueError, match="no linear layer"):
        lora.add(_Model(), layers=r"decoder\..*")
