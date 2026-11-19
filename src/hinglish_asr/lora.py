"""LoRA: train a small low-rank change to chosen linear layers, everything else frozen.

Each chosen layer computes W x + b + scale * B A x, with W and b frozen and only
A (rank x in) and B (out x rank, zero at the start) trained. The model can then
move only a little from what the base model knew, so it forgets less (and
learns less) than updating all weights.

`merge` folds B A into W, giving the base model's own layers and keys: the
checkpoints pushed to the hub, benchmark's winner.nemo and the served model are
plain models, streaming included. Lightning's -last checkpoint keeps the LoRA
form, so a run still resumes.
"""

from __future__ import annotations

import math
import re

import torch
from torch import nn

# Conformer attention and feed-forward layers, and the joint's two input
# projections. Not the joint's output layer: SampledRNNTJoint reads its weight
# directly, which would skip the LoRA.
DEFAULT_LAYERS = (
    r"encoder\.layers\.\d+\.(self_attn\.linear_(q|k|v|out)|feed_forward[12]\.linear[12])"
    r"|joint\.(enc|pred)"
)


class LoRALinear(nn.Linear):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        nn.Module.__init__(self)
        self.in_features, self.out_features = base.in_features, base.out_features
        self.register_parameter("weight", base.weight)  # the same tensors, under the same keys
        self.register_parameter("bias", base.bias)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, dtype=base.weight.dtype, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=base.weight.dtype, device=base.weight.device))
        # A buffer, so `merge` needs nothing but the state dict.
        self.register_buffer("lora_scale", torch.tensor(alpha / rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.lora_scale


def add(model: nn.Module, layers: str = DEFAULT_LAYERS, rank: int = 32, alpha: float = 64.0) -> list[str]:
    """Freeze every weight, then give each nn.Linear whose name fully matches `layers` a trainable LoRA."""
    names = [n for n, m in model.named_modules() if type(m) is nn.Linear and re.fullmatch(layers, n)]
    if not names:
        raise ValueError(f"no linear layer matches {layers!r}")
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.momentum = 0.0  # running stats are weights too: call audio would shift them in train mode
    for name in names:
        parent, _, child = name.rpartition(".")
        owner = model.get_submodule(parent) if parent else model
        setattr(owner, child, LoRALinear(getattr(owner, child), rank, alpha))
    return names


def merge(state_dict: dict) -> dict:
    """The plain state dict: each layer's LoRA folded into its weight. Unchanged when there is none."""
    prefixes = [k[: -len(".lora_A")] for k in state_dict if k.endswith(".lora_A")]
    if not prefixes:
        return state_dict
    out = {k: v for k, v in state_dict.items() if not k.rpartition(".")[2].startswith("lora_")}
    for p in prefixes:
        a, b, scale = state_dict[f"{p}.lora_A"], state_dict[f"{p}.lora_B"], state_dict[f"{p}.lora_scale"]
        w = out[f"{p}.weight"]
        out[f"{p}.weight"] = (w.float() + scale.float() * (b.float() @ a.float())).to(w.dtype)
    return out
