"""MLP_v / MLP_a: 2-layer MLPs projecting into a 512-dim FGW cost space."""
import torch.nn as nn


def make_mlp(d_in: int, d_hidden: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_in, d_hidden),
        nn.GELU(),
        nn.Linear(d_hidden, d_out),
    )
