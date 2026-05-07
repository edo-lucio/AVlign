"""KL divergence on AudioSet classifier logits.

Given (N, C) logits for real and generated audio (must come from the same
classifier), compute average per-clip KL between class probability
distributions. Use a pretrained PANNs CNN14 or AST checkpoint upstream.
"""
import numpy as np
import torch
import torch.nn.functional as F


def kl_divergence(real_logits: torch.Tensor, gen_logits: torch.Tensor) -> float:
    p = F.softmax(real_logits, dim=-1)
    q = F.softmax(gen_logits, dim=-1)
    eps = 1e-12
    kl = (p * (p.add(eps).log() - q.add(eps).log())).sum(-1)
    return float(kl.mean().item())
