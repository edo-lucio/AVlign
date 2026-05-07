"""Symmetric InfoNCE on (z_v, h_a) pairs with in-batch negatives."""
import torch
import torch.nn.functional as F


def infonce_loss(z_v: torch.Tensor, h_a: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """z_v, h_a: (B, D) — must be projected into a common space upstream."""
    z_v = F.normalize(z_v, dim=-1)
    h_a = F.normalize(h_a, dim=-1)
    logits = (z_v @ h_a.t()) / tau                     # (B, B)
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
