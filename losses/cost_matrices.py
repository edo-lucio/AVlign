"""Cost matrices for FGW.

C_v, C_a are intra-batch (B, B) cosine-distance matrices in the projection
space. M is the cross-modal (B, B) cost from CLAP embeddings.
"""
import torch
import torch.nn.functional as F


def cosine_cost(X: torch.Tensor) -> torch.Tensor:
    """Pairwise (1 - cosine) within batch. X: (B, D) -> (B, B)."""
    X = F.normalize(X, dim=-1)
    return 1.0 - X @ X.t()


def cross_modal_cost(E_v: torch.Tensor, E_a: torch.Tensor) -> torch.Tensor:
    """M_ij = 1 - cos(e_v_i, e_a_j). (B, D), (B, D) -> (B, B)."""
    E_v = F.normalize(E_v, dim=-1)
    E_a = F.normalize(E_a, dim=-1)
    return 1.0 - E_v @ E_a.t()
