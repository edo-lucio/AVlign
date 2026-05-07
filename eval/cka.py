"""Linear centered kernel alignment between two representation matrices.

CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)  (after centering)
"""
import torch


def _center(X: torch.Tensor) -> torch.Tensor:
    return X - X.mean(0, keepdim=True)


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """X: (N, D_x), Y: (N, D_y). Returns scalar in [0, 1]."""
    X = _center(X.float())
    Y = _center(Y.float())
    num = (Y.t() @ X).pow(2).sum()
    den = (X.t() @ X).pow(2).sum().sqrt() * (Y.t() @ Y).pow(2).sum().sqrt()
    return float((num / den.clamp_min(1e-12)).item())
