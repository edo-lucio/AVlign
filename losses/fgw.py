"""Fused Gromov-Wasserstein loss via POT (entropic, differentiable).

Returns a scalar loss with autograd through M, C_v, C_a. A separate helper
recomputes the transport plan T (no grad) for H(T) logging.
"""
import math

import torch
import ot


def fgw_loss(C_v: torch.Tensor, C_a: torch.Tensor, M: torch.Tensor,
             alpha: float = 0.5, epsilon: float = 0.05,
             max_iter: int = 50) -> torch.Tensor:
    """Differentiable entropic FGW. All inputs share batch dim B; all (B, B)."""
    B = C_v.shape[0]
    p = torch.full((B,), 1.0 / B, device=C_v.device, dtype=C_v.dtype)
    q = torch.full((B,), 1.0 / B, device=C_v.device, dtype=C_v.dtype)
    # POT supports gradients through (M, C1, C2) with this entry point.
    loss = ot.gromov.entropic_fused_gromov_wasserstein2(
        M, C_v, C_a, p, q,
        loss_fun="square_loss", alpha=alpha,
        epsilon=epsilon, max_iter=max_iter,
        solver="PGD", log=False,
    )
    return loss


@torch.no_grad()
def fgw_plan(C_v: torch.Tensor, C_a: torch.Tensor, M: torch.Tensor,
             alpha: float = 0.5, epsilon: float = 0.05,
             max_iter: int = 50) -> torch.Tensor:
    B = C_v.shape[0]
    p = torch.full((B,), 1.0 / B, device=C_v.device, dtype=C_v.dtype)
    q = torch.full((B,), 1.0 / B, device=C_v.device, dtype=C_v.dtype)
    T = ot.gromov.entropic_fused_gromov_wasserstein(
        M, C_v, C_a, p, q,
        loss_fun="square_loss", alpha=alpha,
        epsilon=epsilon, max_iter=max_iter,
        solver="PGD", log=False,
    )
    return T


def plan_entropy(T: torch.Tensor) -> float:
    """H(T) = -sum T * log T (treating T as a discrete distribution)."""
    eps = 1e-30
    H = -(T * (T + eps).log()).sum().item()
    return H


def uniform_plan_entropy(B: int) -> float:
    """log(B^2) — entropy of the uniform B x B plan with mass 1."""
    # T_uniform = 1/B^2 over B^2 cells; H = log(B^2)
    return 2 * math.log(B)
