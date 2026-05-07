"""Conditional flow matching: linear-interpolation path, MSE velocity loss, ODE solver."""
import torch


def sample_path(z_a: torch.Tensor):
    """Sample (t, x_t, target_v) for conditional flow matching.

    Linear interpolation: x_t = t * z_a + (1 - t) * eps
    Target velocity:      v* = z_a - eps
    """
    B = z_a.shape[0]
    t = torch.rand(B, device=z_a.device)
    eps = torch.randn_like(z_a)
    t_b = t.view(-1, *([1] * (z_a.ndim - 1)))
    x_t = t_b * z_a + (1.0 - t_b) * eps
    target = z_a - eps
    return t, x_t, target


def cfm_loss(v_pred, target):
    return (v_pred - target).pow(2).mean()


@torch.no_grad()
def euler_solve(model, z_v, audio_seq_len: int, audio_latent_dim: int,
                steps: int = 100, device: str = "cuda"):
    """Solve x'(t) = v(x, t, z_v) from t=0 to t=1, starting from x(0) ~ N(0, I)."""
    B = z_v.shape[0]
    x = torch.randn(B, audio_seq_len, audio_latent_dim, device=device)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((B,), i * dt, device=device)
        v, _ = model(x, t, z_v)
        x = x + dt * v
    return x


@torch.no_grad()
def dopri5_solve(model, z_v, audio_seq_len: int, audio_latent_dim: int,
                 device: str = "cuda", rtol: float = 1e-4, atol: float = 1e-4):
    from torchdiffeq import odeint
    B = z_v.shape[0]
    x0 = torch.randn(B, audio_seq_len, audio_latent_dim, device=device)
    def rhs(t, x):
        t_b = torch.full((B,), float(t), device=device)
        v, _ = model(x, t_b, z_v)
        return v
    ts = torch.tensor([0.0, 1.0], device=device)
    sol = odeint(rhs, x0, ts, method="dopri5", rtol=rtol, atol=atol)
    return sol[-1]
