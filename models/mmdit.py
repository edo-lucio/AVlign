"""Small MMDiT velocity network.

Audio is a sequence of T tokens of dim audio_latent_dim. Visual condition is the
CLIP pooled feature, projected to model_dim and used as a single token. Joint
attention concatenates audio + condition along the sequence axis.

Returns predicted velocity (B, T, audio_latent_dim) and the post-final-block
audio representation h_a (B, model_dim) — the mean-pooled audio token stream
used as input to the projection heads for FGW / InfoNCE.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class TimestepEmbedder(nn.Module):
    def __init__(self, d_model: int, n_freqs: int = 256):
        super().__init__()
        self.n_freqs = n_freqs
        self.mlp = nn.Sequential(
            nn.Linear(n_freqs, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) in [0, 1]
        half = self.n_freqs // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None] * 1000.0
        emb = torch.cat([args.cos(), args.sin()], dim=-1)
        return self.mlp(emb)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class AdaLNZero(nn.Module):
    """Per-modality AdaLN-Zero modulation: produces (shift, scale, gate) x 2."""
    def __init__(self, d_cond: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_cond, 6 * d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, c):
        return self.proj(F.silu(c)).chunk(6, dim=-1)


class JointBlock(nn.Module):
    """One MMDiT joint block: separate QKV/MLP per modality, joint attention."""
    def __init__(self, d_model: int, n_heads: int, d_ffn: int, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        # AdaLN per modality
        self.ada_a = AdaLNZero(d_model, d_model)
        self.ada_v = AdaLNZero(d_model, d_model)
        # Norms (pre-mod)
        self.n1_a = nn.LayerNorm(d_model, elementwise_affine=False)
        self.n1_v = nn.LayerNorm(d_model, elementwise_affine=False)
        self.n2_a = nn.LayerNorm(d_model, elementwise_affine=False)
        self.n2_v = nn.LayerNorm(d_model, elementwise_affine=False)
        # QKV per modality
        self.qkv_a = nn.Linear(d_model, 3 * d_model)
        self.qkv_v = nn.Linear(d_model, 3 * d_model)
        # Output projection per modality
        self.proj_a = nn.Linear(d_model, d_model)
        self.proj_v = nn.Linear(d_model, d_model)
        # FFN per modality
        self.ffn_a = nn.Sequential(
            nn.Linear(d_model, d_ffn), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
        )
        self.ffn_v = nn.Sequential(
            nn.Linear(d_model, d_ffn), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
        )

    def forward(self, x_a, x_v, c):
        sa_a, sc_a, gate1_a, sa2_a, sc2_a, gate2_a = self.ada_a(c)
        sa_v, sc_v, gate1_v, sa2_v, sc2_v, gate2_v = self.ada_v(c)

        # --- attention ---
        h_a = modulate(self.n1_a(x_a), sa_a, sc_a)
        h_v = modulate(self.n1_v(x_v), sa_v, sc_v)
        qkv_a = self.qkv_a(h_a)
        qkv_v = self.qkv_v(h_v)
        # concat along seq axis
        qkv = torch.cat([qkv_a, qkv_v], dim=1)             # (B, T_a+T_v, 3D)
        q, k, v = qkv.chunk(3, dim=-1)
        q = rearrange(q, "b t (h d) -> b h t d", h=self.n_heads)
        k = rearrange(k, "b t (h d) -> b h t d", h=self.n_heads)
        v = rearrange(v, "b t (h d) -> b h t d", h=self.n_heads)
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "b h t d -> b t (h d)")
        T_a = x_a.shape[1]
        out_a, out_v = out[:, :T_a], out[:, T_a:]
        x_a = x_a + gate1_a.unsqueeze(1) * self.proj_a(out_a)
        x_v = x_v + gate1_v.unsqueeze(1) * self.proj_v(out_v)

        # --- FFN ---
        x_a = x_a + gate2_a.unsqueeze(1) * self.ffn_a(modulate(self.n2_a(x_a), sa2_a, sc2_a))
        x_v = x_v + gate2_v.unsqueeze(1) * self.ffn_v(modulate(self.n2_v(x_v), sa2_v, sc2_v))
        return x_a, x_v


class MMDiT(nn.Module):
    def __init__(self, audio_latent_dim: int, audio_seq_len: int, clip_dim: int,
                 d_model: int = 512, n_blocks: int = 6, n_heads: int = 8,
                 d_ffn: int = 2048, dropout: float = 0.0, n_cond_tokens: int = 1):
        super().__init__()
        self.audio_latent_dim = audio_latent_dim
        self.audio_seq_len = audio_seq_len
        self.d_model = d_model
        self.n_cond_tokens = n_cond_tokens

        self.audio_in = nn.Linear(audio_latent_dim, d_model)
        self.cond_in = nn.Linear(clip_dim, d_model * n_cond_tokens)
        self.t_embed = TimestepEmbedder(d_model)

        self.audio_pos = nn.Parameter(torch.randn(1, audio_seq_len, d_model) * 0.02)
        self.cond_pos = nn.Parameter(torch.randn(1, n_cond_tokens, d_model) * 0.02)

        self.blocks = nn.ModuleList([
            JointBlock(d_model, n_heads, d_ffn, dropout) for _ in range(n_blocks)
        ])
        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.final_ada = AdaLNZero(d_model, d_model)
        self.audio_out = nn.Linear(d_model, audio_latent_dim)
        nn.init.zeros_(self.audio_out.weight)
        nn.init.zeros_(self.audio_out.bias)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, z_v: torch.Tensor):
        """
        x_t: (B, T, audio_latent_dim) noisy audio latent at time t
        t:   (B,) timestep in [0, 1]
        z_v: (B, clip_dim) CLIP pooled feature
        Returns:
          v_pred: (B, T, audio_latent_dim) predicted velocity
          h_a:    (B, d_model) post-final-block mean-pooled audio repr
        """
        B = x_t.shape[0]
        x_a = self.audio_in(x_t) + self.audio_pos
        x_v = self.cond_in(z_v).view(B, self.n_cond_tokens, self.d_model) + self.cond_pos
        c = self.t_embed(t)

        for block in self.blocks:
            x_a, x_v = block(x_a, x_v, c)

        h_a = x_a.mean(dim=1)                              # (B, d_model)

        # Final modulation + projection (audio only).
        sa, sc, _, _, _, _ = self.final_ada(c)
        out = modulate(self.final_norm(x_a), sa, sc)
        v_pred = self.audio_out(out)
        return v_pred, h_a
