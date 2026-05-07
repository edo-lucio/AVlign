"""Smoke test: run a few training steps in each mode on synthetic data.

Verifies that the model, losses, and training loop are wired correctly without
needing VGGSound or any pretrained encoders.

    python -m pytest tests/test_smoke.py -s
or simply:
    python tests/test_smoke.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from data.dataset import SyntheticDataset, collate
from data.sampler import ClassStratifiedSampler
from torch.utils.data import DataLoader

from models.mmdit import MMDiT
from models.cfm import sample_path, cfm_loss, euler_solve
from models.projection_heads import make_mlp
from losses.cost_matrices import cosine_cost, cross_modal_cost
from losses.fgw import fgw_loss, fgw_plan, plan_entropy
from losses.infonce import infonce_loss


def make_components(audio_seq_len=32, audio_latent_dim=32, clip_dim=64,
                    d_model=64, device="cpu"):
    net = MMDiT(audio_latent_dim=audio_latent_dim, audio_seq_len=audio_seq_len,
                clip_dim=clip_dim, d_model=d_model, n_blocks=2, n_heads=4,
                d_ffn=128).to(device)
    mlp_v = make_mlp(clip_dim, 64, 32).to(device)
    mlp_a = make_mlp(d_model, 64, 32).to(device)
    return net, mlp_v, mlp_a


def make_loader(batch_size=16, classes_per_batch=4, per_class=4, **shape_kwargs):
    ds = SyntheticDataset(
        n=128, n_classes=8,
        audio_seq_len=shape_kwargs["audio_seq_len"],
        audio_latent_dim=shape_kwargs["audio_latent_dim"],
        clip_dim=shape_kwargs["clip_dim"],
        clap_dim=32,
    )
    sampler = ClassStratifiedSampler(ds, classes_per_batch=classes_per_batch,
                                     per_class=per_class)
    return DataLoader(ds, batch_sampler=sampler, num_workers=0, collate_fn=collate)


def step_once(mode: str):
    device = "cpu"
    shape = dict(audio_seq_len=32, audio_latent_dim=32, clip_dim=64)
    net, mlp_v, mlp_a = make_components(d_model=64, device=device, **shape)
    loader = make_loader(**shape)
    batch = next(iter(loader))

    z_v = batch["z_v"].to(device)
    z_a = batch["z_a"].to(device)
    clap_a = batch["clap_audio_emb"].to(device)
    # clap_dim=32 mismatch with z_v=64 — use the same shape for cross-modal cost test
    # by projecting z_v through mlp_v down to 32.
    e_v = mlp_v(z_v).detach()
    e_a = clap_a

    t, x_t, target_v = sample_path(z_a)
    v_pred, h_a = net(x_t, t, z_v)
    l_cfm = cfm_loss(v_pred, target_v)
    assert torch.isfinite(l_cfm)

    if mode == "infonce":
        loss = l_cfm + infonce_loss(mlp_v(z_v), mlp_a(h_a), tau=0.07)
    elif mode == "fgw":
        C_v = cosine_cost(mlp_v(z_v))
        C_a = cosine_cost(mlp_a(h_a))
        M = cross_modal_cost(e_v, e_a)
        l_fgw = fgw_loss(C_v.float(), C_a.float(), M.float(),
                         alpha=0.5, epsilon=0.05, max_iter=20)
        loss = l_cfm + 0.01 * l_fgw
    else:
        loss = l_cfm

    loss.backward()
    print(f"[{mode}] loss={float(loss):.4f}  (cfm={float(l_cfm):.4f})")

    if mode == "fgw":
        T = fgw_plan(C_v.detach().float(), C_a.detach().float(), M.float(),
                     alpha=0.5, epsilon=0.05, max_iter=20)
        H = plan_entropy(T)
        print(f"[{mode}] H(T)={H:.4f}  B={T.shape[0]}")


def test_inference_shape():
    shape = dict(audio_seq_len=32, audio_latent_dim=32, clip_dim=64)
    net, _, _ = make_components(d_model=64, device="cpu", **shape)
    z_v = torch.randn(2, 64)
    out = euler_solve(net, z_v, 32, 32, steps=5, device="cpu")
    assert out.shape == (2, 32, 32)
    print("inference shape ok")


if __name__ == "__main__":
    for mode in ["cfm", "infonce", "fgw"]:
        step_once(mode)
    test_inference_shape()
    print("all smoke checks passed")
