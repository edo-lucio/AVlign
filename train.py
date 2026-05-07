"""Train CFM / CFM+InfoNCE / CFM+FGW.

Usage:
    python train.py --config configs/base.yaml --mode fgw
    python train.py --config configs/base.yaml --mode cfm --synthetic   # smoke test
"""
import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from data.dataset import VGGSoundDataset, SyntheticDataset, collate
from data.sampler import ClassStratifiedSampler
from models.mmdit import MMDiT
from models.cfm import sample_path, cfm_loss
from models.projection_heads import make_mlp
from losses.cost_matrices import cosine_cost, cross_modal_cost
from losses.fgw import fgw_loss, fgw_plan, plan_entropy, uniform_plan_entropy
from losses.infonce import infonce_loss


@dataclass
class TrainState:
    step: int = 0


def build_components(cfg, device):
    m = cfg["model"]
    d = cfg["data"]
    net = MMDiT(
        audio_latent_dim=d["audio_latent_dim"],
        audio_seq_len=d["audio_seq_len"],
        clip_dim=d["clip_dim"],
        d_model=m["d_model"],
        n_blocks=m["n_blocks"],
        n_heads=m["n_heads"],
        d_ffn=m["d_ffn"],
        dropout=m["dropout"],
    ).to(device)
    mlp_v = make_mlp(d["clip_dim"], m["proj_hidden"], m["proj_out"]).to(device)
    mlp_a = make_mlp(m["d_model"], m["proj_hidden"], m["proj_out"]).to(device)
    return net, mlp_v, mlp_a


def build_loader(cfg, synthetic: bool):
    d = cfg["data"]
    b = cfg["batch"]
    if synthetic:
        ds = SyntheticDataset(
            n=512, n_classes=16,
            audio_seq_len=d["audio_seq_len"],
            audio_latent_dim=d["audio_latent_dim"],
            clip_dim=d["clip_dim"], clap_dim=d["clap_dim"],
        )
        class_text_clap = torch.randn(16, d["clap_dim"])
    else:
        ds = VGGSoundDataset(
            d["manifest"], split="train",
            audio_seq_len=d["audio_seq_len"],
            audio_latent_dim=d["audio_latent_dim"],
        )
        class_text_clap = torch.load(d["class_text_clap"], weights_only=False).float()
    sampler = ClassStratifiedSampler(
        ds, classes_per_batch=b["classes_per_batch"], per_class=b["per_class"],
    )
    loader = DataLoader(
        ds, batch_sampler=sampler, num_workers=d["num_workers"],
        collate_fn=collate, pin_memory=(not synthetic),
    )
    return loader, class_text_clap


def fgw_lambda(step: int, fcfg) -> float:
    """Spec schedule: off until warmup_end, ramp from lambda_ramp_start to target."""
    if step < fcfg["fgw_warmup_end"]:
        return 0.0
    if step >= fcfg["fgw_ramp_end"]:
        return fcfg["lambda_target"]
    span = fcfg["fgw_ramp_end"] - fcfg["fgw_warmup_end"]
    frac = (step - fcfg["fgw_warmup_end"]) / span
    return fcfg["lambda_ramp_start"] + frac * (fcfg["lambda_target"] - fcfg["lambda_ramp_start"])


def lr_lambda(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--mode", choices=["cfm", "infonce", "fgw"], default=None)
    ap.add_argument("--synthetic", action="store_true",
                    help="use SyntheticDataset (smoke test, no VGGSound required)")
    ap.add_argument("--steps", type=int, default=None,
                    help="override total_steps (useful for smoke tests)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.mode:
        cfg["loss"]["mode"] = args.mode
    mode = cfg["loss"]["mode"]
    if args.steps:
        cfg["train"]["total_steps"] = args.steps
        cfg["train"]["log_every"] = max(1, min(cfg["train"]["log_every"], args.steps // 2))
        # Compress FGW phases proportionally for smoke runs.
        cfg["loss"]["fgw"]["fgw_warmup_end"] = max(1, args.steps // 8)
        cfg["loss"]["fgw"]["fgw_ramp_end"] = max(2, args.steps // 4)
    if args.synthetic:
        # Override full-VGGSound shapes with smoke-sized ones.
        cfg["data"]["audio_seq_len"] = 32
        cfg["data"]["audio_latent_dim"] = 32
        cfg["data"]["clip_dim"] = 64
        cfg["data"]["clap_dim"] = 32
        cfg["data"]["num_workers"] = 0
        cfg["model"]["d_model"] = 64
        cfg["model"]["n_blocks"] = 2
        cfg["model"]["n_heads"] = 4
        cfg["model"]["d_ffn"] = 128
        cfg["model"]["proj_hidden"] = 64
        cfg["model"]["proj_out"] = 32
        cfg["batch"]["classes_per_batch"] = 4
        cfg["batch"]["per_class"] = 4
        cfg["amp"] = False

    torch.manual_seed(cfg["seed"])
    device = cfg["device"] if torch.cuda.is_available() else "cpu"
    print(f"device={device} mode={mode} synthetic={args.synthetic}")

    net, mlp_v, mlp_a = build_components(cfg, device)
    loader, class_text_clap = build_loader(cfg, args.synthetic)
    class_text_clap = class_text_clap.to(device)

    params = list(net.parameters())
    if mode in ("infonce", "fgw"):
        params += list(mlp_v.parameters()) + list(mlp_a.parameters())
    opt = torch.optim.AdamW(params, lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_lambda(s, cfg["train"]["warmup_steps"], cfg["train"]["total_steps"]))
    scaler = torch.amp.GradScaler(device=device, enabled=cfg["amp"] and device == "cuda")

    ckpt_dir = Path(cfg["train"]["ckpt_dir"]) / cfg["run_name"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "log.jsonl"
    log_f = open(log_path, "a")

    state = TrainState()
    fcfg = cfg["loss"]["fgw"]
    total_steps = cfg["train"]["total_steps"]
    log_every = cfg["train"]["log_every"]
    ckpt_every = cfg["train"]["ckpt_every"]

    t0 = time.time()
    data_iter = iter(loader)
    while state.step < total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        z_v = batch["z_v"].to(device, non_blocking=True)
        z_a = batch["z_a"].to(device, non_blocking=True)
        clap_a = batch["clap_audio_emb"].to(device, non_blocking=True)
        # CLAP-text e_v per sample, looked up by class_id (shared semantic space with e_a).
        class_id = batch["class_id"].to(device)
        e_v_clap = class_text_clap[class_id]

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda" if device == "cuda" else "cpu",
                                enabled=cfg["amp"] and device == "cuda"):
            t, x_t, target_v = sample_path(z_a)
            v_pred, h_a = net(x_t, t, z_v)
            l_cfm = cfm_loss(v_pred, target_v)

            extras = {}
            l_total = l_cfm
            if mode == "infonce":
                p_v = mlp_v(z_v)
                p_a = mlp_a(h_a)
                l_nce = infonce_loss(p_v, p_a, tau=cfg["loss"]["infonce"]["tau"])
                l_total = l_total + l_nce
                extras["l_infonce"] = l_nce.item()
            elif mode == "fgw":
                lam = fgw_lambda(state.step, fcfg)
                if lam > 0.0:
                    p_v = mlp_v(z_v)
                    p_a = mlp_a(h_a)
                    C_v = cosine_cost(p_v)
                    C_a = cosine_cost(p_a)
                    M = cross_modal_cost(e_v_clap, clap_a)
                    # FGW under fp32 for solver stability.
                    l_fgw = fgw_loss(
                        C_v.float(), C_a.float(), M.float(),
                        alpha=fcfg["alpha"], epsilon=fcfg["sinkhorn_eps"],
                        max_iter=fcfg["sinkhorn_iter"],
                    )
                    l_total = l_total + lam * l_fgw
                    extras["l_fgw"] = l_fgw.item()
                    extras["lambda_fgw"] = lam
                else:
                    extras["lambda_fgw"] = 0.0

        scaler.scale(l_total).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, cfg["train"]["grad_clip"])
        scaler.step(opt)
        scaler.update()
        sched.step()
        state.step += 1

        # Log
        if state.step % log_every == 0:
            entry = {
                "step": state.step,
                "l_total": float(l_total.item()),
                "l_cfm": float(l_cfm.item()),
                "lr": opt.param_groups[0]["lr"],
                "dt": time.time() - t0,
                **extras,
            }
            print(json.dumps(entry))
            log_f.write(json.dumps(entry) + "\n")
            log_f.flush()
            t0 = time.time()

        # H(T) diagnostic for FGW
        if mode == "fgw" and state.step % fcfg["log_plan_every"] == 0 \
           and fgw_lambda(state.step, fcfg) > 0.0:
            with torch.no_grad():
                p_v = mlp_v(z_v).float()
                p_a = mlp_a(h_a).float()
                T_plan = fgw_plan(
                    cosine_cost(p_v), cosine_cost(p_a),
                    cross_modal_cost(e_v_clap, clap_a).float(),
                    alpha=fcfg["alpha"], epsilon=fcfg["sinkhorn_eps"],
                    max_iter=fcfg["sinkhorn_iter"],
                )
                B = T_plan.shape[0]
                entry = {
                    "step": state.step, "tag": "plan",
                    "H_T": plan_entropy(T_plan),
                    "H_uniform": uniform_plan_entropy(B),
                }
                print(json.dumps(entry))
                log_f.write(json.dumps(entry) + "\n")
                log_f.flush()

        # Checkpoint
        if state.step % ckpt_every == 0 or state.step == total_steps:
            ckpt = {
                "step": state.step,
                "net": net.state_dict(),
                "mlp_v": mlp_v.state_dict(),
                "mlp_a": mlp_a.state_dict(),
                "config": cfg,
            }
            torch.save(ckpt, ckpt_dir / f"ckpt_{state.step}.pt")
            torch.save(ckpt, ckpt_dir / "ckpt_latest.pt")

    log_f.close()
    print(f"done. ckpts in {ckpt_dir}")


if __name__ == "__main__":
    main()
