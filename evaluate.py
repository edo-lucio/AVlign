"""Driver for the full eval suite.

Generates audio for the test split with a trained checkpoint, then runs all
five metrics. Audio quality metrics (FAD, KL) require pretrained classifiers
that are NOT bundled here — load points are clearly marked.

Outputs runs/<run_name>/eval.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from data.dataset import VGGSoundDataset, collate
from torch.utils.data import DataLoader
from infer import load_ckpt
from models.cfm import euler_solve
from models.projection_heads import make_mlp

from eval.fad import fad
from eval.kl_divergence import kl_divergence
from eval.av_align import av_align
from eval.retrieval import recall_at_k
from eval.cka import linear_cka


def embed_audio_classifier(wavs: torch.Tensor, sample_rate: int):
    """REPLACE ME: load a pretrained PANNs/VGGish, return (N, D) embeddings.

    Returning identity over a small projection so the driver runs end-to-end
    on a smoke test without external weights.
    """
    raise NotImplementedError(
        "Plug a pretrained audio classifier here for FAD/KL. "
        "See https://github.com/qiuqiangkong/audioset_tagging_cnn (PANNs).")


def embed_audio_av(wavs: torch.Tensor, sample_rate: int):
    """REPLACE ME: load CAVP-audio. Returns (N, D)."""
    raise NotImplementedError("Load CAVP audio encoder here.")


def embed_video_av(images):
    """REPLACE ME: load CAVP-video. Returns (N, D)."""
    raise NotImplementedError("Load CAVP video encoder here.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max_clips", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip_audio_metrics", action="store_true",
                    help="skip FAD/KL/AV-align (only run CKA + retrieval on cached embeddings)")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    net, cfg = load_ckpt(args.ckpt, device)
    d = cfg["data"]; m = cfg["model"]
    manifest = args.manifest or d["manifest"]

    ds = VGGSoundDataset(manifest, split="test",
                         audio_seq_len=d["audio_seq_len"],
                         audio_latent_dim=d["audio_latent_dim"])
    n = min(args.max_clips, len(ds))
    loader = DataLoader(torch.utils.data.Subset(ds, list(range(n))),
                        batch_size=8, num_workers=2, collate_fn=collate)

    # Also load projection heads + post-block h_a for CKA / retrieval.
    mlp_v = make_mlp(d["clip_dim"], m["proj_hidden"], m["proj_out"]).to(device).eval()
    mlp_a = make_mlp(m["d_model"], m["proj_hidden"], m["proj_out"]).to(device).eval()
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    if "mlp_v" in ckpt:
        mlp_v.load_state_dict(ckpt["mlp_v"])
        mlp_a.load_state_dict(ckpt["mlp_a"])

    H_v_all, H_a_all, P_v_all, P_a_all = [], [], [], []
    for batch in tqdm(loader, desc="forward"):
        z_v = batch["z_v"].to(device)
        z_a = batch["z_a"].to(device)
        with torch.inference_mode():
            t = torch.full((z_v.shape[0],), 0.5, device=device)
            x_t = 0.5 * z_a + 0.5 * torch.randn_like(z_a)
            _, h_a = net(x_t, t, z_v)
        H_v_all.append(z_v.cpu()); H_a_all.append(h_a.cpu())
        P_v_all.append(mlp_v(z_v).cpu()); P_a_all.append(mlp_a(h_a).cpu())

    H_v = torch.cat(H_v_all); H_a = torch.cat(H_a_all)
    P_v = torch.cat(P_v_all); P_a = torch.cat(P_a_all)

    cka = linear_cka(H_v, H_a)
    r1 = recall_at_k(P_a, P_v, k=1)

    results = {"CKA": cka, "R@1": r1}

    if not args.skip_audio_metrics:
        # Generate audio for the same clips, run FAD/KL/AV-align.
        # Decoding is expensive — only do this for a subset (e.g. 200 clips).
        n_gen = min(200, n)
        print(f"generating audio for {n_gen} clips (this is slow)")
        sub_loader = DataLoader(torch.utils.data.Subset(ds, list(range(n_gen))),
                                batch_size=2, num_workers=0, collate_fn=collate)
        gen_wavs, real_wavs = [], []
        import dac
        dac_model = dac.DAC.load(dac.utils.download(model_type="44khz")).to(device).eval()
        sr = dac_model.sample_rate
        for batch in tqdm(sub_loader, desc="generate"):
            z_v = batch["z_v"].to(device)
            z_a = batch["z_a"].to(device)
            with torch.inference_mode():
                z_gen = euler_solve(net, z_v, d["audio_seq_len"], d["audio_latent_dim"],
                                    steps=cfg["infer"]["ode_steps"], device=device)
                w_gen = dac_model.decode(z_gen.transpose(1, 2)).cpu()  # (B,1,N)
                w_real = dac_model.decode(z_a.transpose(1, 2)).cpu()
            gen_wavs.append(w_gen); real_wavs.append(w_real)
        gen_wavs = torch.cat(gen_wavs); real_wavs = torch.cat(real_wavs)

        try:
            r_emb = embed_audio_classifier(real_wavs, sr).numpy()
            g_emb = embed_audio_classifier(gen_wavs, sr).numpy()
            results["FAD"] = fad(r_emb, g_emb)
            r_logits = embed_audio_classifier(real_wavs, sr)
            g_logits = embed_audio_classifier(gen_wavs, sr)
            results["KL"] = kl_divergence(r_logits, g_logits)
        except NotImplementedError as e:
            results["FAD"] = None
            results["KL"] = None
            print(f"FAD/KL skipped: {e}")
        try:
            a_emb = embed_audio_av(gen_wavs, sr)
            v_emb = embed_video_av(None)  # caller must wire this
            results["AV-align"] = av_align(a_emb, v_emb)
        except NotImplementedError as e:
            results["AV-align"] = None
            print(f"AV-align skipped: {e}")

    out_path = args.out or str(Path(args.ckpt).parent / "eval.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
