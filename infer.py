"""Generate audio from a video frame using a trained checkpoint.

Pipeline:
    image -> CLIP visual -> z_v -> ODE solve -> DAC latent -> DAC decode -> wav
"""
import argparse
from pathlib import Path

import torch
import torchaudio
import yaml
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from models.mmdit import MMDiT
from models.cfm import euler_solve, dopri5_solve


def load_ckpt(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    d = cfg["data"]; m = cfg["model"]
    net = MMDiT(
        audio_latent_dim=d["audio_latent_dim"],
        audio_seq_len=d["audio_seq_len"],
        clip_dim=d["clip_dim"],
        d_model=m["d_model"], n_blocks=m["n_blocks"],
        n_heads=m["n_heads"], d_ffn=m["d_ffn"], dropout=0.0,
    ).to(device).eval()
    net.load_state_dict(ckpt["net"])
    return net, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True, help="path to a frame (jpg/png)")
    ap.add_argument("--out", default="out.wav")
    ap.add_argument("--clip_id", default="openai/clip-vit-large-patch14")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--solver", choices=["euler", "dopri5"], default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    net, cfg = load_ckpt(args.ckpt, device)
    d = cfg["data"]
    steps = args.steps or cfg["infer"]["ode_steps"]
    solver = args.solver or cfg["infer"]["ode_solver"]

    clip_model = CLIPModel.from_pretrained(args.clip_id).to(device).eval()
    clip_proc = CLIPProcessor.from_pretrained(args.clip_id)
    img = Image.open(args.image).convert("RGB")
    inp = clip_proc(images=img, return_tensors="pt").to(device)
    with torch.inference_mode():
        z_v = clip_model.get_image_features(**inp)              # (1, 768)

    if solver == "euler":
        z_a = euler_solve(net, z_v, d["audio_seq_len"], d["audio_latent_dim"],
                          steps=steps, device=device)
    else:
        z_a = dopri5_solve(net, z_v, d["audio_seq_len"], d["audio_latent_dim"],
                           device=device)

    # DAC decode (z is the continuous latent; DAC.decode expects (B, D, T))
    import dac
    dac_path = dac.utils.download(model_type="44khz")
    dac_model = dac.DAC.load(dac_path).to(device).eval()
    z = z_a.transpose(1, 2)                                     # (B, D, T)
    with torch.inference_mode():
        wav = dac_model.decode(z).cpu()                         # (B, 1, N)
    torchaudio.save(args.out, wav[0], dac_model.sample_rate)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
