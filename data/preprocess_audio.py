"""Encode each clip's audio with DAC (continuous latent) and CLAP-audio.

Adds keys to data/cache/shards/<ytid>.pt:
  - dac_latent     (T, D)  continuous DAC latent (pre-quantization)
  - clap_audio_emb (512,)  CLAP-audio pooled embedding

Run after preprocess_video.py — it expects shard files to already exist.

NOTE: stock DAC-44k continuous latents are 1024-dim. The spec uses 64. Either
adjust the model's audio_latent_dim, or add a `proj_to_64` step here.
"""
import argparse
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

import dac
from transformers import ClapModel, ClapProcessor


def load_audio(video_path: Path, target_sr: int = 44100) -> torch.Tensor | None:
    try:
        wav, sr = torchaudio.load(str(video_path))
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        return wav  # (1, N)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir", required="data/vggsound")
    ap.add_argument("--out_dir", default="data/cache")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--clap_id", default="laion/clap-htsat-fused")
    ap.add_argument("--max_seconds", type=float, default=10.0)
    args = ap.parse_args()

    shard_dir = Path(args.out_dir) / "shards"
    shards = sorted(shard_dir.glob("*.pt"))
    if not shards:
        print(f"no shards in {shard_dir}; run preprocess_video.py first")
        return

    # DAC
    dac_path = dac.utils.download(model_type="44khz")
    dac_model = dac.DAC.load(dac_path).to(args.device).eval()
    sr = dac_model.sample_rate

    # CLAP-audio
    clap = ClapModel.from_pretrained(args.clap_id).to(args.device).eval()
    clap_proc = ClapProcessor.from_pretrained(args.clap_id)
    clap_sr = clap_proc.feature_extractor.sampling_rate

    target_len = int(sr * args.max_seconds)

    for shard_path in tqdm(shards):
        d = torch.load(shard_path)
        if "dac_latent" in d and "clap_audio_emb" in d:
            continue
        ytid = shard_path.stem
        vid = Path(args.video_dir) / f"{ytid}.mp4"
        if not vid.exists():
            continue
        wav = load_audio(vid, sr)
        if wav is None:
            continue
        # Pad/trim to fixed length
        if wav.shape[1] < target_len:
            wav = torch.nn.functional.pad(wav, (0, target_len - wav.shape[1]))
        else:
            wav = wav[:, :target_len]

        with torch.inference_mode():
            x = wav.unsqueeze(0).to(args.device)         # (1, 1, N)
            x_pp = dac_model.preprocess(x, sr)
            z, _, _, _, _ = dac_model.encode(x_pp)        # (1, D, T)
            z = z.squeeze(0).transpose(0, 1).cpu()        # (T, D)
            d["dac_latent"] = z

            wav_clap = torchaudio.functional.resample(wav, sr, clap_sr).squeeze(0).numpy()
            inp = clap_proc(audios=wav_clap, sampling_rate=clap_sr,
                            return_tensors="pt").to(args.device)
            d["clap_audio_emb"] = clap.get_audio_features(**inp).squeeze(0).cpu()

        torch.save(d, shard_path)

    # Build manifest
    clips = []
    for shard_path in sorted(shard_dir.glob("*.pt")):
        d = torch.load(shard_path, weights_only=False)
        if all(k in d for k in ("clip_emb", "dac_latent", "clap_audio_emb", "class_id")):
            clips.append({
                "id": shard_path.stem,
                "path": str(shard_path),
                "class_id": int(d["class_id"]),
                "split": d.get("split", "train"),
            })
    import json
    with open(Path(args.out_dir) / "manifest.json", "w") as f:
        json.dump({"clips": clips}, f)
    print(f"manifest: {len(clips)} clips")


if __name__ == "__main__":
    main()
