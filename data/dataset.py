"""VGGSoundDataset: returns per-clip preprocessed tensors.

Also provides a SyntheticDataset for the smoke test (no VGGSound required).
"""
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class VGGSoundDataset(Dataset):
    def __init__(self, manifest_path: str, split: str = "train",
                 audio_seq_len: int = 860, audio_latent_dim: int = 1024):
        with open(manifest_path) as f:
            manifest = json.load(f)
        self.clips = [c for c in manifest["clips"] if c["split"] == split]
        self.audio_seq_len = audio_seq_len
        self.audio_latent_dim = audio_latent_dim

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        c = self.clips[idx]
        d = torch.load(c["path"], weights_only=False)
        z_a = d["dac_latent"]                              # (T, D)
        if z_a.shape[0] < self.audio_seq_len:
            pad = self.audio_seq_len - z_a.shape[0]
            z_a = torch.nn.functional.pad(z_a, (0, 0, 0, pad))
        else:
            z_a = z_a[:self.audio_seq_len]
        return {
            "z_v": d["clip_emb"].float(),                  # (768,)
            "z_a": z_a.float(),                            # (T, D)
            "clap_audio_emb": d["clap_audio_emb"].float(), # (512,)
            "class_id": int(c["class_id"]),
            "id": c["id"],
        }


class SyntheticDataset(Dataset):
    """Random tensors with the right shapes — for the smoke test."""
    def __init__(self, n: int = 256, n_classes: int = 16,
                 audio_seq_len: int = 64, audio_latent_dim: int = 64,
                 clip_dim: int = 768, clap_dim: int = 512, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.items = []
        # Make same-class items cluster a bit so FGW has signal.
        class_centers_v = torch.randn(n_classes, clip_dim, generator=g)
        class_centers_a = torch.randn(n_classes, clap_dim, generator=g)
        for i in range(n):
            cls = i % n_classes
            self.items.append({
                "z_v": class_centers_v[cls] + 0.3 * torch.randn(clip_dim, generator=g),
                "z_a": torch.randn(audio_seq_len, audio_latent_dim, generator=g),
                "clap_audio_emb": class_centers_a[cls] + 0.3 * torch.randn(clap_dim, generator=g),
                "class_id": cls,
                "id": f"syn_{i}",
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch):
    out = {
        "z_v": torch.stack([b["z_v"] for b in batch]),
        "z_a": torch.stack([b["z_a"] for b in batch]),
        "clap_audio_emb": torch.stack([b["clap_audio_emb"] for b in batch]),
        "class_id": torch.tensor([b["class_id"] for b in batch], dtype=torch.long),
        "id": [b["id"] for b in batch],
    }
    return out
