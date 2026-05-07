"""AV-align: cosine similarity between AV-joint embeddings of (gen audio, video).

The original metric uses CAVP (https://github.com/facebookresearch/CAVP). Load
those weights and embed each modality; here we just provide the cosine math.
"""
import torch
import torch.nn.functional as F


def av_align(audio_emb: torch.Tensor, video_emb: torch.Tensor) -> float:
    """audio_emb, video_emb: (N, D) — must be from the same joint embedder."""
    a = F.normalize(audio_emb, dim=-1)
    v = F.normalize(video_emb, dim=-1)
    return float((a * v).sum(-1).mean().item())
