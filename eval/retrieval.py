"""R@k retrieval: given gen-audio embeddings, retrieve true video from a pool."""
import torch
import torch.nn.functional as F


def recall_at_k(audio_emb: torch.Tensor, video_emb: torch.Tensor,
                k: int = 1) -> float:
    """audio_emb[i] should retrieve video_emb[i]. (N, D), (N, D)."""
    a = F.normalize(audio_emb, dim=-1)
    v = F.normalize(video_emb, dim=-1)
    sim = a @ v.t()                                     # (N, N)
    n = sim.shape[0]
    topk = sim.topk(k, dim=-1).indices                  # (N, k)
    target = torch.arange(n, device=sim.device).unsqueeze(-1)
    hits = (topk == target).any(dim=-1).float()
    return float(hits.mean().item())
