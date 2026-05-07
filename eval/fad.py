"""Frechet Audio Distance.

Operates on (N, D) embedding matrices for real and generated audio. The user
supplies an embedder (typically VGGish or PANNs); we provide the FAD math.
"""
import numpy as np
import torch
from scipy import linalg


def gaussian_stats(x: np.ndarray):
    mu = x.mean(0)
    sigma = np.cov(x, rowvar=False)
    return mu, sigma


def frechet_distance(mu1, s1, mu2, s2, eps: float = 1e-6) -> float:
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(s1 @ s2, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(s1.shape[0]) * eps
        covmean = linalg.sqrtm((s1 + offset) @ (s2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def fad(real_emb: np.ndarray, gen_emb: np.ndarray) -> float:
    mu_r, s_r = gaussian_stats(real_emb)
    mu_g, s_g = gaussian_stats(gen_emb)
    return frechet_distance(mu_r, s_r, mu_g, s_g)
