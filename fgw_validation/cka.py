"""Centered Kernel Alignment (CKA) for comparing representation spaces.

Reference:
    Kornblith, Norouzi, Lee, Hinton.
    "Similarity of Neural Network Representations Revisited." ICML 2019.

Two representations X (n × p1) and Y (n × p2) — same n rows, possibly
different feature dims — are compared via

    CKA(X, Y) = HSIC(K, L) / sqrt(HSIC(K, K) · HSIC(L, L))

where K, L are n × n Gram matrices (linear or RBF kernel) and HSIC is
the *biased* empirical estimator (the one Kornblith uses):

    HSIC(K, L) = (1 / (n − 1)^2) · tr(K H L H),   H = I − (1/n) 1 1ᵀ

For the linear kernel this collapses to a closed form that avoids the
n × n Gram entirely:

    CKA_linear(X, Y) = ‖Yc ᵀ Xc‖_F^2 / (‖Xc ᵀ Xc‖_F · ‖Yc ᵀ Yc‖_F)

with Xc, Yc column-mean-centered.

CLI (pairwise CKA across encoders on the same dataset/split/modality):

    python -m fgw_validation.cka --dataset flickr8k --split test --modality image
    python -m fgw_validation.cka --dataset clotho   --split development --modality audio
    python -m fgw_validation.cka --dataset flickr8k --split test --modality text \
        --kernel rbf --n 500 --seed 0
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from tqdm import tqdm


# ─── math ───────────────────────────────────────────────────────────────────

def _to_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _center_gram(K: np.ndarray) -> np.ndarray:
    """Apply H K H where H = I − (1/n) 1 1ᵀ in O(n²) time/memory."""
    means = K.mean(axis=0)
    Kc = K - means[None, :]
    Kc = Kc - means[:, None]
    return Kc + means.mean()


def gram_linear(X: np.ndarray) -> np.ndarray:
    return X @ X.T


def gram_rbf(X: np.ndarray, sigma: float | None = None,
             threshold: float = 1.0) -> np.ndarray:
    """RBF kernel. If sigma is None, use the median-distance heuristic
    (Kornblith default): sigma = threshold · sqrt(median_sq_dist / 2)."""
    sq = np.einsum("ij,ij->i", X, X)
    sqdist = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    np.maximum(sqdist, 0.0, out=sqdist)
    if sigma is None:
        n = sqdist.shape[0]
        iu = np.triu_indices(n, k=1)
        med = np.median(sqdist[iu]) if n > 1 else 1.0
        sigma = float(np.sqrt(0.5 * med)) if med > 0 else 1.0
        sigma *= threshold
    return np.exp(-sqdist / (2.0 * sigma * sigma))


def hsic_biased(K: np.ndarray, L: np.ndarray) -> float:
    """Biased HSIC (Kornblith Eq. 3): tr(K H L H) / (n − 1)²."""
    n = K.shape[0]
    Kc = _center_gram(K)
    Lc = _center_gram(L)
    return float(np.sum(Kc * Lc) / max((n - 1) ** 2, 1))


def cka_from_grams(K: np.ndarray, L: np.ndarray) -> float:
    """CKA from two precomputed Gram matrices."""
    hkl = hsic_biased(K, L)
    hkk = hsic_biased(K, K)
    hll = hsic_biased(L, L)
    denom = float(np.sqrt(hkk * hll))
    return float("nan") if denom <= 0 else float(hkl / denom)


def linear_cka(X, Y) -> float:
    """Closed-form linear CKA — O(n · max(p1, p2)²) instead of O(n²)."""
    X = _to_numpy(X).astype(np.float64, copy=False)
    Y = _to_numpy(Y).astype(np.float64, copy=False)
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    cross = Xc.T @ Yc                     # (p1, p2)
    num = float(np.sum(cross * cross))    # ‖Ycᵀ Xc‖_F²
    denom = float(np.linalg.norm(Xc.T @ Xc, "fro")
                  * np.linalg.norm(Yc.T @ Yc, "fro"))
    return float("nan") if denom <= 0 else num / denom


def kernel_cka(X, Y, kernel: str = "rbf",
               sigma_x: float | None = None, sigma_y: float | None = None,
               threshold: float = 1.0) -> float:
    """Kernel CKA via HSIC. kernel ∈ {linear, rbf}."""
    X = _to_numpy(X).astype(np.float64, copy=False)
    Y = _to_numpy(Y).astype(np.float64, copy=False)
    if kernel == "linear":
        K, L = gram_linear(X), gram_linear(Y)
    elif kernel == "rbf":
        K = gram_rbf(X, sigma=sigma_x, threshold=threshold)
        L = gram_rbf(Y, sigma=sigma_y, threshold=threshold)
    else:
        raise ValueError(f"unknown kernel: {kernel!r}")
    return cka_from_grams(K, L)


def cka(X, Y, kernel: str = "linear", **kwargs) -> float:
    """Dispatch: closed-form linear CKA, or HSIC-based kernel CKA."""
    if kernel == "linear":
        return linear_cka(X, Y)
    return kernel_cka(X, Y, kernel=kernel, **kwargs)


def pairwise_cka(reps: dict[str, np.ndarray], kernel: str = "linear",
                 **kwargs) -> dict:
    """Symmetric CKA matrix over a {name → (n, d_name)} dict.

    All matrices must share n (same row alignment). Returns
    {"names": [...], "matrix": [[...], ...]} with 1.0 on the diagonal.
    """
    names = list(reps.keys())
    ns = {reps[k].shape[0] for k in names}
    if len(ns) > 1:
        raise ValueError(f"row counts disagree across reps: {ns}")
    m = len(names)
    M = np.eye(m, dtype=np.float64)
    for i, j in itertools.combinations(range(m), 2):
        v = cka(reps[names[i]], reps[names[j]], kernel=kernel, **kwargs)
        M[i, j] = M[j, i] = v
    return {"names": names, "matrix": M.tolist()}


# ─── CLI: pairwise CKA across encoders ──────────────────────────────────────

_SIZE_PER_ENCODER = {
    "clip":    "small",
    "dinov2":  "small",
    "clap":    "medium",
    "ast":     "medium",
    "roberta": "small",
    "t5":      "small",
}

_ENCODERS_BY_MODALITY = {
    "image": ("clip", "dinov2"),
    "audio": ("clap", "ast"),
    "text":  ("clip", "clap", "roberta", "t5"),
}


def _emb_path(emb_root: Path, dataset: str, split: str,
              encoder: str, modality: str) -> Path:
    return emb_root / dataset / split / f"{encoder}_{_SIZE_PER_ENCODER[encoder]}_{modality}.pt"


def _load_rep(path: Path, modality: str, idx: np.ndarray) -> tuple[np.ndarray, list]:
    """Returns (emb[idx], ids[idx]). Text reps are mean-pooled across the
    5-caption axis so all three modalities yield (n, D)."""
    d = torch.load(str(path), weights_only=False)
    emb = d["emb"]
    if modality == "text":
        emb = emb.mean(dim=1)             # (N, 5, D) → (N, D)
    emb = emb[idx].numpy().astype(np.float64)
    ids = [d["ids"][i] for i in idx]
    return emb, ids


def _aligned_indices(emb_root: Path, dataset: str, split: str,
                     modality: str, encoders: Iterable[str],
                     n: int | None, seed: int) -> tuple[np.ndarray, list]:
    """Pick n row indices that exist (with the same id ordering) in every
    encoder's file. We rely on encode.py producing identical id orderings
    per (dataset, split, modality), so a simple length check + intersection
    of the first encoder's ids is enough."""
    encoders = list(encoders)
    first = torch.load(str(_emb_path(emb_root, dataset, split, encoders[0], modality)),
                       weights_only=False)
    base_ids = first["ids"]
    N = len(base_ids)
    for e in encoders[1:]:
        d = torch.load(str(_emb_path(emb_root, dataset, split, e, modality)),
                       weights_only=False)
        if d["ids"] != base_ids:
            raise ValueError(
                f"id ordering differs between {encoders[0]} and {e} for "
                f"{dataset}/{split}/{modality}; re-encode to align them"
            )
    rng = np.random.default_rng(seed)
    if n is None or n >= N:
        idx = np.arange(N)
    else:
        idx = rng.choice(N, size=n, replace=False)
        idx.sort()
    sampled_ids = [base_ids[i] for i in idx]
    return idx, sampled_ids


def _resolve_encoders(modality: str, requested: Iterable[str] | None) -> list[str]:
    available = _ENCODERS_BY_MODALITY[modality]
    if not requested:
        return list(available)
    bad = [e for e in requested if e not in available]
    if bad:
        raise ValueError(f"encoders {bad} not valid for modality={modality!r} "
                         f"(allowed: {available})")
    return list(requested)


def _exists(emb_root: Path, dataset: str, split: str,
            encoder: str, modality: str) -> bool:
    return _emb_path(emb_root, dataset, split, encoder, modality).exists()


def _run_cli(args) -> dict:
    emb_root = Path(args.emb_root)
    encoders = _resolve_encoders(args.modality, args.encoders)
    encoders = [e for e in encoders
                if _exists(emb_root, args.dataset, args.split, e, args.modality)]
    if len(encoders) < 2:
        raise SystemExit(
            f"need ≥ 2 encoders with embeddings for "
            f"{args.dataset}/{args.split}/{args.modality}; found {encoders}"
        )

    idx, sampled_ids = _aligned_indices(
        emb_root, args.dataset, args.split, args.modality,
        encoders, args.n, args.seed,
    )

    reps: dict[str, np.ndarray] = {}
    for e in tqdm(encoders, desc=f"load {args.modality}"):
        emb, _ = _load_rep(_emb_path(emb_root, args.dataset, args.split, e, args.modality),
                           args.modality, idx)
        reps[e] = emb

    kernel_kwargs = {}
    if args.kernel == "rbf":
        kernel_kwargs["threshold"] = args.rbf_threshold

    out = pairwise_cka(reps, kernel=args.kernel, **kernel_kwargs)
    out.update({
        "dataset":  args.dataset,
        "split":    args.split,
        "modality": args.modality,
        "kernel":   args.kernel,
        "n":        len(idx),
        "seed":     args.seed,
        "encoders": encoders,
        "ids":      sampled_ids,
    })
    if args.kernel == "rbf":
        out["rbf_threshold"] = args.rbf_threshold
    return out


def _print_matrix(out: dict) -> None:
    names = out["names"]
    M = np.asarray(out["matrix"])
    width = max(len(n) for n in names)
    head = " " * (width + 2) + "  ".join(f"{n:>6}" for n in names)
    print(head)
    for i, n in enumerate(names):
        row = "  ".join(f"{M[i, j]:6.3f}" for j in range(len(names)))
        print(f"{n:>{width}}  {row}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--emb_root", default="fgw_validation/data/embeddings")
    p.add_argument("--dataset", required=True, choices=("flickr8k", "clotho"))
    p.add_argument("--split", required=True,
                   help="flickr8k: train/val/test; clotho: development/validation/evaluation")
    p.add_argument("--modality", required=True,
                   choices=("image", "audio", "text"))
    p.add_argument("--encoders", nargs="+", default=None,
                   help="subset of encoders to compare; default = all available")
    p.add_argument("--kernel", default="linear", choices=("linear", "rbf"))
    p.add_argument("--rbf_threshold", type=float, default=1.0,
                   help="RBF bandwidth = threshold · sqrt(median_sq_dist / 2)")
    p.add_argument("--n", type=int, default=1000,
                   help="random subsample size (use 0 for all rows)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None,
                   help="JSON output path; default fgw_validation/data/cka_<dataset>_<split>_<modality>_<kernel>.json")
    args = p.parse_args()
    if args.n == 0:
        args.n = None

    out = _run_cli(args)
    _print_matrix(out)

    out_path = Path(
        args.out or f"fgw_validation/data/cka_{args.dataset}_{args.split}_"
                    f"{args.modality}_{args.kernel}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
