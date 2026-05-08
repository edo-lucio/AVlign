"""Evaluation metrics for FGW alignment results.

Reads:
    <data_root>/fgw_results.json
    <data_root>/fgw_results_plans/<combo>.npz   (requires --save_plans on encode)
    <data_root>/embeddings/<dataset>/<split>/<encoder>_<size>_<modality>.pt

Writes:
    <data_root>/fgw_eval.json   per-combo metrics, augmenting fgw_results.json

Metrics per combination (with hard match π(i) = argmax_j T[i, j]):

    Semantic validity (per held-out text encoder e ≠ bridge):
        caption_sim_mean_<e>:   mean cos(image-caps[i], audio-caps[π(i)])
        caption_sim_random_<e>: same, with a random permutation π
        caption_sim_lift_<e>:   mean − random
        recall@1/5/10_<e>:      fraction of images whose FGW-predicted match
                                lies in the caption-oracle top-k under e
        mean_rank_<e>:          mean oracle rank of π(i) in [1, n_a]

    Structural / geometric alignment (encoder-independent):
        pearson_dist_corr:   ρ_P(C_i[i,j], C_a[π(i), π(j)]) over all (i, j)
        spearman_dist_corr:  same with rank correlation
        triplet_agreement:   fraction of NN-triplets preserved under π

    Transport plan quality:
        entropy_norm:        H(T) / log(n_i · n_a)  ∈ [0, 1]
        top1_mass:           mean over rows of (max_j T[i, j]) · n_i
                             1.0 = each row puts all mass on its argmax
        mutual_best_rate:    fraction of i with argmax_i T[:, π(i)] = i
        coverage:            |{π(i) : i}| / n_a   (1.0 = every audio matched)

CLI:
    python -m fgw_validation.eval
    python -m fgw_validation.eval --results path/to/fgw_results.json
    python -m fgw_validation.eval --results path/to/results/*.json   # sweep
"""
import argparse
import functools
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


TEXT_ENCODERS = ("clip", "clap", "roberta", "t5")

_SIZE_PER_ENCODER = {
    "clip":    "small",
    "dinov2":  "small",
    "clap":    "medium",
    "ast":     "medium",
    "roberta": "small",
    "t5":      "small",
}


# ─── loading ────────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=None)
def _load_emb_file(path: str) -> dict:
    return torch.load(path, weights_only=False)


def _emb_path(emb_root: Path, dataset: str, split: str,
              encoder: str, modality: str) -> Path:
    return emb_root / dataset / split / f"{encoder}_{_SIZE_PER_ENCODER[encoder]}_{modality}.pt"


def _load_inner_C(emb_root, dataset, split, encoder, modality, idx):
    d = _load_emb_file(str(_emb_path(emb_root, dataset, split, encoder, modality)))
    z = d["emb"][idx].numpy()
    return _cos_dist(z)


def _load_text(emb_root, dataset, split, encoder, idx):
    """Mean-pooled caption embeddings at sampled indices: (n, D)."""
    d = _load_emb_file(str(_emb_path(emb_root, dataset, split, encoder, "text")))
    return d["emb"][idx].mean(dim=1).numpy()


# ─── numpy helpers ──────────────────────────────────────────────────────────

def _l2norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, eps)


def _cos_dist(x: np.ndarray) -> np.ndarray:
    return 1.0 - _l2norm(x) @ _l2norm(x).T


def _rank(x: np.ndarray) -> np.ndarray:
    """1D ordinal ranks (no tie-handling — ties are vanishingly rare on cosines)."""
    order = np.argsort(x)
    r = np.empty_like(order)
    r[order] = np.arange(len(x))
    return r


def _pearson(a, b):  return float(np.corrcoef(a, b)[0, 1])
def _spearman(a, b): return _pearson(_rank(a).astype(float), _rank(b).astype(float))


# ─── metric blocks ──────────────────────────────────────────────────────────

def _transport_stats(T: np.ndarray) -> dict:
    n_i, n_a = T.shape
    pi     = T.argmax(axis=1)
    pi_inv = T.argmax(axis=0)
    H      = -(T * np.log(T + 1e-30)).sum()
    return {
        "entropy_norm":     float(H / np.log(n_i * n_a)),
        "top1_mass":        float(T.max(axis=1).mean() * n_i),
        "mutual_best_rate": float(np.mean([pi_inv[pi[i]] == i for i in range(n_i)])),
        "coverage":         float(len(set(pi.tolist())) / n_a),
    }, pi


def _structural(C_i: np.ndarray, C_a: np.ndarray, pi: np.ndarray) -> dict:
    n = C_i.shape[0]
    iu = np.triu_indices(n, k=1)
    a = C_i[iu[0], iu[1]]
    b = C_a[pi[iu[0]], pi[iu[1]]]
    return {
        "pearson_dist_corr":  _pearson(a, b),
        "spearman_dist_corr": _spearman(a, b),
    }


def _triplet_agreement(C_i, C_a, pi, n_triplets, rng) -> float:
    n = C_i.shape[0]
    if n < 3:
        return float("nan")
    a = rng.integers(0, n, n_triplets)
    b = rng.integers(0, n, n_triplets)
    c = rng.integers(0, n, n_triplets)
    keep = (a != b) & (b != c) & (a != c)
    a, b, c = a[keep], b[keep], c[keep]
    if len(a) == 0:
        return float("nan")
    img_close = C_i[a, b] < C_i[a, c]
    aud_close = C_a[pi[a], pi[b]] < C_a[pi[a], pi[c]]
    return float((img_close == aud_close).mean())


def _semantic(emb_root, splits, idx_i, idx_a, pi, bridge, ks, rng) -> dict:
    """Held-out-encoder caption-similarity metrics."""
    n_i, n_a = len(idx_i), len(idx_a)
    out: dict = {}
    for held in TEXT_ENCODERS:
        if held == bridge:
            continue
        ti = _load_text(emb_root, "flickr8k", splits["flickr8k"], held, idx_i)
        ta = _load_text(emb_root, "clotho",   splits["clotho"],   held, idx_a)
        sim = _l2norm(ti) @ _l2norm(ta).T                                  # (n_i, n_a)

        match    = sim[np.arange(n_i), pi]
        rand_pi  = rng.permutation(n_a)[:n_i]
        rand_sim = sim[np.arange(n_i), rand_pi]

        # rank of pi[i] in row i (1-indexed, lower=better)
        order    = np.argsort(-sim, axis=1)
        rank_pos = np.argmax(order == pi[:, None], axis=1) + 1

        out[f"caption_sim_mean_{held}"]   = float(match.mean())
        out[f"caption_sim_random_{held}"] = float(rand_sim.mean())
        out[f"caption_sim_lift_{held}"]   = float(match.mean() - rand_sim.mean())
        out[f"mean_rank_{held}"]          = float(rank_pos.mean())
        for k in ks:
            out[f"recall@{k}_{held}"] = float((rank_pos <= k).mean())
    return out


# ─── per-combo evaluation ───────────────────────────────────────────────────

def _evaluate_combo(combo, plan_dir, emb_root, splits, n_triplets, ks, rng):
    if "error" in combo:
        return combo
    tag = "__".join([combo["image_encoder"], combo["audio_encoder"],
                     combo["text_encoder"], combo["cost_convention"],
                     combo["caption_agg"]])
    plan_path = plan_dir / f"{tag}.npz"
    if not plan_path.exists():
        return {**combo, "eval_error": f"missing plan: {plan_path.name}"}

    plan  = np.load(plan_path, allow_pickle=True)
    T     = plan["T"]
    idx_i = plan["idx_i"]
    idx_a = plan["idx_a"]

    transport, pi = _transport_stats(T)
    C_i = _load_inner_C(emb_root, "flickr8k", splits["flickr8k"],
                        combo["image_encoder"], "image", idx_i)
    C_a = _load_inner_C(emb_root, "clotho",   splits["clotho"],
                        combo["audio_encoder"], "audio", idx_a)
    structural = _structural(C_i, C_a, pi)
    structural["triplet_agreement"] = _triplet_agreement(C_i, C_a, pi, n_triplets, rng)
    semantic = _semantic(emb_root, splits, idx_i, idx_a, pi,
                         combo["text_encoder"], ks, rng)

    return {**combo, **transport, **structural, **semantic}


# ─── main ───────────────────────────────────────────────────────────────────

def _evaluate_one(results_path: Path, args) -> Path:
    plan_dir = (Path(args.plans) if args.plans
                else results_path.parent / (results_path.stem + "_plans"))
    out_path = (Path(args.out) if args.out
                else results_path.parent / f"{results_path.stem}_eval.json")

    if not plan_dir.exists():
        raise SystemExit(
            f"plan directory not found: {plan_dir}\n"
            f"Re-run `fgw_validation.fgw_text_bridge --save_plans` to produce them."
        )

    with open(results_path) as f:
        bundle = json.load(f)

    splits = bundle["splits"]
    combos = bundle["results"]
    emb_root = Path(args.emb)
    rng = np.random.default_rng(args.seed)

    enriched = []
    for combo in tqdm(combos, desc=f"eval {results_path.name}"):
        try:
            enriched.append(_evaluate_combo(
                combo, plan_dir, emb_root, splits,
                n_triplets=args.n_triplets, ks=args.ks, rng=rng,
            ))
        except Exception as e:
            enriched.append({**combo, "eval_error": f"{type(e).__name__}: {e}"})

    out = {**bundle,
           "results": enriched,
           "eval": {"seed": args.seed, "ks": args.ks, "n_triplets": args.n_triplets}}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[save] {out_path}  ({len(enriched)} rows)")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+",
                    default=["fgw_validation/data/fgw_results.json"],
                    help="One or more results JSON files (shell globs OK).")
    ap.add_argument("--plans",   default=None,
                    help="Plans dir (only valid with one --results); "
                         "default: <stem>_plans/ next to each results file")
    ap.add_argument("--emb",     default="fgw_validation/data/embeddings")
    ap.add_argument("--out",     default=None,
                    help="Output path (only valid with one --results); "
                         "default: <stem>_eval.json next to each results file")
    ap.add_argument("--n_triplets", type=int, default=2000)
    ap.add_argument("--ks",         type=int, nargs="+", default=[1, 5, 10])
    ap.add_argument("--seed",       type=int, default=0)
    args = ap.parse_args()

    # Drop already-evaluated outputs in case the user globbed them in.
    paths = [Path(p) for p in args.results
             if not p.endswith("_eval.json") and not p.endswith("/fgw_eval.json")]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit("results file(s) not found: " + ", ".join(map(str, missing)))
    if not paths:
        raise SystemExit("no results files to evaluate (after filtering *_eval.json)")
    if (args.plans or args.out) and len(paths) > 1:
        raise SystemExit("--plans/--out require exactly one --results file")

    for p in paths:
        _evaluate_one(p, args)


if __name__ == "__main__":
    main()
