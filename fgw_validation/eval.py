"""Evaluation metrics for FGW alignment results.

Reads:
    <data_root>/fgw_results.json
    <data_root>/fgw_results_plans/<combo>.npz   (requires --save_plans on encode)
    <data_root>/embeddings/<dataset>/<split>/<encoder>_<size>_<modality>.pt

Writes:
    <data_root>/fgw_eval.json   per-combo metrics, augmenting fgw_results.json

Two reference points for every metric (with hard match π(i) = argmax_j T[i, j]):

    FGW: π is taken from the FGW transport plan T.
    Baseline: π_baseline(i) = argmax_j cos(bridge_text(img_caps[i]),
                                           bridge_text(aud_caps[j])) — text-only
        caption-to-caption retrieval in the bridge encoder, no FGW. The same
        metrics computed with this π appear with a `baseline_` prefix. Reading
        FGW vs baseline tells you whether FGW adds value beyond bridge-text
        retrieval *for semantic metrics*. For structural metrics, the right
        comparison is the random-π null (`null_*` columns) — the text-only
        baseline ignores C_i, C_a and necessarily scores low on structural.
        Disable baseline with --no_baseline.

Metrics per combination:

    Semantic validity (per witness — held-out text encoder e ≠ bridge,
                                       or encoder-free `lex` Jaccard):
        caption_sim_mean_<e>:    mean cos/jaccard for π(i)
        caption_sim_random_<e>:  same, with one random permutation π
        caption_sim_lift_<e>:    mean − random
        recall@{1,5,10}_<e>:     fraction of i where π(i) ∈ top-k of row i
        soft_recall@{1,5,10}_<e>: Σ_{i,j} T[i,j]·1[j ∈ topk(i)] (FGW only)
        mean_rank_<e>:           mean oracle rank of π(i) in [1, n_a]
        median_rank_<e>:         median rank (robust to right tail)
        mrr_<e>:                 mean reciprocal rank

    Structural / geometric alignment (encoder-independent):
        pearson_dist_corr:   ρ_P(C_i[i,j], C_a[π(i), π(j)]) over upper triangle
        spearman_dist_corr:  same with rank correlation
        triplet_agreement:   fraction of NN-triplets preserved under π
        null_pearson_*:      pearson under K random permutations (structural null)
        null_spearman_*:     same for spearman

    Transport plan quality (FGW-only):
        entropy_norm, top1_mass, mutual_best_rate, coverage.

Bootstrap CIs:
    Every semantic, structural, and triplet metric carries `_lo` and `_hi`
    fields giving the bootstrap percentile interval (default 95 % from
    B = 1000 resamples). Structural correlations use **item-level** bootstrap
    (resample image indices, recompute the n×n submatrix correlation) to
    respect inter-pair dependence. Tune via --bootstrap_B and --bootstrap_ci.

CLI:
    python -m fgw_validation.eval
    python -m fgw_validation.eval --results path/to/fgw_results.json
    python -m fgw_validation.eval --results path/to/results/*.json   # sweep
"""
import argparse
import functools
import json
import re
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


# ─── bootstrap helpers ──────────────────────────────────────────────────────

def _bootstrap_mean_ci(values: np.ndarray, B: int, ci: float, rng) -> tuple:
    """Percentile bootstrap CI of the mean of a 1-D vector."""
    n = len(values)
    if n == 0 or B <= 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, n, size=(B, n))
    means = values[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def _bootstrap_struct_corr_ci(C_i: np.ndarray, C_a_pi: np.ndarray,
                              B: int, ci: float, rng, chunk: int = 16
                              ) -> tuple[tuple[float, float], tuple[float, float]]:
    """**Item-level** bootstrap CI of (Pearson, Spearman) for the structural
    correlation between `C_i` and `π`-aligned `C_a`.

    Each item participates in n−1 pairs of `C`, so resampling pairs of pairs
    ignores that dependence and produces anticonservative CIs. Here we resample
    image indices `b ~ Uniform([0, n))^n` with replacement and recompute the
    upper-triangle correlation on the resampled n×n submatrix.

    Returns ((p_lo, p_hi), (s_lo, s_hi)).
    """
    n = C_i.shape[0]
    nan_pair = (float("nan"), float("nan"))
    if n < 3 or B <= 0:
        return nan_pair, nan_pair
    iu0, iu1 = np.triu_indices(n, k=1)
    p_vals = np.empty(B, dtype=np.float64)
    s_vals = np.empty(B, dtype=np.float64)
    pos = 0
    while pos < B:
        cur = min(chunk, B - pos)
        b = rng.integers(0, n, size=(cur, n))
        u = b[:, iu0]; v = b[:, iu1]                       # (cur, P)
        a = C_i[u, v]; c = C_a_pi[u, v]                    # (cur, P)
        # Pearson per row.
        am = a - a.mean(axis=1, keepdims=True)
        cm = c - c.mean(axis=1, keepdims=True)
        num = (am * cm).sum(axis=1)
        den = np.sqrt((am * am).sum(axis=1) * (cm * cm).sum(axis=1)) + 1e-30
        p_vals[pos:pos + cur] = num / den
        # Spearman: rank within each resample (argsort-of-argsort, no tie handling).
        a_r = np.argsort(np.argsort(a, axis=1), axis=1).astype(np.float64)
        c_r = np.argsort(np.argsort(c, axis=1), axis=1).astype(np.float64)
        am = a_r - a_r.mean(axis=1, keepdims=True)
        cm = c_r - c_r.mean(axis=1, keepdims=True)
        num = (am * cm).sum(axis=1)
        den = np.sqrt((am * am).sum(axis=1) * (cm * cm).sum(axis=1)) + 1e-30
        s_vals[pos:pos + cur] = num / den
        pos += cur
    alpha = (1.0 - ci) / 2.0
    return ((float(np.quantile(p_vals, alpha)), float(np.quantile(p_vals, 1.0 - alpha))),
            (float(np.quantile(s_vals, alpha)), float(np.quantile(s_vals, 1.0 - alpha))))


# ─── structural null (random-π reference for structural correlations) ──────

def _structural_null(C_i: np.ndarray, C_a: np.ndarray,
                     K: int, ci: float, rng) -> dict:
    """Random-permutation null for `pearson_dist_corr` / `spearman_dist_corr`.

    Independent of any chosen `π`: depends only on (C_i, C_a). Reports the
    distribution mean and a percentile interval. A structural correlation only
    exceeds chance if it lies outside [`null_*_lo`, `null_*_hi`].
    """
    n = C_i.shape[0]
    if n < 3 or K <= 0:
        return {}
    iu0, iu1 = np.triu_indices(n, k=1)
    a = C_i[iu0, iu1]
    p_vals = np.empty(K, dtype=np.float64)
    s_vals = np.empty(K, dtype=np.float64)
    for k in range(K):
        rp = rng.permutation(n)
        b = C_a[rp[iu0], rp[iu1]]
        p_vals[k] = _pearson(a, b)
        s_vals[k] = _spearman(a, b)
    alpha = (1.0 - ci) / 2.0
    return {
        "null_pearson_mean":  float(p_vals.mean()),
        "null_pearson_lo":    float(np.quantile(p_vals, alpha)),
        "null_pearson_hi":    float(np.quantile(p_vals, 1.0 - alpha)),
        "null_spearman_mean": float(s_vals.mean()),
        "null_spearman_lo":   float(np.quantile(s_vals, alpha)),
        "null_spearman_hi":   float(np.quantile(s_vals, 1.0 - alpha)),
    }


# ─── matching helpers ──────────────────────────────────────────────────────

def _text_only_pi(emb_root, splits, idx_i, idx_a, bridge) -> np.ndarray:
    """Baseline matching: π(i) = argmax_j cos(bridge(img-caps[i]), bridge(aud-caps[j]))."""
    ti = _load_text(emb_root, "flickr8k", splits["flickr8k"], bridge, idx_i)
    ta = _load_text(emb_root, "clotho",   splits["clotho"],   bridge, idx_a)
    sim = _l2norm(ti) @ _l2norm(ta).T
    return sim.argmax(axis=1)


# ─── encoder-free lexical witness ──────────────────────────────────────────

_TOK_RE = re.compile(r"\b[a-z']+\b")
# Small functional stop-word list — purely to remove tokens that contribute
# no semantic discrimination (so Jaccard isn't dominated by "the").
_STOP = frozenset("""
a an the and or but of in on at to for from with by as is are was were be been being
this that these those it its his her our their my your he she we they i you us them him me
have has had do does did will would should could may might must can not no
up down out over under more most some any all each every few many several both
""".split())


def _tokenize_caps(caps) -> set:
    """Union token set across the 5 captions, lowercased and stop-stripped."""
    toks: set = set()
    for c in caps:
        for t in _TOK_RE.findall(c.lower()):
            if t not in _STOP and len(t) >= 3:
                toks.add(t)
    return toks


@functools.lru_cache(maxsize=8)
def _captions_for(emb_root_str: str, dataset: str, split: str) -> dict:
    """{id: [5 caption strings]} for the (dataset, split). Returns {} on failure
    so the lexical witness silently degrades when raw datasets aren't present.
    """
    try:
        from fgw_validation.datasets import Flickr8kDataset, ClothoDataset
    except Exception:
        return {}
    data_root = Path(emb_root_str).parent
    try:
        if dataset == "flickr8k":
            ds = Flickr8kDataset(data_root / "flickr8k", split=split)
        elif dataset == "clotho":
            ds = ClothoDataset(data_root / "clotho", split=split)
        else:
            return {}
        return {item["id"]: item["captions"] for item in ds}
    except Exception:
        return {}


def _ids_for(emb_root, dataset, split, idx) -> list:
    """Resolve `idx` into the embedding's id list. Tries each text encoder
    in turn so we don't depend on any single one being present."""
    for enc in TEXT_ENCODERS:
        try:
            d = _load_emb_file(str(_emb_path(emb_root, dataset, split, enc, "text")))
            ids_all = d["ids"]
            return [ids_all[k] for k in (idx.tolist() if hasattr(idx, "tolist") else list(idx))]
        except FileNotFoundError:
            continue
    return []


def _lexical_sim(emb_root, splits, idx_i, idx_a):
    """Encoder-free Jaccard similarity between the union token sets of each
    item's 5 captions. Returns None if dataset captions can't be loaded."""
    ids_i = _ids_for(emb_root, "flickr8k", splits["flickr8k"], idx_i)
    ids_a = _ids_for(emb_root, "clotho",   splits["clotho"],   idx_a)
    if not ids_i or not ids_a:
        return None
    caps_i = _captions_for(str(emb_root), "flickr8k", splits["flickr8k"])
    caps_a = _captions_for(str(emb_root), "clotho",   splits["clotho"])
    if not caps_i or not caps_a:
        return None

    toks_i = [_tokenize_caps(caps_i.get(k, [])) for k in ids_i]
    toks_a = [_tokenize_caps(caps_a.get(k, [])) for k in ids_a]
    n_i, n_a = len(toks_i), len(toks_a)
    sim = np.zeros((n_i, n_a), dtype=np.float64)
    for i in range(n_i):
        ti = toks_i[i]
        if not ti:
            continue
        for j in range(n_a):
            tj = toks_a[j]
            if not tj:
                continue
            inter = len(ti & tj)
            union = len(ti | tj)
            if union:
                sim[i, j] = inter / union
    return sim


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


def _structural(C_i: np.ndarray, C_a: np.ndarray, pi: np.ndarray,
                B: int, ci: float, rng, prefix: str = "") -> dict:
    n = C_i.shape[0]
    iu0, iu1 = np.triu_indices(n, k=1)
    C_a_pi = C_a[pi][:, pi]
    a = C_i[iu0, iu1]
    b = C_a_pi[iu0, iu1]
    out = {
        f"{prefix}pearson_dist_corr":  _pearson(a, b),
        f"{prefix}spearman_dist_corr": _spearman(a, b),
    }
    if B > 0:
        (p_lo, p_hi), (s_lo, s_hi) = _bootstrap_struct_corr_ci(C_i, C_a_pi, B, ci, rng)
        out[f"{prefix}pearson_dist_corr_lo"]  = p_lo
        out[f"{prefix}pearson_dist_corr_hi"]  = p_hi
        out[f"{prefix}spearman_dist_corr_lo"] = s_lo
        out[f"{prefix}spearman_dist_corr_hi"] = s_hi
    return out


def _triplet_agreement(C_i, C_a, pi, n_triplets, B, ci, rng,
                       prefix: str = "") -> dict:
    key = f"{prefix}triplet_agreement"
    out: dict = {key: float("nan")}
    if B > 0:
        out[f"{key}_lo"] = float("nan")
        out[f"{key}_hi"] = float("nan")
    n = C_i.shape[0]
    if n < 3:
        return out
    a = rng.integers(0, n, n_triplets)
    b = rng.integers(0, n, n_triplets)
    c = rng.integers(0, n, n_triplets)
    keep = (a != b) & (b != c) & (a != c)
    a, b, c = a[keep], b[keep], c[keep]
    if len(a) == 0:
        return out
    img_close = C_i[a, b] < C_i[a, c]
    aud_close = C_a[pi[a], pi[b]] < C_a[pi[a], pi[c]]
    agree = (img_close == aud_close).astype(float)
    out[key] = float(agree.mean())
    if B > 0:
        out[f"{key}_lo"], out[f"{key}_hi"] = _bootstrap_mean_ci(agree, B, ci, rng)
    return out


def _witness_sim(emb_root, splits, idx_i, idx_a, witness: str):
    """Witness similarity matrix `(n_i, n_a)` for a held-out witness.

    Encoder witnesses use mean-pooled caption cosine; `lex` uses Jaccard over
    tokenised captions (encoder-free). Returns None if the witness is
    unavailable (e.g. dataset captions missing for `lex`).
    """
    if witness == "lex":
        return _lexical_sim(emb_root, splits, idx_i, idx_a)
    ti = _load_text(emb_root, "flickr8k", splits["flickr8k"], witness, idx_i)
    ta = _load_text(emb_root, "clotho",   splits["clotho"],   witness, idx_a)
    return _l2norm(ti) @ _l2norm(ta).T


def _semantic(emb_root, splits, idx_i, idx_a, pi, bridge, ks,
              B, ci, rng, prefix: str = "", T=None) -> dict:
    """Per-witness caption-similarity metrics with optional bootstrap CIs.

    Witnesses: every text encoder ≠ bridge, plus the encoder-free `lex`
    (Jaccard over caption tokens). When `T` is provided, FGW soft-recall
    `Σ_{i,j} T[i,j]·1[j ∈ topk(i)]` is also reported.
    """
    n_i, n_a = len(idx_i), len(idx_a)
    out: dict = {}

    witnesses = [enc for enc in TEXT_ENCODERS if enc != bridge] + ["lex"]

    for held in witnesses:
        sim = _witness_sim(emb_root, splits, idx_i, idx_a, held)
        if sim is None:
            continue                                  # graceful skip (e.g. lex without raw captions)

        match    = sim[np.arange(n_i), pi]
        rand_pi  = rng.permutation(n_a)[:n_i]
        rand_sim = sim[np.arange(n_i), rand_pi]
        lift     = match - rand_sim

        # rank of pi[i] in row i (1-indexed, lower=better)
        order    = np.argsort(-sim, axis=1)
        rank_pos = (np.argmax(order == pi[:, None], axis=1) + 1).astype(float)
        recip    = 1.0 / rank_pos

        out[f"{prefix}caption_sim_mean_{held}"]   = float(match.mean())
        out[f"{prefix}caption_sim_random_{held}"] = float(rand_sim.mean())
        out[f"{prefix}caption_sim_lift_{held}"]   = float(lift.mean())
        out[f"{prefix}mean_rank_{held}"]          = float(rank_pos.mean())
        out[f"{prefix}median_rank_{held}"]        = float(np.median(rank_pos))
        out[f"{prefix}mrr_{held}"]                = float(recip.mean())

        if B > 0:
            for vec, key in ((match,    f"caption_sim_mean_{held}"),
                             (lift,     f"caption_sim_lift_{held}"),
                             (rank_pos, f"mean_rank_{held}"),
                             (recip,    f"mrr_{held}")):
                lo, hi = _bootstrap_mean_ci(vec, B, ci, rng)
                out[f"{prefix}{key}_lo"] = lo
                out[f"{prefix}{key}_hi"] = hi

        for k in ks:
            r_at_k = (rank_pos <= k).astype(float)
            out[f"{prefix}recall@{k}_{held}"] = float(r_at_k.mean())
            if B > 0:
                lo, hi = _bootstrap_mean_ci(r_at_k, B, ci, rng)
                out[f"{prefix}recall@{k}_{held}_lo"] = lo
                out[f"{prefix}recall@{k}_{held}_hi"] = hi
            # Soft recall: T-weighted top-k membership in the witness ranking.
            # Only meaningful for FGW (baseline has no transport plan).
            if T is not None:
                top_k_idx = order[:, :k]
                mask = np.zeros_like(sim, dtype=bool)
                rows = np.arange(n_i)[:, None]
                mask[rows, top_k_idx] = True
                out[f"{prefix}soft_recall@{k}_{held}"] = float((T * mask).sum())
    return out


# ─── per-combo evaluation ───────────────────────────────────────────────────

def _evaluate_combo(combo, plan_dir, emb_root, splits,
                    n_triplets, ks, rng, B, ci, null_K, with_baseline):
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
    bridge = combo["text_encoder"]

    out = {**combo, **transport}
    out.update(_structural(C_i, C_a, pi, B, ci, rng))
    out.update(_triplet_agreement(C_i, C_a, pi, n_triplets, B, ci, rng))
    out.update(_semantic(emb_root, splits, idx_i, idx_a, pi, bridge, ks,
                         B, ci, rng, T=T))
    # Structural null: depends only on (C_i, C_a), so compute once per combo
    # rather than once per π.
    out.update(_structural_null(C_i, C_a, null_K, ci, rng))

    if with_baseline:
        pi_b = _text_only_pi(emb_root, splits, idx_i, idx_a, bridge)
        out.update(_structural(C_i, C_a, pi_b, B, ci, rng, prefix="baseline_"))
        out.update(_triplet_agreement(C_i, C_a, pi_b, n_triplets, B, ci, rng,
                                      prefix="baseline_"))
        # No T for the baseline → soft_recall absent for baseline_*.
        out.update(_semantic(emb_root, splits, idx_i, idx_a, pi_b, bridge, ks,
                             B, ci, rng, prefix="baseline_", T=None))

    return out


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
                B=args.bootstrap_B, ci=args.bootstrap_ci, null_K=args.null_K,
                with_baseline=not args.no_baseline,
            ))
        except Exception as e:
            enriched.append({**combo, "eval_error": f"{type(e).__name__}: {e}"})

    out = {**bundle,
           "results": enriched,
           "eval": {"seed":         args.seed,
                    "ks":           args.ks,
                    "n_triplets":   args.n_triplets,
                    "bootstrap_B":  args.bootstrap_B,
                    "bootstrap_ci": args.bootstrap_ci,
                    "null_K":       args.null_K,
                    "baseline":     not args.no_baseline}}
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
    ap.add_argument("--bootstrap_B",  type=int,   default=1000,
                    help="Bootstrap iterations for CIs (0 to disable).")
    ap.add_argument("--bootstrap_ci", type=float, default=0.95,
                    help="Confidence level for bootstrap percentile intervals.")
    ap.add_argument("--null_K",       type=int,   default=200,
                    help="Random permutations for the structural null distribution "
                         "(null_pearson_*, null_spearman_*). 0 to disable.")
    ap.add_argument("--no_baseline",  action="store_true",
                    help="Skip the text-only retrieval baseline.")
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
