"""Visualizations for the FGW validation suite.

Reads:
    fgw_validation/data/results/*_eval.json    (per-alpha enriched results)
    fgw_validation/data/cka_*.json             (pairwise CKA matrices)
    fgw_validation/data/embeddings/...         (raw embeddings, for UMAP and
                                                cross-modal CKA-vs-perf)

Writes (default):
    fgw_validation/data/figures/
        alpha_<metric>.{png,pdf}
        leaderboard_<metric>__<bridge>.png
        transport_<metric>.png
        cka_<dataset>_<split>_<modality>_<kernel>.png
        umap_<dataset>_<split>_<modality>_<encoder>.png
        perf_vs_cka__text_<metric>.png
        perf_vs_cka__image_text_<metric>.png

Seven plot families:

1. Alpha-sweep curves       — recall@k, structural corr, transport entropy
                              vs α, faceted by bridge text encoder.
2. Leaderboard heatmaps     — image_enc × audio_enc cells coloured by
                              best-α metric, one per (metric, bridge).
3. Transport diagnostics    — boxplots of plan stats across combos, vs α.
4. CKA matrices             — heatmap per cka_*.json file.
5. UMAP scatters            — 2D embedding of each encoder's raw space.
6. Perf vs CKA              — recall@k (and other) vs CKA between bridge
                              and held-out encoders (text-text CKA), and
                              vs CKA(image_enc, text_bridge) on Flickr8k
                              (cross-modal CKA).
7. Geodesic ablation        — for the contrastive-hypersphere pair (CLIP
                              image, CLAP audio), compare cost conventions
                              (cos_cos vs cos_neg vs geo_cos) head-to-head
                              across α, faceted by bridge.

CLI:
    python -m fgw_validation.plots                       # all sections
    python -m fgw_validation.plots --skip umap           # exclude one
    python -m fgw_validation.plots --only alpha cka      # restrict to two
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from .cka import linear_cka, pairwise_cka


# ─── shared helpers ─────────────────────────────────────────────────────────

_HELDOUT_PREFIXES = (
    "recall@1_", "recall@5_", "recall@10_",
    "caption_sim_mean_", "caption_sim_random_", "caption_sim_lift_",
    "mean_rank_",
)

_K_VALUES = (1, 5, 10)

_SIZE_PER_ENCODER = {
    "clip":    "small",
    "dinov2":  "small",
    "clap":    "medium",
    "ast":     "medium",
    "roberta": "small",
    "t5":      "small",
}


def _load_eval(paths: list[Path]) -> list[dict]:
    """Flatten *_eval.json bundles into one record per combo."""
    rows: list[dict] = []
    for p in paths:
        with open(p) as f:
            bundle = json.load(f)
        alpha = bundle.get("alpha")
        for combo in bundle["results"]:
            row = dict(combo)
            row.setdefault("alpha", alpha)
            row["_source"] = str(p)
            rows.append(row)
    return rows


def _held_out_avg(rec: dict, prefix: str) -> float:
    """Average a metric across held-out encoders (keys matching `<prefix><enc>`)."""
    vals = [v for k, v in rec.items()
            if k.startswith(prefix) and isinstance(v, (int, float))
            and not isinstance(v, bool) and np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _aggregate_per_alpha(records: list[dict], metric_fn,
                         group_keys: tuple[str, ...]) -> dict:
    """{group_tuple: [(alpha, mean, se, n_seeds), ...] sorted by alpha}.

    `se` is the across-seed standard error (std/√n_seeds) when multiple
    records share the same (group, alpha) — i.e. multi-seed runs. With a
    single seed `se = 0` and `n_seeds = 1`. Plot callers can ignore the
    extra fields (slice with `pts[:][:2]`) for backward-compat or render
    error bars when `n_seeds > 1`.
    """
    by: dict[tuple, list[tuple[float, float]]] = defaultdict(list)
    for r in records:
        if "eval_error" in r:
            continue
        v = metric_fn(r)
        if not np.isfinite(v):
            continue
        key = tuple(r.get(k) for k in group_keys)
        by[key].append((float(r["alpha"]), v))

    out = {}
    for key, items in by.items():
        per_alpha: dict[float, list[float]] = defaultdict(list)
        for a, v in items:
            per_alpha[a].append(v)
        rows = []
        for a, vs in per_alpha.items():
            arr = np.asarray(vs, dtype=float)
            mean = float(arr.mean())
            se = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
            rows.append((a, mean, se, len(arr)))
        out[key] = sorted(rows)
    return out


def _seed_count(records: list[dict]) -> int:
    """Distinct seeds present in the records (for headline annotations)."""
    seeds = {r.get("seed") for r in records if r.get("seed") is not None}
    return len(seeds) if seeds else 1


def cross_grid_spearman(records: list[dict], x_key: str, y_key: str,
                        n_perm: int = 1000, rng=None) -> dict:
    """Spearman correlation between two metrics *across* the (combo × α) grid,
    with a permutation p-value (shuffle y_key labels n_perm times).

    Tests the prediction that combos with high structural fit also have high
    semantic fit — i.e. that the two metric families track the same underlying
    alignment quality. A near-zero correlation means structural and semantic
    are dissociated and FGW may be solving its own objective without producing
    a meaningful matching.
    """
    rng = rng or np.random.default_rng(0)
    xs, ys = [], []
    for r in records:
        if "eval_error" in r:
            continue
        x, y = r.get(x_key), r.get(y_key)
        if x is None or y is None:
            continue
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        xs.append(float(x)); ys.append(float(y))
    if len(xs) < 5:
        return {"rho": float("nan"), "p": float("nan"), "n": len(xs)}
    xs = np.asarray(xs); ys = np.asarray(ys)

    def _spearman(a, b):
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        ra -= ra.mean(); rb -= rb.mean()
        return float((ra * rb).sum() /
                     (np.sqrt((ra * ra).sum() * (rb * rb).sum()) + 1e-30))

    rho = _spearman(xs, ys)
    null = np.empty(n_perm)
    for k in range(n_perm):
        null[k] = _spearman(xs, rng.permutation(ys))
    # two-sided p-value
    p = float((np.abs(null) >= abs(rho)).mean())
    return {"rho": rho, "p": p, "n": int(len(xs)),
            "null_mean": float(null.mean()),
            "null_lo":   float(np.quantile(null, 0.025)),
            "null_hi":   float(np.quantile(null, 0.975))}


def _savefig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─── 1. alpha-sweep curves ──────────────────────────────────────────────────

def plot_alpha_sweeps(records: list[dict], out_dir: Path) -> None:
    """One figure per metric, faceted by bridge text encoder (4 panels).
    Each panel shows curves grouped by image_encoder × audio_encoder."""
    if not records:
        print("[plots] no records — skipping alpha sweeps")
        return

    # (metric_key, label, value_fn)
    metrics = []
    for k in _K_VALUES:
        metrics.append((f"recall_at_{k}_avg", f"recall@{k} (avg over held-out)",
                        lambda r, k=k: _held_out_avg(r, f"recall@{k}_")))
    metrics += [
        ("caption_sim_lift_avg", "caption_sim_lift (avg over held-out)",
         lambda r: _held_out_avg(r, "caption_sim_lift_")),
        ("pearson_dist_corr",  "Pearson dist correlation",
         lambda r: float(r.get("pearson_dist_corr", np.nan))),
        ("spearman_dist_corr", "Spearman dist correlation",
         lambda r: float(r.get("spearman_dist_corr", np.nan))),
        ("triplet_agreement",  "triplet agreement",
         lambda r: float(r.get("triplet_agreement", np.nan))),
        ("entropy_norm",       "transport entropy_norm",
         lambda r: float(r.get("entropy_norm", np.nan))),
        ("top1_mass",          "transport top1_mass",
         lambda r: float(r.get("top1_mass", np.nan))),
    ]

    bridges = sorted({r["text_encoder"] for r in records})
    image_encoders = sorted({r["image_encoder"] for r in records})
    audio_encoders = sorted({r["audio_encoder"] for r in records})
    cmap = plt.get_cmap("tab10")
    image_colors = {e: cmap(i) for i, e in enumerate(image_encoders)}
    line_styles = {e: ls for e, ls in zip(audio_encoders, ("-", "--", ":", "-."))}

    for slug, label, fn in metrics:
        agg = _aggregate_per_alpha(records, fn,
                                   ("text_encoder", "image_encoder", "audio_encoder"))
        if not agg:
            continue
        ncols = max(1, len(bridges))
        fig, axes = plt.subplots(1, ncols, figsize=(4.5 * ncols, 4.0),
                                 sharey=True, squeeze=False)
        axes = axes[0]
        for ax, bridge in zip(axes, bridges):
            for (te, ie, ae), pts in agg.items():
                if te != bridge or not pts:
                    continue
                xs, ys, ses, ns = zip(*pts)
                xs, ys, ses = np.asarray(xs), np.asarray(ys), np.asarray(ses)
                ax.plot(xs, ys,
                        color=image_colors[ie],
                        linestyle=line_styles.get(ae, "-"),
                        marker="o", markersize=3, linewidth=1.2,
                        label=f"{ie}×{ae}")
                # Across-seed SE band when multi-seed.
                if max(ns) > 1:
                    ax.fill_between(xs, ys - ses, ys + ses,
                                    color=image_colors[ie], alpha=0.15, linewidth=0)
            ax.set_title(f"bridge: {bridge}")
            ax.set_xlabel("α")
            ax.grid(True, alpha=0.3)
        axes[0].set_ylabel(label)
        # one legend, dedup
        handles, labels = axes[0].get_legend_handles_labels()
        seen, h2, l2 = set(), [], []
        for h, l in zip(handles, labels):
            if l in seen:
                continue
            seen.add(l)
            h2.append(h)
            l2.append(l)
        if h2:
            fig.legend(h2, l2, loc="upper center", ncol=min(len(l2), 6),
                       bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=8)
        fig.suptitle(label)
        _savefig(fig, out_dir / f"alpha_{slug}.png")


# ─── 2. leaderboard heatmaps ────────────────────────────────────────────────

def plot_leaderboard(records: list[dict], out_dir: Path) -> None:
    """For each (metric, bridge), grid of image_enc × audio_enc cells coloured
    by the best-over-α value (cost/agg averaged out)."""
    if not records:
        return
    metrics = [
        ("recall_at_5_avg", "recall@5 (best α)", lambda r: _held_out_avg(r, "recall@5_"), "max"),
        ("recall_at_1_avg", "recall@1 (best α)", lambda r: _held_out_avg(r, "recall@1_"), "max"),
        ("pearson_dist_corr", "Pearson dist corr (best α)",
         lambda r: float(r.get("pearson_dist_corr", np.nan)), "max"),
        ("spearman_dist_corr", "Spearman dist corr (best α)",
         lambda r: float(r.get("spearman_dist_corr", np.nan)), "max"),
        ("triplet_agreement", "triplet agreement (best α)",
         lambda r: float(r.get("triplet_agreement", np.nan)), "max"),
    ]
    bridges = sorted({r["text_encoder"] for r in records})
    img_enc = sorted({r["image_encoder"] for r in records})
    aud_enc = sorted({r["audio_encoder"] for r in records})

    for slug, title, fn, agg in metrics:
        for bridge in bridges:
            mat = np.full((len(img_enc), len(aud_enc)), np.nan)
            for i, ie in enumerate(img_enc):
                for j, ae in enumerate(aud_enc):
                    vs = [fn(r) for r in records
                          if r.get("text_encoder") == bridge
                          and r.get("image_encoder") == ie
                          and r.get("audio_encoder") == ae
                          and "eval_error" not in r]
                    vs = [v for v in vs if np.isfinite(v)]
                    if not vs:
                        continue
                    mat[i, j] = max(vs) if agg == "max" else min(vs)
            if np.all(np.isnan(mat)):
                continue
            fig, ax = plt.subplots(figsize=(0.9 * max(3, len(aud_enc)) + 1,
                                            0.9 * max(3, len(img_enc)) + 1))
            im = ax.imshow(mat, aspect="auto", cmap="viridis")
            ax.set_xticks(range(len(aud_enc)), aud_enc)
            ax.set_yticks(range(len(img_enc)), img_enc)
            ax.set_xlabel("audio encoder")
            ax.set_ylabel("image encoder")
            ax.set_title(f"{title}\nbridge={bridge}")
            for i in range(len(img_enc)):
                for j in range(len(aud_enc)):
                    if np.isfinite(mat[i, j]):
                        ax.text(j, i, f"{mat[i, j]:.3f}",
                                ha="center", va="center", color="white", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            _savefig(fig, out_dir / f"leaderboard_{slug}__{bridge}.png")


# ─── 3. transport diagnostics ───────────────────────────────────────────────

def plot_transport_diagnostics(records: list[dict], out_dir: Path) -> None:
    if not records:
        return
    metrics = [
        ("entropy_norm",     "transport entropy_norm"),
        ("top1_mass",        "transport top1_mass"),
        ("mutual_best_rate", "mutual best rate"),
        ("coverage",         "coverage"),
    ]
    alphas = sorted({float(r["alpha"]) for r in records if r.get("alpha") is not None})
    if not alphas:
        return
    for key, label in metrics:
        data = [[float(r[key]) for r in records
                 if float(r.get("alpha", np.nan)) == a
                 and key in r and np.isfinite(float(r[key]))
                 and "eval_error" not in r]
                for a in alphas]
        if not any(data):
            continue
        fig, ax = plt.subplots(figsize=(max(5, 0.9 * len(alphas)), 4))
        ax.boxplot(data, tick_labels=[f"{a:g}" for a in alphas],
                   showmeans=True, meanline=True)
        ax.set_xlabel("α")
        ax.set_ylabel(label)
        ax.set_title(f"{label} across all combos, by α")
        ax.grid(True, axis="y", alpha=0.3)
        _savefig(fig, out_dir / f"transport_{key}.png")


# ─── 4. CKA heatmaps ────────────────────────────────────────────────────────

_CKA_PATTERN = re.compile(
    r"cka_(?P<dataset>[^_]+)_(?P<split>[^_]+)_(?P<modality>[^_]+)_(?P<kernel>[^.]+)\.json$"
)


def plot_cka_heatmaps(cka_paths: list[Path], out_dir: Path) -> None:
    for p in cka_paths:
        with open(p) as f:
            d = json.load(f)
        names = d["names"]
        M = np.asarray(d["matrix"])
        fig, ax = plt.subplots(figsize=(0.7 * max(3, len(names)) + 1.5,
                                        0.7 * max(3, len(names)) + 1.0))
        im = ax.imshow(M, vmin=0.0, vmax=1.0, cmap="magma")
        ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
        ax.set_yticks(range(len(names)), names)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        color=("white" if M[i, j] < 0.6 else "black"), fontsize=8)
        title = (f"CKA — {d.get('dataset','?')}/{d.get('split','?')}"
                 f"/{d.get('modality','?')}  ({d.get('kernel','?')}, n={d.get('n','?')})")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        _savefig(fig, out_dir / f"{p.stem}.png")


# ─── 5. UMAP per encoder ────────────────────────────────────────────────────

_UMAP_TARGETS = (
    ("flickr8k", "test", "image"),
    ("flickr8k", "test", "text"),
    ("clotho", "development", "audio"),
    ("clotho", "development", "text"),
)


def _emb_path(emb_root: Path, dataset: str, split: str,
              encoder: str, modality: str) -> Path:
    return emb_root / dataset / split / f"{encoder}_{_SIZE_PER_ENCODER[encoder]}_{modality}.pt"


def plot_umaps(emb_root: Path, out_dir: Path, n: int, seed: int) -> None:
    try:
        import umap  # type: ignore
    except ModuleNotFoundError:
        print("[plots] umap-learn not installed — skipping UMAP plots")
        return

    rng = np.random.default_rng(seed)
    for dataset, split, modality in _UMAP_TARGETS:
        for enc, size in _SIZE_PER_ENCODER.items():
            path = emb_root / dataset / split / f"{enc}_{size}_{modality}.pt"
            if not path.exists():
                continue
            d = torch.load(str(path), weights_only=False)
            emb = d["emb"]
            if modality == "text":
                emb = emb.mean(dim=1)
            emb = emb.numpy()
            N = emb.shape[0]
            if N < 5:
                continue
            idx = (np.arange(N) if N <= n
                   else np.sort(rng.choice(N, size=n, replace=False)))
            X = emb[idx].astype(np.float32)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                reducer = umap.UMAP(n_neighbors=min(15, max(2, len(idx) - 1)),
                                    min_dist=0.1, metric="cosine",
                                    random_state=seed)
                Z = reducer.fit_transform(X)
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(Z[:, 0], Z[:, 1], s=6, alpha=0.6)
            ax.set_title(f"UMAP — {dataset}/{split}/{modality}\n{enc} (n={len(idx)})")
            ax.set_xticks([])
            ax.set_yticks([])
            _savefig(fig, out_dir / f"umap_{dataset}_{split}_{modality}_{enc}.png")


# ─── 6. perf vs CKA ─────────────────────────────────────────────────────────

def _load_cka_index(cka_paths: list[Path]) -> dict:
    """Returns {(dataset, split, modality, kernel): {(a,b): cka}}."""
    index: dict[tuple, dict[tuple[str, str], float]] = {}
    for p in cka_paths:
        m = _CKA_PATTERN.search(p.name)
        if not m:
            continue
        with open(p) as f:
            d = json.load(f)
        names = d["names"]
        M = np.asarray(d["matrix"])
        key = (m.group("dataset"), m.group("split"),
               m.group("modality"), m.group("kernel"))
        pairs = {}
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                pairs[(a, b)] = float(M[i, j])
        index[key] = pairs
    return index


def _crossmodal_cka_image_text(emb_root: Path, dataset: str, split: str,
                               n: int, seed: int) -> dict[tuple[str, str], float]:
    """For Flickr8k: CKA(image_enc image emb, text_enc mean-pooled text emb)
    on aligned ids. Returns {(image_enc, text_enc): cka_linear}."""
    out: dict[tuple[str, str], float] = {}
    image_encs = ("clip", "dinov2")
    text_encs = ("clip", "clap", "roberta", "t5")
    rng = np.random.default_rng(seed)
    base_ids = None
    cache: dict[tuple[str, str], np.ndarray] = {}
    for enc, mod in [(e, "image") for e in image_encs] + [(e, "text") for e in text_encs]:
        path = _emb_path(emb_root, dataset, split, enc, mod)
        if not path.exists():
            continue
        d = torch.load(str(path), weights_only=False)
        if base_ids is None:
            base_ids = d["ids"]
        if d["ids"] != base_ids:
            print(f"[plots] {dataset}/{split}/{enc}/{mod}: id ordering "
                  "differs from baseline — skipping cross-modal CKA")
            return {}
        e = d["emb"]
        if mod == "text":
            e = e.mean(dim=1)
        cache[(enc, mod)] = e.numpy()
    if base_ids is None:
        return {}
    N = len(base_ids)
    idx = (np.arange(N) if N <= n
           else np.sort(rng.choice(N, size=n, replace=False)))
    for ie in image_encs:
        if (ie, "image") not in cache:
            continue
        for te in text_encs:
            if (te, "text") not in cache:
                continue
            X = cache[(ie, "image")][idx]
            Y = cache[(te, "text")][idx]
            out[(ie, te)] = float(linear_cka(X, Y))
    return out


def plot_perf_vs_cka(records: list[dict], cka_paths: list[Path],
                     emb_root: Path, out_dir: Path,
                     crossmodal_n: int, crossmodal_seed: int) -> None:
    """Two scatter families:

    (a) text-text: x = CKA(bridge, held_out) on Flickr8k captions (linear);
        y = recall@k_<held>. One point per (combo, held-out).
    (b) image-text cross-modal: x = CKA(image_enc, text_bridge) on Flickr8k
        (image emb vs mean-pooled caption emb); y = pearson_dist_corr / recall@k.
    """
    if not records:
        return
    cka_index = _load_cka_index(cka_paths)

    # (a) text-text — prefer linear kernel on flickr8k/test, fallback to clotho/dev
    text_cka = (cka_index.get(("flickr8k", "test", "text", "linear"))
                or cka_index.get(("clotho", "development", "text", "linear"))
                or cka_index.get(("flickr8k", "test", "text", "rbf"))
                or cka_index.get(("clotho", "development", "text", "rbf")))
    if text_cka:
        for k in _K_VALUES:
            xs, ys, colors, markers, labels = [], [], [], [], []
            cmap = plt.get_cmap("tab10")
            ie_list = sorted({r["image_encoder"] for r in records})
            ie_color = {e: cmap(i) for i, e in enumerate(ie_list)}
            ae_list = sorted({r["audio_encoder"] for r in records})
            ae_marker = {e: m for e, m in zip(ae_list, ("o", "s", "^", "D"))}
            for r in records:
                if "eval_error" in r:
                    continue
                bridge = r["text_encoder"]
                for held in ("clip", "clap", "roberta", "t5"):
                    if held == bridge:
                        continue
                    rk = r.get(f"recall@{k}_{held}")
                    if rk is None or not np.isfinite(rk):
                        continue
                    cka_v = text_cka.get((bridge, held))
                    if cka_v is None or not np.isfinite(cka_v):
                        continue
                    xs.append(cka_v)
                    ys.append(float(rk))
                    colors.append(ie_color[r["image_encoder"]])
                    markers.append(ae_marker.get(r["audio_encoder"], "o"))
                    labels.append((r["image_encoder"], r["audio_encoder"]))
            if not xs:
                continue
            fig, ax = plt.subplots(figsize=(6, 5))
            for x, y, c, m in zip(xs, ys, colors, markers):
                ax.scatter(x, y, c=[c], marker=m, alpha=0.6, s=28,
                           edgecolors="none")
            xs_a, ys_a = np.asarray(xs), np.asarray(ys)
            if len(xs_a) >= 3:
                rho = np.corrcoef(xs_a, ys_a)[0, 1]
                ax.set_title(f"recall@{k} vs text-text CKA(bridge, held-out)\n"
                             f"Pearson ρ = {rho:.3f}, n={len(xs_a)}")
            else:
                ax.set_title(f"recall@{k} vs text-text CKA")
            ax.set_xlabel("CKA(text bridge, held-out text)")
            ax.set_ylabel(f"recall@{k} (held-out)")
            ax.grid(True, alpha=0.3)
            # legends
            color_handles = [plt.Line2D([0], [0], marker="o", linestyle="",
                                        color=ie_color[e], label=e)
                             for e in ie_list]
            mark_handles = [plt.Line2D([0], [0], marker=ae_marker[e], linestyle="",
                                       color="gray", label=e)
                            for e in ae_list]
            ax.legend(handles=color_handles + mark_handles,
                      title="image (color) / audio (marker)",
                      fontsize=7, loc="best")
            _savefig(fig, out_dir / f"perf_vs_cka__text_recall_at_{k}.png")

    # (b) image-text cross-modal — compute on Flickr8k test
    cm = _crossmodal_cka_image_text(emb_root, "flickr8k", "test",
                                    n=crossmodal_n, seed=crossmodal_seed)
    if not cm:
        print("[plots] cross-modal CKA could not be computed — skipping")
        return
    targets = (
        ("recall@5_avg",       lambda r: _held_out_avg(r, "recall@5_")),
        ("pearson_dist_corr",  lambda r: float(r.get("pearson_dist_corr", np.nan))),
        ("triplet_agreement",  lambda r: float(r.get("triplet_agreement", np.nan))),
    )
    cmap = plt.get_cmap("tab10")
    ae_list = sorted({r["audio_encoder"] for r in records})
    ae_color = {e: cmap(i) for i, e in enumerate(ae_list)}
    bridges_present = sorted({r["text_encoder"] for r in records})
    for slug, fn in targets:
        xs, ys, colors, markers = [], [], [], []
        marker_for_bridge = {b: m for b, m in zip(bridges_present,
                                                  ("o", "s", "^", "D"))}
        for r in records:
            if "eval_error" in r:
                continue
            ie = r["image_encoder"]
            te = r["text_encoder"]
            cka_v = cm.get((ie, te))
            v = fn(r)
            if cka_v is None or not np.isfinite(cka_v) or not np.isfinite(v):
                continue
            xs.append(cka_v)
            ys.append(v)
            colors.append(ae_color[r["audio_encoder"]])
            markers.append(marker_for_bridge.get(te, "o"))
        if not xs:
            continue
        fig, ax = plt.subplots(figsize=(6, 5))
        for x, y, c, m in zip(xs, ys, colors, markers):
            ax.scatter(x, y, c=[c], marker=m, alpha=0.6, s=28, edgecolors="none")
        xs_a, ys_a = np.asarray(xs), np.asarray(ys)
        if len(xs_a) >= 3:
            rho = np.corrcoef(xs_a, ys_a)[0, 1]
            ax.set_title(f"{slug} vs CKA(image_enc, text_bridge) — Flickr8k/test\n"
                         f"Pearson ρ = {rho:.3f}, n={len(xs_a)}")
        else:
            ax.set_title(f"{slug} vs cross-modal CKA")
        ax.set_xlabel("CKA(image_enc image, text_bridge text)  [linear, Flickr8k]")
        ax.set_ylabel(slug)
        ax.grid(True, alpha=0.3)
        color_handles = [plt.Line2D([0], [0], marker="o", linestyle="",
                                    color=ae_color[e], label=e)
                         for e in ae_list]
        mark_handles = [plt.Line2D([0], [0], marker=marker_for_bridge[b],
                                   linestyle="", color="gray", label=b)
                        for b in bridges_present]
        ax.legend(handles=color_handles + mark_handles,
                  title="audio (color) / bridge (marker)",
                  fontsize=7, loc="best")
        _savefig(fig, out_dir / f"perf_vs_cka__image_text_{slug}.png")


# ─── 7. geodesic ablation (CLIP+CLAP only) ──────────────────────────────────

_HYPERSPHERICAL_PAIR = ("clip", "clap")  # (image_encoder, audio_encoder)


def plot_geodesic_ablation(records: list[dict], out_dir: Path) -> None:
    """For the contrastive-hypersphere pair (CLIP image, CLAP audio):
    one figure per metric, panels per bridge, lines per cost_convention.
    Surfaces whether `geo_cos` actually moves the needle vs `cos_cos`/`cos_neg`."""
    img_enc, aud_enc = _HYPERSPHERICAL_PAIR
    pair = [r for r in records
            if r.get("image_encoder") == img_enc
            and r.get("audio_encoder") == aud_enc
            and "eval_error" not in r]
    if not pair:
        print("[plots] no records for (CLIP, CLAP) — skipping geodesic ablation")
        return
    conventions = sorted({r.get("cost_convention") for r in pair if r.get("cost_convention")})
    if "geo_cos" not in conventions:
        print("[plots] no geo_cos data — skipping geodesic ablation "
              "(re-run experiments with the updated cost_convention set)")
        return

    metrics = []
    for k in _K_VALUES:
        metrics.append((f"recall_at_{k}_avg", f"recall@{k} (avg over held-out)",
                        lambda r, k=k: _held_out_avg(r, f"recall@{k}_")))
    metrics += [
        ("pearson_dist_corr",  "Pearson dist correlation",
         lambda r: float(r.get("pearson_dist_corr", np.nan))),
        ("spearman_dist_corr", "Spearman dist correlation",
         lambda r: float(r.get("spearman_dist_corr", np.nan))),
        ("triplet_agreement",  "triplet agreement",
         lambda r: float(r.get("triplet_agreement", np.nan))),
        ("entropy_norm",       "transport entropy_norm",
         lambda r: float(r.get("entropy_norm", np.nan))),
        ("top1_mass",          "transport top1_mass",
         lambda r: float(r.get("top1_mass", np.nan))),
    ]

    bridges = sorted({r["text_encoder"] for r in pair})
    cmap = plt.get_cmap("tab10")
    cc_color = {cc: cmap(i) for i, cc in enumerate(conventions)}

    for slug, label, fn in metrics:
        agg = _aggregate_per_alpha(pair, fn,
                                   ("text_encoder", "cost_convention"))
        if not agg:
            continue
        ncols = max(1, len(bridges))
        fig, axes = plt.subplots(1, ncols, figsize=(4.5 * ncols, 4.0),
                                 sharey=True, squeeze=False)
        axes = axes[0]
        for ax, bridge in zip(axes, bridges):
            for (te, cc), pts in agg.items():
                if te != bridge or not pts:
                    continue
                xs, ys, ses, ns = zip(*pts)
                xs, ys, ses = np.asarray(xs), np.asarray(ys), np.asarray(ses)
                ax.plot(xs, ys, color=cc_color[cc],
                        marker="o", markersize=4, linewidth=1.6, label=cc)
                if max(ns) > 1:
                    ax.fill_between(xs, ys - ses, ys + ses,
                                    color=cc_color[cc], alpha=0.18, linewidth=0)
            ax.set_title(f"bridge: {bridge}")
            ax.set_xlabel("α")
            ax.grid(True, alpha=0.3)
        axes[0].set_ylabel(label)
        handles, labels = axes[0].get_legend_handles_labels()
        seen, h2, l2 = set(), [], []
        for h, l in zip(handles, labels):
            if l in seen:
                continue
            seen.add(l)
            h2.append(h)
            l2.append(l)
        if h2:
            fig.legend(h2, l2, loc="upper center", ncol=len(l2),
                       bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=9)
        fig.suptitle(f"{label}  —  CLIP image × CLAP audio")
        _savefig(fig, out_dir / f"geodesic_ablation_{slug}.png")


# ─── 8. struct vs semantic correlation across the grid ────────────────────

# Pre-registered primary configuration (see EXPERIMENT.md §4.1).
PRIMARY_CONFIG = {
    "image_encoder":   "clip",
    "audio_encoder":   "clap",
    "text_encoder":    "clip",
    "cost_convention": "cos_cos",
    "caption_agg":     "mean",
}


def plot_struct_vs_semantic(records: list[dict], out_dir: Path,
                            primary_recall_witness: str = "recall@1_clap"
                            ) -> None:
    """Structural ↔ semantic correlation across the grid, at *fixed α*.

    α directly trades off GW vs W, so pooling across α produces an
    anti-correlation that's an artifact of the FGW formulation rather than a
    test of alignment quality. The well-posed question — "do good encoder
    choices win on both metric families?" — is only meaningful at fixed α.

    α* is chosen as the α that maximises the primary-config witness on the
    primary configuration (default: clip×clap×clip×cos_cos×mean,
    `recall@1_clap`). For completeness we also compute the all-α version
    (informative about tradeoff steepness, not alignment quality).
    """
    if not records:
        print("[plots] no records — skipping struct-vs-semantic diagnostic")
        return

    # Pick α* on the primary configuration: the α that maximises the chosen
    # witness, averaged across seeds for that combo.
    pri = [r for r in records
           if "eval_error" not in r
           and all(r.get(k) == v for k, v in PRIMARY_CONFIG.items())]
    if pri:
        per_alpha: dict[float, list[float]] = defaultdict(list)
        for r in pri:
            v = r.get(primary_recall_witness)
            if v is not None and np.isfinite(v):
                per_alpha[float(r["alpha"])].append(float(v))
        if per_alpha:
            alpha_star = max(per_alpha, key=lambda a: float(np.mean(per_alpha[a])))
        else:
            alpha_star = None
    else:
        alpha_star = None

    struct = [
        ("pearson_dist_corr",  "Pearson dist correlation"),
        ("spearman_dist_corr", "Spearman dist correlation"),
        ("triplet_agreement",  "triplet agreement"),
    ]
    sem = [
        ("recall@1_lex",   "recall@1 (lex, encoder-free)"),
        ("recall@1_t5",    "recall@1 (t5)"),
        ("mrr_lex",        "MRR (lex)"),
    ]

    rng = np.random.default_rng(0)
    summary: list[dict] = []

    for slice_label, slice_records in (
        ("alpha*", [r for r in records if alpha_star is not None
                    and abs(float(r.get("alpha", -999)) - alpha_star) < 1e-9]),
        ("all_alpha", records),
    ):
        if not slice_records:
            continue
        n_rows, n_cols = len(struct), len(sem)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.6 * n_rows),
                                 squeeze=False)
        for i, (xk, xlab) in enumerate(struct):
            for j, (yk, ylab) in enumerate(sem):
                ax = axes[i, j]
                xs, ys = [], []
                for r in slice_records:
                    if "eval_error" in r: continue
                    x, y = r.get(xk), r.get(yk)
                    if x is None or y is None: continue
                    if not (np.isfinite(x) and np.isfinite(y)): continue
                    xs.append(x); ys.append(y)
                if len(xs) < 5:
                    ax.text(0.5, 0.5, "insufficient data",
                            ha="center", va="center", transform=ax.transAxes)
                    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
                    continue
                stats = cross_grid_spearman(slice_records, xk, yk, n_perm=1000, rng=rng)
                ax.scatter(xs, ys, s=10, alpha=0.45)
                ax.set_xlabel(xlab); ax.set_ylabel(ylab)
                ax.set_title(f"ρ_S = {stats['rho']:+.3f}   p = {stats['p']:.3f}   n = {stats['n']}",
                             fontsize=10)
                ax.grid(True, alpha=0.3)
                summary.append({"slice": slice_label, "alpha_star": alpha_star,
                                "x": xk, "y": yk, **stats})
        suffix = (f"α = α* = {alpha_star}" if slice_label == "alpha*" and alpha_star is not None
                  else "all α (tradeoff dominated, not the H_FGW test)")
        fig.suptitle(f"Structural ↔ semantic correlation — {suffix}", y=1.00)
        _savefig(fig, out_dir / f"struct_vs_semantic_{slice_label}.png")

    out_path = out_dir / "struct_vs_semantic.json"
    with open(out_path, "w") as f:
        json.dump({"alpha_star": alpha_star, "results": summary}, f, indent=2)
    print(f"[plots] struct↔semantic stats → {out_path} (α* = {alpha_star})")


# ─── 9. primary-configuration spotlight ───────────────────────────────────

def plot_primary_config(records: list[dict], out_dir: Path) -> None:
    """Headline figure: for the §4.1 pre-registered primary configuration,
    plot FGW vs baseline vs random-π null, across α, with across-seed SE
    bands when available. This is the figure that addresses **H_FGW**
    directly without leaning on the exploratory leaderboard.
    """
    pri = [r for r in records
           if "eval_error" not in r
           and all(r.get(k) == v for k, v in PRIMARY_CONFIG.items())]
    if not pri:
        print(f"[plots] no records match primary config {PRIMARY_CONFIG} — skipping")
        return

    metrics = [
        ("recall@1_lex",       "recall@1 (lex, encoder-free witness)"),
        ("recall@1_t5",        "recall@1 (t5 witness — pre-registered)"),
        ("recall@1_clap",      "recall@1 (clap witness — selection)"),
        ("pearson_dist_corr",  "Pearson dist correlation"),
        ("triplet_agreement",  "triplet agreement"),
        ("entropy_norm",       "transport entropy_norm"),
    ]
    n = len(metrics)
    cols = 3; rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 3.5 * rows),
                             squeeze=False)
    for i, (mk, lab) in enumerate(metrics):
        ax = axes[i // cols, i % cols]
        # FGW
        agg = _aggregate_per_alpha(pri,
                                   lambda r, mk=mk: float(r.get(mk, np.nan)),
                                   ("image_encoder",))
        for _, pts in agg.items():
            xs, ys, ses, ns = zip(*pts)
            xs, ys, ses = np.asarray(xs), np.asarray(ys), np.asarray(ses)
            ax.plot(xs, ys, "o-", color="C0", linewidth=1.6, label="FGW", markersize=4)
            if max(ns) > 1:
                ax.fill_between(xs, ys - ses, ys + ses, color="C0", alpha=0.2)
        # Baseline
        bk = f"baseline_{mk}"
        agg_b = _aggregate_per_alpha(pri,
                                     lambda r, bk=bk: float(r.get(bk, np.nan)),
                                     ("image_encoder",))
        for _, pts in agg_b.items():
            xs, ys, ses, ns = zip(*pts)
            xs, ys, ses = np.asarray(xs), np.asarray(ys), np.asarray(ses)
            ax.plot(xs, ys, "s--", color="C1", linewidth=1.4,
                    label="baseline (text-only)", markersize=4)
            if max(ns) > 1:
                ax.fill_between(xs, ys - ses, ys + ses, color="C1", alpha=0.2)
        # Null reference for structural metrics
        if mk in ("pearson_dist_corr", "spearman_dist_corr"):
            null_key = mk.replace("_dist_corr", "")
            null_key = f"null_{null_key}_mean"
            agg_n = _aggregate_per_alpha(pri,
                                         lambda r, nk=null_key: float(r.get(nk, np.nan)),
                                         ("image_encoder",))
            for _, pts in agg_n.items():
                xs, ys, _, _ = zip(*pts)
                ax.plot(xs, ys, ":", color="gray", linewidth=1.2,
                        label="random-π null")
        ax.set_xlabel("α")
        ax.set_ylabel(lab)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8, loc="best")
    cfg = " × ".join(f"{k}={v}" for k, v in PRIMARY_CONFIG.items())
    seeds = _seed_count(pri)
    fig.suptitle(f"Primary configuration spotlight  —  {cfg}  —  {seeds} seed(s)",
                 fontsize=11)
    _savefig(fig, out_dir / "primary_config.png")


# ─── main ───────────────────────────────────────────────────────────────────

_SECTIONS = ("alpha", "leaderboard", "transport", "cka", "umap", "perfcka",
             "geo", "diag", "primary")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_glob",
                    default="fgw_validation/data/results/fgw_*_eval.json",
                    help="Shell glob for *_eval.json bundles.")
    ap.add_argument("--cka_glob",
                    default="fgw_validation/data/cka_*.json",
                    help="Shell glob for cka_*.json files.")
    ap.add_argument("--emb_root", default="fgw_validation/data/embeddings")
    ap.add_argument("--out_dir",  default="fgw_validation/data/figures")
    ap.add_argument("--umap_n",   type=int, default=1500)
    ap.add_argument("--seed",     type=int, default=0)
    ap.add_argument("--only",     nargs="+", choices=_SECTIONS, default=None)
    ap.add_argument("--skip",     nargs="+", choices=_SECTIONS, default=[])
    args = ap.parse_args()

    sections = set(args.only) if args.only else set(_SECTIONS)
    sections -= set(args.skip)

    out_dir = Path(args.out_dir)
    eval_paths = [Path(p) for p in sorted(glob.glob(args.results_glob))]
    cka_paths  = [Path(p) for p in sorted(glob.glob(args.cka_glob))]

    print(f"[plots] eval files: {len(eval_paths)}; cka files: {len(cka_paths)}")
    print(f"[plots] sections:   {sorted(sections)}")
    print(f"[plots] output dir: {out_dir}")

    records = _load_eval(eval_paths) if eval_paths else []

    if "alpha" in sections:
        plot_alpha_sweeps(records, out_dir)
    if "leaderboard" in sections:
        plot_leaderboard(records, out_dir)
    if "transport" in sections:
        plot_transport_diagnostics(records, out_dir)
    if "cka" in sections and cka_paths:
        plot_cka_heatmaps(cka_paths, out_dir)
    if "umap" in sections:
        plot_umaps(Path(args.emb_root), out_dir, n=args.umap_n, seed=args.seed)
    if "perfcka" in sections:
        plot_perf_vs_cka(records, cka_paths, Path(args.emb_root), out_dir,
                         crossmodal_n=args.umap_n, crossmodal_seed=args.seed)
    if "geo" in sections:
        plot_geodesic_ablation(records, out_dir)
    if "diag" in sections:
        plot_struct_vs_semantic(records, out_dir)
    if "primary" in sections:
        plot_primary_config(records, out_dir)
    print(f"[plots] done — figures in {out_dir}")


if __name__ == "__main__":
    main()
