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
    """{group_tuple: [(alpha, mean_value), ...] sorted by alpha}."""
    by: dict[tuple, list[tuple[float, float]]] = defaultdict(list)
    for r in records:
        if "eval_error" in r:
            continue
        v = metric_fn(r)
        if not np.isfinite(v):
            continue
        key = tuple(r.get(k) for k in group_keys)
        by[key].append((float(r["alpha"]), v))
    # average within each (group, alpha)
    out = {}
    for key, items in by.items():
        per_alpha: dict[float, list[float]] = defaultdict(list)
        for a, v in items:
            per_alpha[a].append(v)
        out[key] = sorted((a, float(np.mean(vs))) for a, vs in per_alpha.items())
    return out


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
                xs, ys = zip(*pts)
                ax.plot(xs, ys,
                        color=image_colors[ie],
                        linestyle=line_styles.get(ae, "-"),
                        marker="o", markersize=3, linewidth=1.2,
                        label=f"{ie}×{ae}")
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
                xs, ys = zip(*pts)
                ax.plot(xs, ys, color=cc_color[cc],
                        marker="o", markersize=4, linewidth=1.6, label=cc)
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


# ─── main ───────────────────────────────────────────────────────────────────

_SECTIONS = ("alpha", "leaderboard", "transport", "cka", "umap", "perfcka", "geo")


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
    print(f"[plots] done — figures in {out_dir}")


if __name__ == "__main__":
    main()
