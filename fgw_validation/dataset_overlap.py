"""Pre-experiment diagnostic: how much semantic overlap exists between
Flickr8k and Clotho under one witness encoder, *before* any FGW?

Two complementary diagnostics:

    (1) Per-image best-match similarity:
            s_i = max_j  sim(witness(img_caps_i), witness(aud_caps_j))
        Low median → most images have no plausible audio counterpart;
        every method is then operating in a low-signal regime.

    (2) Coverage asymmetry — how concentrated are the row-wise argmaxes?
        Few "hub" audios capturing most argmaxes means balanced FGW
        (uniform marginals → permutation T → 1-to-1 matching) pays a
        structural penalty regardless of encoder quality, while row-wise
        greedy retrieval is free to pile multiple images on each hub.
        Reports fraction of unused audios, max-hub-fraction, top-10-hub
        fraction, and the Gini coefficient of the per-audio argmax count.

The lexical witness (`--witness lex`) is encoder-free and shares no training
data with any FGW component — most informative robustness check.

Outputs:
    <out_dir>/dataset_overlap_<witness>.png   histogram + summary
    <out_dir>/dataset_overlap_<witness>.json  numeric summary

Usage:
    python -m fgw_validation.dataset_overlap --witness clip
    python -m fgw_validation.dataset_overlap --witness lex --n 1000
"""
import argparse
import json
from pathlib import Path

import numpy as np

from fgw_validation.eval import (
    _l2norm, _load_text, _lexical_sim, _emb_path,
)


def _max_per_row(sim: np.ndarray, q_levels=(0.05, 0.25, 0.5, 0.75, 0.95)) -> dict:
    s = sim.max(axis=1)
    return {
        "n_images":      int(sim.shape[0]),
        "n_audios":      int(sim.shape[1]),
        "max_per_row":   {
            "mean":   float(s.mean()),
            "std":    float(s.std()),
            "min":    float(s.min()),
            "max":    float(s.max()),
            "median": float(np.median(s)),
            "q":      {f"{q:.2f}": float(np.quantile(s, q)) for q in q_levels},
        },
    }


def _coverage_asymmetry(sim: np.ndarray) -> dict:
    """How concentrated is `argmax_j sim[i, j]` across audios?

    Balanced FGW (uniform marginals) forbids column collisions; row-wise
    greedy retrieval doesn't. If a few audio "hubs" are the argmax for many
    images, balanced FGW pays a structural penalty regardless of encoder
    quality. Reports per-audio hub-count distribution + Gini coefficient.
    """
    n_i, n_a = sim.shape
    counts = np.bincount(sim.argmax(axis=1), minlength=n_a)
    nz = counts[counts > 0]
    sorted_counts = np.sort(counts.astype(float))
    cum = np.cumsum(sorted_counts)
    gini = float((n_a + 1 - 2 * cum.sum() / cum[-1]) / n_a) if cum[-1] > 0 else 0.0
    top_k = min(10, n_a)
    return {
        "fraction_audios_used":    float((counts > 0).mean()),
        "fraction_audios_unused":  float((counts == 0).mean()),
        "max_hub_fraction":        float(counts.max() / n_i),
        "top10_hub_fraction":      float(np.sort(counts)[-top_k:].sum() / n_i),
        "gini":                    gini,
        "hub_count_distribution":  counts.tolist(),
    }


def _plot(s: np.ndarray, hub_counts: np.ndarray, witness: str, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    ax = axes[0]
    ax.hist(s, bins=40, edgecolor="black", alpha=0.85)
    ax.set_xlabel(f"max_j  sim(img_caps_i, aud_caps_j)  under '{witness}'")
    ax.set_ylabel("# images")
    ax.set_title("Per-image best-match similarity")
    for q, label in [(0.5, "median"), (0.25, "q25"), (0.95, "q95")]:
        ax.axvline(np.quantile(s, q), linestyle="--", linewidth=1,
                   label=f"{label}={np.quantile(s, q):.3f}")
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.hist(hub_counts, bins=max(20, int(hub_counts.max()) + 1),
            edgecolor="black", alpha=0.85)
    n_a = len(hub_counts); n_i = int(hub_counts.sum())
    ax.set_xlabel("# images for which audio j is argmax")
    ax.set_ylabel("# audios")
    unused = float((hub_counts == 0).mean())
    ax.set_title(f"Coverage asymmetry  (unused audios: {unused:.0%}, "
                 f"max hub: {hub_counts.max()}/{n_i})")
    fig.suptitle(f"Dataset overlap diagnostic — witness = {witness}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--witness", default="clip",
                    choices=["clip", "clap", "roberta", "t5", "lex"],
                    help="Witness encoder (or 'lex' for encoder-free Jaccard).")
    ap.add_argument("--emb_root",     default="fgw_validation/data/embeddings")
    ap.add_argument("--flickr_split", default="test")
    ap.add_argument("--clotho_split", default="development")
    ap.add_argument("--n",            type=int, default=0,
                    help="Subsample n items from each side (0 = all).")
    ap.add_argument("--seed",         type=int, default=0)
    ap.add_argument("--out_dir",      default="fgw_validation/data/figures")
    args = ap.parse_args()

    emb_root = Path(args.emb_root)
    splits = {"flickr8k": args.flickr_split, "clotho": args.clotho_split}
    rng = np.random.default_rng(args.seed)

    # Determine N_i, N_a from any text embedding file we can find.
    N_i = N_a = None
    for enc in ("clip", "clap", "roberta", "t5"):
        try:
            di = __import__("torch").load(
                str(_emb_path(emb_root, "flickr8k", splits["flickr8k"], enc, "text")),
                weights_only=False,
            )
            da = __import__("torch").load(
                str(_emb_path(emb_root, "clotho", splits["clotho"], enc, "text")),
                weights_only=False,
            )
            N_i = di["emb"].shape[0]
            N_a = da["emb"].shape[0]
            break
        except FileNotFoundError:
            continue
    if N_i is None:
        raise SystemExit("could not locate any text embedding files; run encode.py first")

    if args.n and args.n < min(N_i, N_a):
        idx_i = rng.choice(N_i, size=args.n, replace=False)
        idx_a = rng.choice(N_a, size=args.n, replace=False)
    else:
        idx_i = np.arange(N_i)
        idx_a = np.arange(N_a)

    if args.witness == "lex":
        sim = _lexical_sim(emb_root, splits, idx_i, idx_a)
        if sim is None:
            raise SystemExit("lexical witness unavailable — raw caption files missing")
    else:
        ti = _load_text(emb_root, "flickr8k", splits["flickr8k"], args.witness, idx_i)
        ta = _load_text(emb_root, "clotho",   splits["clotho"],   args.witness, idx_a)
        sim = _l2norm(ti) @ _l2norm(ta).T

    summary = _max_per_row(sim)
    summary["coverage"]    = _coverage_asymmetry(sim)
    summary["witness"]     = args.witness
    summary["splits"]      = splits
    summary["seed"]        = args.seed
    summary["subsampled"]  = bool(args.n)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path  = out_dir / f"dataset_overlap_{args.witness}.png"
    json_path = out_dir / f"dataset_overlap_{args.witness}.json"

    hub_counts = np.asarray(summary["coverage"]["hub_count_distribution"])
    _plot(sim.max(axis=1), hub_counts, args.witness, fig_path)
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    s = summary["max_per_row"]; c = summary["coverage"]
    print(f"[{args.witness}]  n_img={summary['n_images']}  n_aud={summary['n_audios']}")
    print(f"  best-match sim:  median={s['median']:.4f}  q25={s['q']['0.25']:.4f}  q75={s['q']['0.75']:.4f}")
    print(f"  coverage asym :  audios_used={c['fraction_audios_used']:.1%}  "
          f"max_hub={c['max_hub_fraction']:.1%}  top10_hubs={c['top10_hub_fraction']:.1%}  "
          f"gini={c['gini']:.3f}")
    print(f"  → {fig_path}")
    print(f"  → {json_path}")


if __name__ == "__main__":
    main()
