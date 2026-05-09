"""Pre-experiment diagnostic: how much semantic overlap exists between
Flickr8k and Clotho under one witness encoder, *before* any FGW?

For each Flickr8k image, compute

    s_i = max_j  cos(witness_text(img_caps_i),  witness_text(aud_caps_j))

and report the distribution of `{s_i}_i`. Low values mean most images have
no plausible audio counterpart in Clotho — any cross-modal alignment method
is then operating in a low-signal regime. The eval pipeline's recall metrics
are bounded above by what this distribution can support: if the median of
`{s_i}` is ~0, recall@1 cannot meaningfully exceed chance regardless of FGW.

The lexical witness (`--witness lex`) is encoder-free and shares no training
data with any FGW component — useful as a robustness check.

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


def _plot(s: np.ndarray, witness: str, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.hist(s, bins=40, edgecolor="black", alpha=0.85)
    ax.set_xlabel(f"max_j  sim(img_caps_i, aud_caps_j)  under '{witness}'")
    ax.set_ylabel("# images")
    ax.set_title(f"Dataset overlap diagnostic — witness = {witness}")
    for q, label in [(0.5, "median"), (0.25, "q25"), (0.95, "q95")]:
        ax.axvline(np.quantile(s, q), linestyle="--", linewidth=1, label=f"{label}={np.quantile(s, q):.3f}")
    ax.legend()
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
    summary["witness"]     = args.witness
    summary["splits"]      = splits
    summary["seed"]        = args.seed
    summary["subsampled"]  = bool(args.n)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path  = out_dir / f"dataset_overlap_{args.witness}.png"
    json_path = out_dir / f"dataset_overlap_{args.witness}.json"

    _plot(sim.max(axis=1), args.witness, fig_path)
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    s = summary["max_per_row"]
    print(f"[{args.witness}]  n_img={summary['n_images']}  n_aud={summary['n_audios']}")
    print(f"  median max-sim = {s['median']:.4f}   "
          f"q25 = {s['q']['0.25']:.4f}   q75 = {s['q']['0.75']:.4f}")
    print(f"  → {fig_path}")
    print(f"  → {json_path}")


if __name__ == "__main__":
    main()
