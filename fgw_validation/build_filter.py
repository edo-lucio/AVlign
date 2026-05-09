"""Pre-filter Flickr8k × Clotho to a mutual-NN subset.

For each Flickr8k image *i* and Clotho audio *j*, keep the pair only when *i*
is *j*'s argmax under a witness encoder *and* *j* is *i*'s argmax. The output
is a list of (i, j) index pairs that defines a symmetric-coverage subset
where balanced FGW (uniform-marginal LP-assignment) has a fair chance —
unlike the unfiltered datasets where ~74 % of Clotho audios are nobody's
argmax.

To avoid handing FGW the answer, **the filter witness must differ from any
encoder used in FGW evaluation**. Recommended: filter with `t5` (text-only,
no contrastive multimodal training); evaluate with `clap`/`clip`/`lex`.

Usage:
    python -m fgw_validation.build_filter --witness t5 \\
        --out fgw_validation/data/filters/t5_mnn.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from fgw_validation.eval import _l2norm, _emb_path


def _mnn_pairs(emb_root: Path, splits: dict, witness: str):
    """Returns (pairs_with_sim, ids_i, ids_a). pairs_with_sim is a list of
    (i, j, sim_value) for the kept MNN pairs."""
    di = torch.load(str(_emb_path(emb_root, "flickr8k", splits["flickr8k"],
                                  witness, "text")), weights_only=False)
    da = torch.load(str(_emb_path(emb_root, "clotho", splits["clotho"],
                                  witness, "text")), weights_only=False)
    ti = di["emb"].mean(dim=1).numpy()           # (N_i, D)
    ta = da["emb"].mean(dim=1).numpy()           # (N_a, D)
    sim = _l2norm(ti) @ _l2norm(ta).T            # (N_i, N_a)
    best_for_i = sim.argmax(axis=1)              # best audio per image
    best_for_j = sim.argmax(axis=0)              # best image per audio
    pairs = [(int(i), int(best_for_i[i]), float(sim[i, best_for_i[i]]))
             for i in range(sim.shape[0])
             if best_for_j[best_for_i[i]] == i]
    return pairs, di["ids"], da["ids"]


def _write_gallery(pairs, ids_i, ids_a, emb_root: Path, splits: dict,
                   witness: str, out_path: Path, top_k: int = 30) -> None:
    """Markdown gallery of the top-k pairs by similarity, plus a histogram
    of all kept-pair similarities. Lets you eyeball whether the MNN filter
    is selecting genuinely-paired items vs spurious matches.

    Caption lookup degrades silently if the raw datasets aren't reachable
    (e.g. embeddings copied to a machine without the dataset folders).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sims = np.array([s for _, _, s in pairs]) if pairs else np.array([0.0])

    # similarity histogram
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.hist(sims, bins=min(30, max(5, len(sims) // 3)), edgecolor="black", alpha=0.85)
    ax.set_xlabel(f"MNN pair similarity under '{witness}'")
    ax.set_ylabel("# pairs")
    ax.set_title(f"{len(pairs)} MNN pairs — witness={witness}")
    if len(sims) > 1:
        ax.axvline(float(np.median(sims)), ls="--", color="C1",
                   label=f"median={np.median(sims):.3f}")
        ax.legend()
    fig.tight_layout()
    hist_path = out_path.with_name(out_path.stem + "_sim_hist.png")
    fig.savefig(hist_path, dpi=120); plt.close(fig)

    # captions (lazy + tolerant: skip if dataset folders missing)
    try:
        from fgw_validation.eval import _captions_for
        caps_i = _captions_for(str(emb_root), "flickr8k", splits["flickr8k"])
        caps_a = _captions_for(str(emb_root), "clotho",   splits["clotho"])
    except Exception:
        caps_i = caps_a = {}

    pairs_sorted = sorted(pairs, key=lambda p: -p[2])[:top_k]
    md_path = out_path.with_name(out_path.stem + "_gallery.md")

    def _two(caps):
        if not caps: return "_(no captions)_"
        return "<br>".join(c.strip() for c in caps[:2])

    with open(md_path, "w") as f:
        f.write(f"# MNN filter gallery — witness: **{witness}**\n\n")
        f.write(f"- Strategy: mutual nearest-neighbour pairs under `{witness}`-text mean-pooled cosine.\n")
        f.write(f"- **{len(pairs)}** pairs between Flickr8k `{splits['flickr8k']}` "
                f"and Clotho `{splits['clotho']}`.\n")
        if len(sims) > 1:
            f.write(f"- Pair similarity: median **{np.median(sims):.3f}**, "
                    f"q25 {np.quantile(sims, 0.25):.3f}, q75 {np.quantile(sims, 0.75):.3f}.\n")
        f.write(f"- Histogram: `{hist_path.name}`\n\n")
        f.write(f"## Top {len(pairs_sorted)} pairs by similarity\n\n")
        f.write("| # |  sim | flickr8k id | flickr8k captions | clotho id | clotho captions |\n")
        f.write("|---|-----:|---|---|---|---|\n")
        for k, (i, j, s) in enumerate(pairs_sorted, 1):
            iid = ids_i[i]; aid = ids_a[j]
            f.write(f"| {k} | {s:.3f} | `{iid}` | {_two(caps_i.get(iid, []))} "
                    f"| `{aid}` | {_two(caps_a.get(aid, []))} |\n")
    print(f"[mnn] gallery → {md_path}")
    print(f"[mnn] hist    → {hist_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--witness", required=True,
                    choices=["clip", "clap", "roberta", "t5"],
                    help="Encoder used to define mutual NN. Must differ from "
                         "the encoders used as FGW eval witnesses.")
    ap.add_argument("--emb_root",     default="fgw_validation/data/embeddings")
    ap.add_argument("--flickr_split", default="test")
    ap.add_argument("--clotho_split", default="development")
    ap.add_argument("--out",          required=True)
    ap.add_argument("--top_k_gallery", type=int, default=30,
                    help="Number of top-similarity pairs to show in the markdown "
                         "gallery (0 to disable gallery + histogram).")
    args = ap.parse_args()

    splits = {"flickr8k": args.flickr_split, "clotho": args.clotho_split}
    emb_root = Path(args.emb_root)
    pairs_with_sim, ids_i, ids_a = _mnn_pairs(emb_root, splits, args.witness)

    # JSON consumed by fgw_text_bridge stores (i, j) only; sim is gallery-only.
    pairs_ij = [(i, j) for i, j, _ in pairs_with_sim]
    out = {"witness": args.witness, "strategy": "mnn",
           "splits": splits, "n_pairs": len(pairs_ij), "pairs": pairs_ij}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[mnn] witness={args.witness}  pairs found: {len(pairs_ij)}  → {out_path}")

    if args.top_k_gallery > 0:
        _write_gallery(pairs_with_sim, ids_i, ids_a, emb_root, splits,
                       args.witness, out_path, top_k=args.top_k_gallery)


if __name__ == "__main__":
    main()
