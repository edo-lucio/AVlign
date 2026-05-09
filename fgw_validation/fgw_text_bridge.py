"""FGW image→audio alignment with captions as the cross-modal bridge.

Loads encoder embeddings produced by `fgw_validation.encode`, samples n items
from each dataset, builds:

    C_i  =  cosine-distance Gram matrix on image embeddings   (n_i × n_i)
    C_a  =  cosine-distance Gram matrix on audio embeddings   (n_a × n_a)
    M    =  cross-modal feature cost from caption embeddings  (n_i × n_a)

then runs Fused Gromov-Wasserstein. Sweeps any subset of:

    image encoder      ∈ {clip, dinov2}
    audio encoder      ∈ {clap, ast}
    text bridge        ∈ {clip, clap, roberta, t5}     (same model both sides)
    cost convention    ∈ {cos_cos, cos_neg, geo_cos}   (see _build_costs)
    caption aggregation∈ {mean, first}

`geo_cos` is gated to (image, audio) pairs whose encoders share a
contrastive InfoNCE-trained hypersphere — currently only (clip, clap).
For other pairs the geodesic distance is meaningless, so those combos are
silently dropped from the sweep. With the default sets this yields:

      cos_cos  : 2 × 2 × 4 × 2  = 32 combos
      cos_neg  : 2 × 2 × 4 × 2  = 32 combos
      geo_cos  : 1 × 1 × 4 × 2  =  8 combos
                                  ──
                                  72 total.

CLI:

    python -m fgw_validation.fgw_text_bridge --n 200
    python -m fgw_validation.fgw_text_bridge --image_encoders clip \\
        --audio_encoders clap --text_encoders clip --cost_conventions cos_cos \\
        --caption_aggs mean
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import ot
import torch
from tqdm import tqdm


IMAGE_ENCODERS    = ("clip", "dinov2")
AUDIO_ENCODERS    = ("clap", "ast")
TEXT_ENCODERS     = ("clip", "clap", "roberta", "t5")
COST_CONVENTIONS  = ("cos_cos", "cos_neg", "geo_cos")
CAPTION_AGGS      = ("mean", "first")

# Encoders trained with a contrastive InfoNCE on L2-normalized features —
# their embeddings live on a unit hypersphere where geodesic distance is
# meaningful. `geo_cos` is only emitted for combos where BOTH the image and
# audio encoders are in this set; for non-contrastive encoders (DINOv2 image,
# AST audio) the spherical structure is just imposed and arccos(⟨·,·⟩) is
# not a principled cost.
HYPERSPHERICAL_ENCODERS = frozenset({"clip", "clap"})


def _combo_is_valid(img_enc: str, aud_enc: str, cost_conv: str) -> bool:
    """`geo_cos` only fires when both image and audio encoders share a
    contrastively-trained hypersphere; other combos are dropped from the sweep."""
    if cost_conv == "geo_cos":
        return img_enc in HYPERSPHERICAL_ENCODERS and aud_enc in HYPERSPHERICAL_ENCODERS
    return True

# Sizes must match what `fgw_validation.encode` actually wrote to disk.
_SIZE_PER_ENCODER = {
    "clip":    "small",
    "dinov2":  "small",
    "clap":    "medium",
    "ast":     "medium",
    "roberta": "small",
    "t5":      "small",
}


# ─── loading ─────────────────────────────────────────────────────────────────

def _emb_path(emb_root: Path, dataset: str, split: str,
              encoder: str, modality: str) -> Path:
    return emb_root / dataset / split / f"{encoder}_{_SIZE_PER_ENCODER[encoder]}_{modality}.pt"


def _load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"missing embeddings: {path}\n"
            f"Run `python -m fgw_validation.encode` first to produce them."
        )
    return torch.load(path, weights_only=False)


# ─── building C_i, C_a, M ────────────────────────────────────────────────────

def _l2norm(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp(min=eps)


def _aggregate_captions(text_emb: torch.Tensor, mode: str) -> torch.Tensor:
    """text_emb: (N, 5, D) -> (N, D)."""
    if mode == "mean":
        return text_emb.mean(dim=1)
    if mode == "first":
        return text_emb[:, 0]
    raise ValueError(f"unknown caption aggregation {mode!r}")


def _build_costs(z_i: torch.Tensor, z_a: torch.Tensor,
                 t_i: torch.Tensor, t_a: torch.Tensor,
                 convention: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (C_i, C_a, M) as float64 numpy arrays for POT.

    Conventions:

      cos_cos:  C = 1 − cos(z, z)    on [0, 2]   (chord on unit sphere)
                M = 1 − cos(T_i, T_a) on [0, 2]
      cos_neg:  C = 1 − cos(z, z)    on [0, 2]
                M = −⟨T_i, T_a⟩      raw (sign-flipped so FGW minimizes)
      geo_cos:  C = arccos(cos)/π     on [0, 1]   (geodesic on S^{d−1},
                                                   rescaled for parity with cos_cos)
                M = 1 − cos(T_i, T_a) on [0, 2]
                ⟨ gated to (image, audio) pairs in HYPERSPHERICAL_ENCODERS ⟩

    `geo_cos` is only meaningful when both source spaces are contrastive
    hyperspheres; the sweep drops other combos via `_combo_is_valid`.
    """
    z_i_n = _l2norm(z_i)
    z_a_n = _l2norm(z_a)

    if convention in ("cos_cos", "cos_neg"):
        C_i = (1.0 - z_i_n @ z_i_n.T).clamp(min=0).cpu().numpy()
        C_a = (1.0 - z_a_n @ z_a_n.T).clamp(min=0).cpu().numpy()
    elif convention == "geo_cos":
        # arccos has divergent gradient at ±1 — clamp before applying it.
        eps = 1e-7
        cos_i = (z_i_n @ z_i_n.T).clamp(-1.0 + eps, 1.0 - eps)
        cos_a = (z_a_n @ z_a_n.T).clamp(-1.0 + eps, 1.0 - eps)
        C_i = (torch.arccos(cos_i) / torch.pi).cpu().numpy()
        C_a = (torch.arccos(cos_a) / torch.pi).cpu().numpy()
    else:
        raise ValueError(f"unknown cost convention {convention!r}")

    if convention in ("cos_cos", "geo_cos"):
        t_i_n = _l2norm(t_i)
        t_a_n = _l2norm(t_a)
        M = (1.0 - t_i_n @ t_a_n.T).clamp(min=0).cpu().numpy()
    elif convention == "cos_neg":
        M = -(t_i @ t_a.T).cpu().numpy()

    return C_i.astype(np.float64), C_a.astype(np.float64), M.astype(np.float64)


# ─── one combination ────────────────────────────────────────────────────────

def _run_one(combo, *, embs, splits, n, alpha, seed, save_plan_dir=None,
             filter_pairs=None, solver="balanced", mass=1.0):
    img_enc, aud_enc, txt_enc, cost_conv, cap_agg = combo
    flickr_split, clotho_split = splits["flickr8k"], splits["clotho"]

    img    = embs[("flickr8k", flickr_split, img_enc, "image")]
    aud    = embs[("clotho",   clotho_split, aud_enc, "audio")]
    txt_i  = embs[("flickr8k", flickr_split, txt_enc, "text")]
    txt_a  = embs[("clotho",   clotho_split, txt_enc, "text")]

    # Same seed -> same items selected across every combination.
    rng = np.random.default_rng(seed)
    if filter_pairs is not None:
        # Symmetric MNN-filtered subset: sample n matched (i, j) pairs.
        # n_i = n_a = n, and every column gets used exactly once → fair
        # for balanced-FGW LP-assignment. See build_filter.py.
        n_eff = min(n, len(filter_pairs))
        order = rng.permutation(len(filter_pairs))[:n_eff]
        chosen = np.asarray(filter_pairs)[order]
        idx_i = chosen[:, 0].astype(np.int64)
        idx_a = chosen[:, 1].astype(np.int64)
        n_i = n_a = n_eff
    else:
        n_i = min(n, len(img["ids"]))
        n_a = min(n, len(aud["ids"]))
        idx_i = rng.choice(len(img["ids"]), n_i, replace=False)
        idx_a = rng.choice(len(aud["ids"]), n_a, replace=False)

    z_i = img["emb"][idx_i]
    z_a = aud["emb"][idx_a]
    t_i = _aggregate_captions(txt_i["emb"][idx_i], cap_agg)
    t_a = _aggregate_captions(txt_a["emb"][idx_a], cap_agg)

    C_i, C_a, M = _build_costs(z_i, z_a, t_i, t_a, cost_conv)

    p = np.full(n_i, 1.0 / n_i)
    q = np.full(n_a, 1.0 / n_a)

    if solver == "balanced":
        T, log = ot.gromov.fused_gromov_wasserstein(
            M, C_i, C_a, p, q,
            loss_fun="square_loss", alpha=alpha, symmetric=True, log=True,
        )
        fgw_dist = float(log.get("fgw_dist", float("nan")))
    elif solver == "partial":
        # Partial FGW: transport only `mass` fraction of total mass — the rest
        # is left unmatched. Lifts the bijection constraint that hurts
        # balanced FGW on asymmetric distributions (one side has few hubs).
        T, log = ot.gromov.partial_fused_gromov_wasserstein(
            M, C_i, C_a, p, q,
            m=mass, loss_fun="square_loss", alpha=alpha, log=True,
        )
        fgw_dist = float(log.get("partial_fgw_dist", log.get("fgw_dist", float("nan"))))
    else:
        raise ValueError(f"unknown solver {solver!r}")

    result = {
        "image_encoder": img_enc,
        "audio_encoder": aud_enc,
        "text_encoder":  txt_enc,
        "cost_convention": cost_conv,
        "caption_agg":   cap_agg,
        "n_image":       int(n_i),
        "n_audio":       int(n_a),
        "alpha":         alpha,
        "seed":          seed,
        "solver":        solver,
        "mass":          float(mass) if solver != "balanced" else 1.0,
        "fgw_dist":      fgw_dist,
        "transport_entropy": float(-(T * np.log(T + 1e-30)).sum()),
        "transport_max":     float(T.max()),
    }

    if save_plan_dir is not None:
        save_plan_dir = Path(save_plan_dir)
        save_plan_dir.mkdir(parents=True, exist_ok=True)
        tag = "__".join(combo)
        np.savez_compressed(
            save_plan_dir / f"{tag}.npz",
            T=T, idx_i=idx_i, idx_a=idx_a,
            ids_i=np.array(img["ids"], dtype=object)[idx_i],
            ids_a=np.array(aud["ids"], dtype=object)[idx_a],
        )

    return result


# ─── sweep + CLI ─────────────────────────────────────────────────────────────

def _validate_subset(name: str, requested: list[str], allowed: tuple[str, ...]) -> list[str]:
    if requested == ["all"]:
        return list(allowed)
    bad = [r for r in requested if r not in allowed]
    if bad:
        raise SystemExit(f"--{name}: unknown values {bad}; pick from {list(allowed)}")
    return requested


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_encoders",   nargs="+", default=["all"])
    ap.add_argument("--audio_encoders",   nargs="+", default=["all"])
    ap.add_argument("--text_encoders",    nargs="+", default=["all"])
    ap.add_argument("--cost_conventions", nargs="+", default=["all"])
    ap.add_argument("--caption_aggs",     nargs="+", default=["all"])

    ap.add_argument("--n",      type=int,   default=200,
                    help="Random subsample size per dataset (clamped to dataset size)")
    ap.add_argument("--alpha",  type=float, default=0.5,
                    help="FGW interpolation: 0=Wasserstein, 1=GW")
    ap.add_argument("--seed",   type=int,   default=0)
    ap.add_argument("--flickr_split", default="train",
                    choices=["train", "val", "test"])
    ap.add_argument("--clotho_split", default="development",
                    choices=["development", "validation", "evaluation"])

    ap.add_argument("--data_root", default="fgw_validation/data")
    ap.add_argument("--emb_dir",   default=None,
                    help="Default: <data_root>/embeddings")
    ap.add_argument("--out", default=None,
                    help="Default: <data_root>/fgw_results.json")
    ap.add_argument("--save_plans", action="store_true",
                    help="Also dump each transport plan as .npz next to --out")
    ap.add_argument("--filter_indices", default=None,
                    help="Path to a build_filter.py JSON. When set, FGW samples "
                         "n matched (image, audio) pairs from the filter instead "
                         "of independent uniform sampling — gives a "
                         "symmetric-coverage subset for balanced FGW.")
    ap.add_argument("--solver", default="balanced", choices=["balanced", "partial"],
                    help="balanced: classical FGW with uniform marginals "
                         "(T is a permutation matrix). partial: relaxes the "
                         "marginal constraint so only `--mass` fraction of mass "
                         "is transported — addresses the LP-bijection penalty "
                         "on asymmetric distributions.")
    ap.add_argument("--mass", type=float, default=0.5,
                    help="Total mass to transport when --solver=partial, in [0, 1]. "
                         "Lower → more rows can be unmatched. Ignored for balanced.")
    args = ap.parse_args()

    img_set  = _validate_subset("image_encoders",   args.image_encoders,   IMAGE_ENCODERS)
    aud_set  = _validate_subset("audio_encoders",   args.audio_encoders,   AUDIO_ENCODERS)
    txt_set  = _validate_subset("text_encoders",    args.text_encoders,    TEXT_ENCODERS)
    cc_set   = _validate_subset("cost_conventions", args.cost_conventions, COST_CONVENTIONS)
    cap_set  = _validate_subset("caption_aggs",     args.caption_aggs,     CAPTION_AGGS)

    data_root = Path(args.data_root)
    emb_root  = Path(args.emb_dir) if args.emb_dir else data_root / "embeddings"
    out_path  = Path(args.out) if args.out else data_root / "fgw_results.json"
    plan_dir  = (out_path.with_suffix("").parent / (out_path.stem + "_plans")
                 if args.save_plans else None)

    splits = {"flickr8k": args.flickr_split, "clotho": args.clotho_split}

    filter_pairs = None
    if args.filter_indices:
        with open(args.filter_indices) as f:
            flt = json.load(f)
        if flt.get("splits") != splits:
            raise SystemExit(
                f"--filter_indices splits {flt.get('splits')} do not match "
                f"current splits {splits}; rebuild the filter or change splits."
            )
        filter_pairs = flt["pairs"]
        print(f"[filter] {flt.get('strategy', '?')} on witness={flt.get('witness')}: "
              f"{len(filter_pairs)} pairs available")

    # Drop combos that fail the cost-convention manifold gate (geo_cos is
    # only valid when both image and audio encoders share a contrastive
    # hypersphere).
    all_combos = list(itertools.product(img_set, aud_set, txt_set, cc_set, cap_set))
    combos = [c for c in all_combos if _combo_is_valid(c[0], c[1], c[3])]
    skipped = len(all_combos) - len(combos)

    # Pre-load every embedding tensor that surviving combos will touch.
    needed: set[tuple[str, str, str, str]] = set()
    for img, aud, txt, _, _ in combos:
        needed.add(("flickr8k", splits["flickr8k"], img, "image"))
        needed.add(("flickr8k", splits["flickr8k"], txt, "text"))
        needed.add(("clotho",   splits["clotho"],   aud, "audio"))
        needed.add(("clotho",   splits["clotho"],   txt, "text"))
    embs = {tup: _load(_emb_path(emb_root, *tup)) for tup in sorted(needed)}
    print(f"[load] {len(embs)} embedding tensors")

    msg = f"[run] {len(combos)} combinations, n={args.n}, alpha={args.alpha}, seed={args.seed}"
    if skipped:
        msg += f"  ({skipped} dropped by cost-convention manifold gate)"
    print(msg)

    results = []
    for combo in tqdm(combos, desc="FGW"):
        try:
            r = _run_one(combo, embs=embs, splits=splits,
                         n=args.n, alpha=args.alpha, seed=args.seed,
                         save_plan_dir=plan_dir,
                         filter_pairs=filter_pairs,
                         solver=args.solver, mass=args.mass)
            results.append(r)
        except Exception as e:
            results.append({
                "image_encoder": combo[0], "audio_encoder": combo[1],
                "text_encoder":  combo[2], "cost_convention": combo[3],
                "caption_agg":   combo[4], "error": f"{type(e).__name__}: {e}",
            })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "splits": splits,
            "n": args.n, "alpha": args.alpha, "seed": args.seed,
            "solver": args.solver, "mass": args.mass,
            "results": results,
        }, f, indent=2)
    print(f"[save] {out_path}  ({len(results)} rows)")


if __name__ == "__main__":
    main()
