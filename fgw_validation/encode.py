"""Pre-compute encoder embeddings for Flickr8k & Clotho.

Reads datasets from `<data_root>/{flickr8k,clotho}/...` and writes embeddings
to `<data_root>/embeddings/<dataset>/<split>/<encoder>_<size>_<modality>.pt`
where each .pt holds:

    {"emb": Tensor, "ids": list[str], "encoder": str,
     "modality": "image"|"audio"|"text", "dim": int}

Image / audio embeddings are (N, D); text embeddings are (N, 5, D) — both
benchmarks ship 5 captions per item.

CLI:

    python -m fgw_validation.encode                   # all datasets, all encoders
    python -m fgw_validation.encode --dataset clotho --encoders clap ast
    python -m fgw_validation.encode --size medium --batch_size 8
"""
import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from datasets import CLOTHO_SPLITS, ClothoDataset, Flickr8kDataset
from models import ENCODERS, MODEL_IDS, build_encoder


# ─── modality / split tables ─────────────────────────────────────────────────

DATASET_MODALITY = {
    "flickr8k": "image",
    "clotho":   "audio",
}

DATASET_SPLITS = {
    "flickr8k": ("train", "val", "test"),
    "clotho":   CLOTHO_SPLITS,
}

IMAGE_ENCODERS = ("clip", "dinov2")
AUDIO_ENCODERS = ("clap", "ast")
TEXT_ENCODERS  = ("clip", "clap", "roberta", "t5")


def _is_applicable(enc_name: str, dataset_name: str) -> bool:
    if enc_name in TEXT_ENCODERS:
        return True
    if DATASET_MODALITY[dataset_name] == "image" and enc_name in IMAGE_ENCODERS:
        return True
    if DATASET_MODALITY[dataset_name] == "audio" and enc_name in AUDIO_ENCODERS:
        return True
    return False


def _resolve_size(enc_name: str, requested: str) -> str:
    """Fall back to the only-available size for families that don't ship the requested one."""
    available = MODEL_IDS[enc_name]
    if requested in available:
        return requested
    return next(iter(available))


# ─── helpers ─────────────────────────────────────────────────────────────────

def _open_dataset(name: str, split: str, root: Path):
    if name == "flickr8k":
        return Flickr8kDataset(root / "flickr8k", split=split)
    if name == "clotho":
        return ClothoDataset(root / "clotho", split=split)
    raise KeyError(name)


def _batched(fn, items, batch_size: int, desc: str) -> torch.Tensor:
    out = []
    for i in tqdm(range(0, len(items), batch_size), desc=desc, leave=False):
        out.append(fn(items[i:i + batch_size]))
    return torch.cat(out, dim=0)


def _save(path: Path, *, emb: torch.Tensor, ids: list[str],
          encoder: str, modality: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "emb": emb,
        "ids": ids,
        "encoder": encoder,
        "modality": modality,
        "dim": emb.shape[-1],
    }, path)


# ─── per-split encoding ──────────────────────────────────────────────────────

def _encode_split(enc, enc_name: str, size: str, dataset_name: str, split: str,
                  ds, out_dir: Path, batch_size: int, overwrite: bool) -> None:
    items       = [ds[i] for i in range(len(ds))]
    ids         = [it["id"] for it in items]
    captions    = [it["captions"] for it in items]
    if not all(len(c) == 5 for c in captions):
        raise ValueError(f"{dataset_name}/{split}: not every item has 5 captions")

    media_mod  = DATASET_MODALITY[dataset_name]                # "image" | "audio"
    media_key  = f"{media_mod}_path"
    media_pts  = [it[media_key] for it in items]
    base       = out_dir / dataset_name / split
    tag        = f"{enc_name}_{size}"

    # media (image or audio) ---------------------------------------------------
    media_fn   = getattr(enc, f"encode_{media_mod}", None)
    media_path = base / f"{tag}_{media_mod}.pt"
    if media_fn is not None and (overwrite or not media_path.exists()):
        emb = _batched(media_fn, media_pts, batch_size,
                       desc=f"{dataset_name}/{split} {tag} {media_mod}")
        _save(media_path, emb=emb, ids=ids, encoder=tag, modality=media_mod)
        print(f"[save] {media_path} {tuple(emb.shape)}")
    elif media_fn is not None:
        print(f"[skip] {media_path} (exists)")

    # text ---------------------------------------------------------------------
    text_path = base / f"{tag}_text.pt"
    if hasattr(enc, "encode_text") and (overwrite or not text_path.exists()):
        flat = [c for caps in captions for c in caps]
        flat_emb = _batched(enc.encode_text, flat, batch_size,
                            desc=f"{dataset_name}/{split} {tag} text")
        emb = flat_emb.reshape(len(ids), 5, -1)
        _save(text_path, emb=emb, ids=ids, encoder=tag, modality="text")
        print(f"[save] {text_path} {tuple(emb.shape)}")
    elif hasattr(enc, "encode_text"):
        print(f"[skip] {text_path} (exists)")


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all", choices=["flickr8k", "clotho", "all"])
    ap.add_argument("--splits", nargs="+", default=None,
                    help="Splits to encode (default: all available for each dataset)")
    ap.add_argument("--encoders", nargs="+", default=["all"],
                    help=f"Subset of {sorted(ENCODERS) + ['all']}")
    ap.add_argument("--size", default="small", choices=["small", "medium"],
                    help="Falls back per family if the size isn't published (CLAP/AST → medium)")
    ap.add_argument("--data_root", default="fgw_validation/data",
                    help="Folder containing flickr8k/ and clotho/ subdirs")
    ap.add_argument("--out_dir", default=None,
                    help="Default: <data_root>/embeddings")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", default=None, help="default: cuda if available else cpu")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir   = Path(args.out_dir) if args.out_dir else data_root / "embeddings"

    encoder_names = list(ENCODERS) if args.encoders == ["all"] else args.encoders
    for n in encoder_names:
        if n not in ENCODERS:
            raise SystemExit(f"unknown encoder {n!r}; pick from {sorted(ENCODERS)}")

    datasets = ["flickr8k", "clotho"] if args.dataset == "all" else [args.dataset]

    # Outer loop over encoders so each HF model loads once.
    for enc_name in encoder_names:
        applicable = [d for d in datasets if _is_applicable(enc_name, d)]
        if not applicable:
            print(f"[skip] {enc_name}: no applicable datasets in {datasets}")
            continue

        size = _resolve_size(enc_name, args.size)
        enc = build_encoder(enc_name, size=size, device=args.device)
        print(f"[load] {enc_name} ({size}) dim={enc.dim} -> {applicable}")

        for dataset_name in applicable:
            wanted = args.splits or DATASET_SPLITS[dataset_name]
            for split in wanted:
                if split not in DATASET_SPLITS[dataset_name]:
                    continue                       # silently ignore irrelevant splits
                try:
                    ds = _open_dataset(dataset_name, split, data_root)
                except FileNotFoundError as e:
                    print(f"[skip] {dataset_name}/{split}: {e}")
                    continue
                _encode_split(enc, enc_name, size, dataset_name, split, ds,
                              out_dir, args.batch_size, args.overwrite)

        del enc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
