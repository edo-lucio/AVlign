"""Flickr8k (image-text) and Clotho (audio-text) for FGW validation.

Both are 5-caption-per-item benchmarks. Each Dataset yields paths + caption
lists; loading/preprocessing is left to the caller. CLI:

    python -m fgw_validation.datasets --dataset flickr8k --out_dir data/flickr8k
    python -m fgw_validation.datasets --dataset clotho   --out_dir data/clotho

Sources:
  Flickr8k — github.com/jbrownlee/Datasets release (stable mirror)
  Clotho v2.1 — Zenodo record 4783391
"""
import argparse
import csv
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

from torch.utils.data import Dataset
from tqdm import tqdm


# ─── Source URLs ─────────────────────────────────────────────────────────────

FLICKR8K_URLS = {
    "images": "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip",
    "text":   "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip",
}

CLOTHO_BASE   = "https://zenodo.org/records/4783391/files"
CLOTHO_SPLITS = ("development", "validation", "evaluation")


# ─── Download helpers ────────────────────────────────────────────────────────

def _download(url: str, dst: Path) -> Path:
    if dst.exists() and dst.stat().st_size > 0:
        print(f"[skip] {dst.name} already present")
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"[get] {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "fgw-validation/0.1"})
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        with open(tmp, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dst.name, leave=False
        ) as bar:
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                f.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dst)
    return dst


def _unzip(zip_path: Path, out_dir: Path) -> None:
    print(f"[unzip] {zip_path.name} -> {out_dir}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)


def _un7z(archive: Path, out_dir: Path) -> None:
    if shutil.which("7z") is None:
        raise RuntimeError(
            "`7z` not found — install p7zip (e.g. `apt-get install p7zip-full`) "
            "to extract Clotho .7z archives"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[un7z] {archive.name} -> {out_dir}")
    subprocess.run(
        ["7z", "x", "-y", f"-o{out_dir}", str(archive)],
        check=True, stdout=subprocess.DEVNULL,
    )


# ─── Flickr8k ────────────────────────────────────────────────────────────────

# The images zip extracts to "Flicker8k_Dataset" (sic — official typo).
_FLICKR_IMG_DIR = "Flicker8k_Dataset"
_FLICKR_SPLIT_FILES = {
    "train": "Flickr_8k.trainImages.txt",
    "val":   "Flickr_8k.devImages.txt",
    "test":  "Flickr_8k.testImages.txt",
}


def download_flickr8k(out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    images_zip = _download(FLICKR8K_URLS["images"], out_dir / "Flickr8k_Dataset.zip")
    text_zip   = _download(FLICKR8K_URLS["text"],   out_dir / "Flickr8k_text.zip")
    if not (out_dir / _FLICKR_IMG_DIR).exists():
        _unzip(images_zip, out_dir)
    if not (out_dir / "Flickr8k.token.txt").exists():
        _unzip(text_zip, out_dir)
    print(f"[done] flickr8k -> {out_dir}")
    return out_dir


def _read_flickr8k_captions(root: Path) -> dict[str, list[str]]:
    """image_filename -> list of 5 captions, parsed from Flickr8k.token.txt."""
    captions: dict[str, list[str]] = {}
    for line in (root / "Flickr8k.token.txt").read_text().splitlines():
        if not line.strip():
            continue
        # format:  '1000268201_693b08cb0e.jpg#0\tA child in a pink dress ...'
        key, caption = line.split("\t", 1)
        img = key.split("#", 1)[0]
        captions.setdefault(img, []).append(caption.strip())
    return captions


class Flickr8kDataset(Dataset):
    """Yields {'image_path', 'captions' (5×str), 'id'}."""

    def __init__(self, root: str | Path, split: str = "train"):
        if split not in _FLICKR_SPLIT_FILES:
            raise ValueError(f"split must be one of {list(_FLICKR_SPLIT_FILES)}, got {split!r}")
        self.root = Path(root)
        self.images_dir = self.root / _FLICKR_IMG_DIR
        split_file = self.root / _FLICKR_SPLIT_FILES[split]
        files = [ln.strip() for ln in split_file.read_text().splitlines() if ln.strip()]
        captions = _read_flickr8k_captions(self.root)
        self.items: list[tuple[str, list[str]]] = [(f, captions[f]) for f in files if f in captions]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        fname, caps = self.items[i]
        return {
            "image_path": str(self.images_dir / fname),
            "captions": caps,
            "id": fname,
        }


# ─── Clotho v2.1 ─────────────────────────────────────────────────────────────

_CLOTHO_USER_TO_NATIVE = {
    "train": "development",
    "val":   "validation",
    "test":  "evaluation",
    "development": "development",
    "validation":  "validation",
    "evaluation":  "evaluation",
}


def download_clotho(
    out_dir: str | Path,
    splits: tuple[str, ...] = CLOTHO_SPLITS,
) -> Path:
    out_dir = Path(out_dir)
    for split in splits:
        if split not in CLOTHO_SPLITS:
            raise ValueError(f"unknown clotho split {split!r}")
        audio_archive = _download(
            f"{CLOTHO_BASE}/clotho_audio_{split}.7z",
            out_dir / f"clotho_audio_{split}.7z",
        )
        audio_subdir = out_dir / split
        if not (audio_subdir.exists() and any(audio_subdir.iterdir())):
            _un7z(audio_archive, out_dir)
        _download(
            f"{CLOTHO_BASE}/clotho_captions_{split}.csv",
            out_dir / f"clotho_captions_{split}.csv",
        )
    print(f"[done] clotho -> {out_dir}")
    return out_dir


class ClothoDataset(Dataset):
    """Yields {'audio_path', 'captions' (5×str), 'id'}.

    Accepts either native split names (development/validation/evaluation) or
    the train/val/test aliases.
    """

    def __init__(self, root: str | Path, split: str = "train"):
        if split not in _CLOTHO_USER_TO_NATIVE:
            raise ValueError(
                f"split must be one of {list(_CLOTHO_USER_TO_NATIVE)}, got {split!r}"
            )
        self.root = Path(root)
        native = _CLOTHO_USER_TO_NATIVE[split]
        self.audio_dir = self.root / native
        captions_csv = self.root / f"clotho_captions_{native}.csv"
        self.items: list[tuple[str, list[str]]] = []
        missing = 0
        with open(captions_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row["file_name"]
                if not (self.audio_dir / fname).exists():
                    missing += 1
                    continue
                caps = [row[f"caption_{i}"] for i in range(1, 6)]
                self.items.append((fname, caps))
        if missing:
            print(f"[clotho/{native}] dropped {missing} item(s) with missing audio")
        if not self.items:
            raise FileNotFoundError(
                f"clotho/{native}: no audio files found under {self.audio_dir} "
                f"(captions present, audio archive not extracted)"
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        fname, caps = self.items[i]
        return {
            "audio_path": str(self.audio_dir / fname),
            "captions": caps,
            "id": fname,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["flickr8k", "clotho", "all"])
    ap.add_argument("--out_dir", required=True,
                    help="If --dataset=all, flickr8k/ and clotho/ subdirs are created here")
    ap.add_argument("--clotho_splits", nargs="+", default=list(CLOTHO_SPLITS),
                    choices=list(CLOTHO_SPLITS))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if args.dataset in ("flickr8k", "all"):
        download_flickr8k(out_dir / "flickr8k" if args.dataset == "all" else out_dir)
    if args.dataset in ("clotho", "all"):
        target = out_dir / "clotho" if args.dataset == "all" else out_dir
        download_clotho(target, splits=tuple(args.clotho_splits))


if __name__ == "__main__":
    main()
