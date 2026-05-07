"""Extract middle frame from each clip, encode with CLIP ViT-L/14, save .pt.

Writes to data/cache/shards/<ytid>.pt with key 'clip_emb' (768,). Other keys are
added by preprocess_audio.py.
"""
import argparse
import csv
import json
from pathlib import Path

import av
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm


def middle_frame(path: Path) -> Image.Image | None:
    try:
        container = av.open(str(path))
        stream = container.streams.video[0]
        n = stream.frames or 0
        target = n // 2 if n else 0
        for i, frame in enumerate(container.decode(stream)):
            if i >= target:
                return frame.to_image()
    except Exception:
        return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="vggsound.csv")
    ap.add_argument("--video_dir", default="data/vggsound")
    ap.add_argument("--out_dir", default="data/cache")
    ap.add_argument("--clip_id", default="openai/clip-vit-large-patch14")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    rows = []
    with open(args.csv) as f:
        for r in csv.reader(f):
            if len(r) >= 4:
                rows.append({"ytid": r[0], "label": r[2].strip(), "split": r[3].strip()})

    with open(Path(args.out_dir) / "class_names.json") as f:
        class_names = json.load(f)
    cls2id = {c: i for i, c in enumerate(class_names)}

    model = CLIPModel.from_pretrained(args.clip_id).to(args.device).eval()
    proc = CLIPProcessor.from_pretrained(args.clip_id)

    shard_dir = Path(args.out_dir) / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    pending_imgs, pending_rows = [], []

    def flush():
        if not pending_imgs:
            return
        inp = proc(images=pending_imgs, return_tensors="pt").to(args.device)
        with torch.inference_mode():
            feats = model.get_image_features(**inp).cpu()
        for r, e in zip(pending_rows, feats):
            shard_path = shard_dir / f"{r['ytid']}.pt"
            existing = torch.load(shard_path) if shard_path.exists() else {}
            existing["clip_emb"] = e
            existing["class_id"] = cls2id[r["label"]]
            existing["split"] = r["split"]
            torch.save(existing, shard_path)
        pending_imgs.clear()
        pending_rows.clear()

    for r in tqdm(rows):
        vid = Path(args.video_dir) / f"{r['ytid']}.mp4"
        if not vid.exists():
            continue
        img = middle_frame(vid)
        if img is None:
            continue
        pending_imgs.append(img)
        pending_rows.append(r)
        if len(pending_imgs) >= args.batch:
            flush()
    flush()
    print("done")


if __name__ == "__main__":
    main()
