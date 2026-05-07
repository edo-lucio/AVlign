"""Encode the 309 VGGSound class label strings with CLAP-text.

Output: data/cache/class_text_clap.pt   tensor (309, 512)
        data/cache/class_names.json     list[str]
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import ClapModel, ClapProcessor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="vggsound.csv")
    ap.add_argument("--out_dir", default="data/cache")
    ap.add_argument("--clap_id", default="laion/clap-htsat-fused")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    # Collect unique class names in a stable, sorted order.
    classes = set()
    import csv
    with open(args.csv) as f:
        for row in csv.reader(f):
            if len(row) >= 3:
                classes.add(row[2].strip())
    class_names = sorted(classes)
    print(f"found {len(class_names)} classes")

    model = ClapModel.from_pretrained(args.clap_id).to(args.device).eval()
    proc = ClapProcessor.from_pretrained(args.clap_id)

    embs = []
    with torch.inference_mode():
        for i in range(0, len(class_names), 32):
            batch = class_names[i:i + 32]
            inp = proc(text=batch, return_tensors="pt", padding=True).to(args.device)
            e = model.get_text_features(**inp)
            embs.append(e.cpu())
    embs = torch.cat(embs, dim=0)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(embs, out / "class_text_clap.pt")
    with open(out / "class_names.json", "w") as f:
        json.dump(class_names, f)
    print(f"saved {embs.shape} to {out / 'class_text_clap.pt'}")


if __name__ == "__main__":
    main()
