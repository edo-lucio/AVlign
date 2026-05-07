"""Minimal VGGSound downloader.

Reads the official vggsound.csv (columns: ytid, start_sec, label, split) and
fetches a 10s clip per row using yt-dlp + ffmpeg.

This is a thin wrapper. yt-dlp is brittle on YouTube — expect a non-trivial
fraction of failures and rate-limiting. Run in batches and retry.
"""
import argparse
import csv
import os
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


def download_one(ytid: str, start_sec: float, out_path: Path) -> bool:
    if out_path.exists():
        return True
    out_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://www.youtube.com/watch?v={ytid}"
    end_sec = start_sec + 10
    cmd = [
        "yt-dlp",
        "-f", "best[height<=360]/best",
        "--quiet", "--no-warnings",
        "--download-sections", f"*{start_sec}-{end_sec}",
        "-o", str(out_path),
        url,
    ]
    try:
        subprocess.run(cmd, check=True, timeout=120)
        return True
    except Exception as e:
        print(f"[fail] {ytid}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="vggsound.csv")
    ap.add_argument("--out_dir", default="./data/vggsound")
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    rows = []
    with open(args.csv) as f:
        reader = csv.reader(f)
        for r in reader:
            if len(r) < 4:
                continue
            ytid, start, label, split = r[0], float(r[1]), r[2], r[3]
            if split.strip() != args.split:
                continue
            rows.append((ytid, start, label))
            if args.limit and len(rows) >= args.limit:
                break

    out_dir = Path(args.out_dir)
    def task(row):
        ytid, start, _ = row
        return download_one(ytid, start, out_dir / f"{ytid}.mp4")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(task, rows))

    print(f"downloaded {sum(results)}/{len(rows)}")


if __name__ == "__main__":
    main()
