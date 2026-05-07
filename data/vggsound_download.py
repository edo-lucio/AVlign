"""Minimal VGGSound downloader.
Reads the official vggsound.csv (columns: ytid, start_sec, label, split) and
fetches a 10s clip per row using yt-dlp + ffmpeg.
Cookie rotation across a folder of Netscape-format cookie files keeps the
session fresh. Progress is persisted to done.txt so jobs can be killed and
safely resumed.
"""
import argparse
import csv
import random
import subprocess
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


# ─── Cookie pool ─────────────────────────────────────────────────────────────

class CookiePool:
    """Round-robin cookie file rotator with per-file cooldown after a 429.

    Put any number of Netscape-format cookie.txt files in a folder:
        cookies/
            account1.txt
            account2.txt
            ...
    Export them with the browser extension 'Get cookies.txt LOCALLY'.
    Each worker thread picks the next ready cookie file in rotation.
    If a file is on cooldown it is skipped until the cooldown expires.
    """

    def __init__(self, cookie_dir: Path, cooldown: float = 1.0):
        files = sorted(cookie_dir.glob("*.txt"))
        if not files:
            raise FileNotFoundError(f"No *.txt cookie files found in {cookie_dir}")
        self._files = files
        self._cooldown = cooldown
        self._index = 0
        self._lock = threading.Lock()
        self._on_cooldown: dict[Path, float] = {}  # path -> timestamp when flagged
        print(f"[cookies] loaded {len(files)} file(s): {[f.name for f in files]}")

    def get(self) -> Path:
        """Return the next ready cookie file (blocks if all are cooling down)."""
        with self._lock:
            now = time.time()
            ready = [
                f for f in self._files
                if now - self._on_cooldown.get(f, 0) >= self._cooldown
            ]
            if not ready:
                wait = self._cooldown - (now - min(self._on_cooldown.values()))
                print(f"[cookies] all files cooling down, waiting {wait:.0f}s")
                time.sleep(max(wait, 1))
                return self.get()
            chosen = ready[self._index % len(ready)]
            self._index += 1
            return chosen

    def flag_rate_limited(self, cookie_file: Path):
        """Call this when yt-dlp signals a 429 for a given cookie file."""
        with self._lock:
            self._on_cooldown[cookie_file] = time.time()
            print(f"[cookies] {cookie_file.name} flagged — cooldown {self._cooldown:.0f}s")


# ─── Progress tracker ─────────────────────────────────────────────────────────

class ProgressTracker:
    """Thread-safe set of completed ytids backed by a plain-text file.

    Format: one ytid per line. Safe to kill the process at any time —
    each ytid is flushed immediately after a successful download.
    On restart the file is read back and already-done ids are skipped.
    """

    def __init__(self, path: Path):
        self._path = path
        self._lock = threading.Lock()
        self._done: set[str] = set()
        if path.exists():
            self._done = set(path.read_text().splitlines())
            print(f"[resume] {len(self._done)} already done, skipping")

    def is_done(self, ytid: str) -> bool:
        return ytid in self._done

    def mark_done(self, ytid: str):
        with self._lock:
            if ytid not in self._done:
                self._done.add(ytid)
                with self._path.open("a") as fh:
                    fh.write(ytid + "\n")


# ─── Downloader ───────────────────────────────────────────────────────────────

_RATE_LIMIT_SIGNALS = ("429", "rate limit", "too many requests", "sign in")

def download_one(
    ytid: str,
    start_sec: float,
    out_path: Path,
    pool: CookiePool,
    retries: int = 4,
) -> bool:
    if out_path.exists():
        return True
    out_path.parent.mkdir(parents=True, exist_ok=True)

    url = f"https://www.youtube.com/watch?v={ytid}"
    end_sec = start_sec + 10

    for attempt in range(retries):
        cookie_file = pool.get()
        cmd = [
            "yt-dlp",
            "-f", "best[height<=360]/best",
            "--quiet", "--no-warnings",
            "--cookies", str(cookie_file),          # <-- rotated cookie
            "--download-sections", f"*{start_sec}-{end_sec}",
            "--sleep-requests", "1",                # 1s between yt-dlp internal requests
            "-o", str(out_path),
            url,
        ]
        try:
            result = subprocess.run(
                cmd, check=True, timeout=120,
                capture_output=True, text=True,
            )
            return True

        except subprocess.CalledProcessError as e:
            stderr = (e.stderr or "").lower()
            is_rate_limited = any(sig in stderr for sig in _RATE_LIMIT_SIGNALS)

            if is_rate_limited:
                pool.flag_rate_limited(cookie_file)
                # no sleep here — pool cooldown handles the wait
            else:
                backoff = 5 * (2 ** attempt) + random.uniform(0, 3)
                print(f"[retry {attempt+1}/{retries}] {ytid} — wait {backoff:.1f}s")
                time.sleep(backoff)

        except Exception as e:
            print(f"[fail] {ytid}: {e}")
            break

    return False


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",         default="vggsound.csv")
    ap.add_argument("--out_dir",     default="./data/vggsound")
    ap.add_argument("--cookies_dir", default="./cookies",
                    help="Folder of Netscape *.txt cookie files")
    ap.add_argument("--split",       default="train", choices=["train", "test"])
    ap.add_argument("--limit",       type=int,   default=None)
    ap.add_argument("--workers",     type=int,   default=2)
    ap.add_argument("--cooldown",    type=float, default=60.0,
                    help="Seconds to cool a cookie file after a 429")
    ap.add_argument("--done_file",   default="done.txt",
                    help="Resume checkpoint — one ytid per line")
    args = ap.parse_args()

    cookie_pool = CookiePool(Path(args.cookies_dir), cooldown=args.cooldown)
    tracker     = ProgressTracker(Path(args.done_file))

    rows = []
    with open(args.csv) as f:
        reader = csv.reader(f)
        for r in reader:
            if len(r) < 4:
                continue
            ytid, start, label, split = r[0], float(r[1]), r[2], r[3]
            if split.strip() != args.split:
                continue
            if tracker.is_done(ytid):      # <-- skip already completed
                continue
            rows.append((ytid, start, label))
            if args.limit and len(rows) >= args.limit:
                break

    out_dir = Path(args.out_dir)
    print(f"[start] {len(rows)} clips remaining, {args.workers} worker(s)")

    def task(row):
        ytid, start, _ = row
        ok = download_one(ytid, start, out_dir / f"{ytid}.mp4", cookie_pool)
        if ok:
            tracker.mark_done(ytid)
        time.sleep(random.uniform(1.0, 3.5))   # per-worker jitter
        return ok

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(task, rows))

    print(f"downloaded {sum(results)}/{len(rows)}")

if __name__ == "__main__":
    main()