"""Download the Big Buck Bunny sample clip into this benchmarks folder.

Skips the download if the file already exists (and is non-empty). OS-agnostic.
"""

import sys
import urllib.request
from pathlib import Path

# Big Buck Bunny (CC-BY 3.0), 360p 10s ~10MB cut, valid-cert mirror.
VIDEO_URL = "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_10MB.mp4"
FILENAME = "bigbugsbunny.mp4"

DEST = Path(__file__).resolve().parent / FILENAME


def _progress(done: int, total: int) -> None:
    if total <= 0:
        sys.stdout.write(f"\rDownloading {FILENAME}: {done} bytes")
    else:
        pct = min(done, total) * 100 // total
        sys.stdout.write(f"\rDownloading {FILENAME}: {pct:3d}% ({done}/{total} bytes)")
    sys.stdout.flush()


def main() -> int:
    if DEST.exists() and DEST.stat().st_size > 0:
        print(f"Already downloaded: {DEST} ({DEST.stat().st_size} bytes)")
        return 0

    print(f"Downloading {VIDEO_URL}\n            -> {DEST}")
    tmp = DEST.with_suffix(DEST.suffix + ".part")
    # Custom UA: some mirrors 403 the default urllib agent.
    req = urllib.request.Request(
        VIDEO_URL, headers={"User-Agent": "Mozilla/5.0 (TAS-benchmark)"}
    )
    try:
        with urllib.request.urlopen(req) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            done = 0
            with open(tmp, "wb") as fh:
                while True:
                    chunk = resp.read(1 << 16)
                    if not chunk:
                        break
                    fh.write(chunk)
                    done += len(chunk)
                    _progress(done, total)
    except Exception as exc:  # noqa: BLE001
        if tmp.exists():
            tmp.unlink()
        print(f"\nDownload failed: {exc}", file=sys.stderr)
        return 1

    tmp.replace(DEST)
    print(f"\nDone: {DEST} ({DEST.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
