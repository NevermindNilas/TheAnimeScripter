"""End-to-end TAS benchmark runner.

Runs main.py across backend variants of three capabilities (interpolate / upscale /
depth) on the sample clip and produces, FOR EACH capability, a graph of:

    FPS, SSIM, PSNR, MSE, VRAM, CPU usage

Quality metrics (SSIM/PSNR/MSE) are measured **relative to the base input video**
(per-frame, output resized to input resolution, nearest input frame by normalized
timestamp). They are intended as regression tripwires: once known-good values are
established, a large divergence implies a backend/model regression rather than a
meaningful absolute fidelity score (a depth map vs an RGB frame is "wrong" but
stable, so it still flags breakage).

Usage (from anywhere):
    python benchmarks/benchmark.py
    python benchmarks/benchmark.py --groups interpolate depth --warmup 0
    python benchmarks/benchmark.py --input path/to/clip.mp4

Outputs land in benchmarks/results/: interpolate.png, upscale.png, depth.png, results.json.
Graphs need matplotlib (pip install -r benchmarks/requirements.txt); without it the
JSON is still written and PNGs are skipped with a hint.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import psutil

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DEFAULT_INPUT = HERE / "bigbugsbunny.mp4"
RESULTS_DIR = HERE / "results"
TMP_DIR = RESULTS_DIR / "tmp"

# Each group: the --<enable> store_true flag, the --<x>_method flag, and the methods
# (one model/backend per bar). Edit `methods` to tweak what gets benchmarked.
BENCHMARKS = {
    "interpolate": {
        "enable_flag": "--interpolate",
        "method_flag": "--interpolate_method",
        "methods": ["rife4.6", "rife4.6-directml", "rife4.6-tensorrt"],
    },
    "upscale": {
        "enable_flag": "--upscale",
        "method_flag": "--upscale_method",
        "methods": ["adore", "shufflecugan-directml", "rtmosr-tensorrt"],
    },
    "depth": {
        "enable_flag": "--depth",
        "method_flag": "--depth_method",
        "methods": ["small_v2", "small_v2-directml", "small_v2-tensorrt"],
    },
}

# metric -> (pretty label, True if higher-is-better)
METRICS = {
    "fps": ("FPS", True),
    "ssim": ("SSIM (vs input)", True),
    "psnr": ("PSNR dB (vs input)", True),
    "mse": ("MSE (vs input)", False),
    "vram_mb": ("Peak VRAM (MB)", False),
    "cpu_pct": ("Mean CPU (%)", False),
}

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
# Standard pipeline prints "... FPS: 56.78"; specialized ops (depth) only render the
# progress bar column "FPS  101.76" (no colon). Try the explicit line first, else the
# last bar reading (cumulative final throughput).
FPS_RE = re.compile(r"FPS:\s*([0-9]+\.?[0-9]*)")
BAR_FPS_RE = re.compile(r"FPS\s+([0-9]+\.?[0-9]*)")
TIME_RE = re.compile(r"Total Execution Time:\s*([0-9]+\.?[0-9]*)")


def parse_fps(text: str) -> float:
    m = FPS_RE.search(text)
    if m:
        return float(m.group(1))
    bars = BAR_FPS_RE.findall(text)
    return float(bars[-1]) if bars else math.nan


# --------------------------------------------------------------------------- #
# Resource monitor (VRAM via nvidia-smi per-pid, CPU via psutil over the tree)
# --------------------------------------------------------------------------- #
class ResourceMonitor(threading.Thread):
    def __init__(self, root_pid: int, interval: float = 0.25):
        super().__init__(daemon=True)
        self.root_pid = root_pid
        self.interval = interval
        self._stop = threading.Event()
        self.cpu_samples: list[float] = []
        self.perpid_samples: list[float] = []  # per-process VRAM (unavailable on WDDM)
        self.total_samples: list[float] = []  # total GPU mem used
        self._nvsmi = self._find_nvsmi()
        # Baseline total VRAM before the process allocates -> attribute the delta to it.
        base = self._query_total()
        self.baseline_mb = base if base is not None else 0.0

    @staticmethod
    def _find_nvsmi() -> str | None:
        from shutil import which

        return which("nvidia-smi")

    def _query_total(self) -> float | None:
        if not self._nvsmi:
            return None
        try:
            out = subprocess.run(
                [
                    self._nvsmi,
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout
        except subprocess.SubprocessError, OSError:
            return None
        for line in out.splitlines():
            try:
                return float(line.strip())
            except ValueError:
                continue
        return None

    def _pids(self) -> set[int]:
        try:
            root = psutil.Process(self.root_pid)
        except psutil.Error:
            return set()
        pids = {self.root_pid}
        try:
            pids |= {c.pid for c in root.children(recursive=True)}
        except psutil.Error:
            pass
        return pids

    def _query_perpid(self, pids: set[int]) -> float | None:
        """Sum per-process VRAM for our pid tree. None if nvidia-smi missing or no
        matching row (per-process memory is hidden on Windows WDDM)."""
        if not self._nvsmi:
            return None
        try:
            out = subprocess.run(
                [
                    self._nvsmi,
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout
        except subprocess.SubprocessError, OSError:
            return None
        total = 0.0
        found = False
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 2:
                continue
            try:
                pid, mem = int(parts[0]), float(parts[1])
            except ValueError:
                continue
            if pid in pids:
                total += mem
                found = True
        return total if found else None

    def run(self) -> None:
        # Prime cpu_percent counters (first call always returns 0.0).
        procs: dict[int, psutil.Process] = {}
        for pid in self._pids():
            try:
                p = psutil.Process(pid)
                p.cpu_percent(None)
                procs[pid] = p
            except psutil.Error:
                pass

        while not self._stop.wait(self.interval):
            pids = self._pids()
            for pid in pids:
                if pid not in procs:
                    try:
                        p = psutil.Process(pid)
                        p.cpu_percent(None)
                        procs[pid] = p
                    except psutil.Error:
                        pass
            cpu = 0.0
            for pid, p in list(procs.items()):
                try:
                    cpu += p.cpu_percent(None)
                except psutil.Error:
                    procs.pop(pid, None)
            self.cpu_samples.append(cpu)
            pp = self._query_perpid(pids)
            if pp is not None:
                self.perpid_samples.append(pp)
            tot = self._query_total()
            if tot is not None:
                self.total_samples.append(tot)

    def stop(self) -> dict:
        self._stop.set()
        self.join(timeout=5)
        # Prefer real per-process VRAM; else fall back to total-used delta vs baseline.
        if self.perpid_samples:
            vram = float(np.max(self.perpid_samples))
        elif self.total_samples:
            vram = max(0.0, float(np.max(self.total_samples)) - self.baseline_mb)
        else:
            vram = math.nan
        return {
            "cpu_pct": float(np.mean(self.cpu_samples))
            if self.cpu_samples
            else math.nan,
            "cpu_pct_peak": float(np.max(self.cpu_samples))
            if self.cpu_samples
            else math.nan,
            "vram_mb": vram,
        }


# --------------------------------------------------------------------------- #
# Quality metrics vs base input
# --------------------------------------------------------------------------- #
def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Single-channel SSIM (Wang et al.), 11x11 Gaussian, sigma 1.5, on [0,255]."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    k, s = (11, 11), 1.5
    mu1 = cv2.GaussianBlur(a, k, s)
    mu2 = cv2.GaussianBlur(b, k, s)
    mu1_sq, mu2_sq, mu1mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sig1 = cv2.GaussianBlur(a * a, k, s) - mu1_sq
    sig2 = cv2.GaussianBlur(b * b, k, s) - mu2_sq
    sig12 = cv2.GaussianBlur(a * b, k, s) - mu1mu2
    ssim_map = ((2 * mu1mu2 + c1) * (2 * sig12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sig1 + sig2 + c2)
    )
    return float(ssim_map.mean())


def _frame_count(cap: cv2.VideoCapture) -> int:
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return n if n > 0 else 0


def _read_gray_at(path: Path, indices: set[int]) -> dict[int, np.ndarray]:
    """Sequentially decode `path`, keeping grayscale frames whose index is in `indices`."""
    cap = cv2.VideoCapture(str(path))
    kept: dict[int, np.ndarray] = {}
    idx, want = 0, max(indices) if indices else -1
    while idx <= want:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in indices:
            kept[idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        idx += 1
    cap.release()
    return kept


def quality_vs_input(input_path: Path, output_path: Path, n_samples: int = 30) -> dict:
    """SSIM/PSNR/MSE of output frames vs nearest input frame (output resized to input res)."""
    nan = {"ssim": math.nan, "psnr": math.nan, "mse": math.nan}
    if not output_path.exists():
        return nan

    cin = cv2.VideoCapture(str(input_path))
    cout = cv2.VideoCapture(str(output_path))
    n_in, n_out = _frame_count(cin), _frame_count(cout)
    in_w = int(cin.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cin.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cin.release()
    cout.release()
    if n_in <= 0 or n_out <= 0 or in_w <= 0 or in_h <= 0:
        return nan

    # Sample evenly across the OUTPUT; map each to nearest INPUT frame by position.
    k = min(n_samples, n_out)
    out_idx = sorted({int(round(i * (n_out - 1) / max(k - 1, 1))) for i in range(k)})
    pairs = [(oi, int(round((oi / max(n_out - 1, 1)) * (n_in - 1)))) for oi in out_idx]

    out_frames = _read_gray_at(output_path, {oi for oi, _ in pairs})
    in_frames = _read_gray_at(input_path, {ii for _, ii in pairs})

    ssims, psnrs, mses = [], [], []
    for oi, ii in pairs:
        of, inf = out_frames.get(oi), in_frames.get(ii)
        if of is None or inf is None:
            continue
        if of.shape != inf.shape:
            of = cv2.resize(of, (in_w, in_h), interpolation=cv2.INTER_AREA)
        mse = float(np.mean((of.astype(np.float64) - inf.astype(np.float64)) ** 2))
        mses.append(mse)
        psnrs.append(99.0 if mse == 0 else 10.0 * math.log10((255.0**2) / mse))
        ssims.append(_ssim(of, inf))

    if not ssims:
        return nan
    return {
        "ssim": float(np.mean(ssims)),
        "psnr": float(np.mean(psnrs)),
        "mse": float(np.mean(mses)),
    }


# --------------------------------------------------------------------------- #
# Run one method
# --------------------------------------------------------------------------- #
def run_method(
    group: str, cfg: dict, method: str, input_path: Path, warmup: int, n_samples: int
) -> dict:
    safe = re.sub(r"[^a-zA-Z0-9._-]", "_", f"{group}_{method}")
    out_path = TMP_DIR / f"{safe}.mp4"
    base_cmd = [
        sys.executable,
        "main.py",
        "--input",
        str(input_path),
        "--output",
        str(out_path),
        cfg["enable_flag"],
        cfg["method_flag"],
        method,
    ]

    # Warmup runs (build TRT engines, warm cudnn/caches) — timing discarded.
    for _ in range(warmup):
        if out_path.exists():
            out_path.unlink()
        subprocess.run(
            base_cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
        )

    if out_path.exists():
        out_path.unlink()

    # utf-8/replace: main.py emits colored/unicode logs that crash the default cp1252 decode.
    proc = subprocess.Popen(
        base_cmd,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
    )
    mon = ResourceMonitor(proc.pid)
    mon.start()
    out_text = ANSI_RE.sub("", proc.communicate()[0] or "")
    res = mon.stop()
    ok = proc.returncode == 0

    fps = parse_fps(out_text)
    elapsed = (
        float(TIME_RE.search(out_text).group(1))
        if TIME_RE.search(out_text)
        else math.nan
    )

    quality = (
        quality_vs_input(input_path, out_path, n_samples)
        if ok
        else {"ssim": math.nan, "psnr": math.nan, "mse": math.nan}
    )

    if not ok:
        tail = "\n".join(out_text.strip().splitlines()[-5:])
        print(f"  ! {method} FAILED (exit {proc.returncode}). Last lines:\n{tail}")

    return {
        "method": method,
        "ok": ok,
        "exit_code": proc.returncode,
        "fps": fps,
        "elapsed_s": elapsed,
        **quality,
        "vram_mb": res["vram_mb"],
        "cpu_pct": res["cpu_pct"],
        "cpu_pct_peak": res["cpu_pct_peak"],
    }


# --------------------------------------------------------------------------- #
# Graphing
# --------------------------------------------------------------------------- #
def make_graph(group: str, rows: list[dict], out_png: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    methods = [r["method"] for r in rows]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"TAS benchmark — {group}", fontsize=15, fontweight="bold")

    for ax, (key, (label, higher)) in zip(axes.flat, METRICS.items(), strict=False):
        vals = [r.get(key, math.nan) for r in rows]
        plot_vals = [0 if (v is None or math.isnan(v)) else v for v in vals]
        bars = ax.bar(methods, plot_vals, color="#4C72B0")
        ax.set_title(
            label + ("  (higher better)" if higher else "  (lower better)"), fontsize=10
        )
        ax.tick_params(axis="x", labelrotation=20, labelsize=8)
        for b, v in zip(bars, vals, strict=False):
            txt = (
                "FAIL"
                if (v is None or math.isnan(v))
                else (f"{v:.3f}" if abs(v) < 100 else f"{v:.0f}")
            )
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height(),
                txt,
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="TAS end-to-end benchmark")
    ap.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input clip (default: benchmarks/bigbugsbunny.mp4)",
    )
    ap.add_argument(
        "--groups",
        nargs="+",
        choices=list(BENCHMARKS),
        default=list(BENCHMARKS),
        help="Which capability groups to run",
    )
    ap.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup runs per method (TRT engine build/cudnn warm); default 1",
    )
    ap.add_argument(
        "--samples",
        type=int,
        default=30,
        help="Frames sampled per video for quality metrics",
    )
    args = ap.parse_args()

    if not args.input.exists():
        print(
            f"Input not found: {args.input}\nRun: python benchmarks/download_sample.py"
        )
        return 1

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, list[dict]] = {}
    graph_ok = True
    for group in args.groups:
        cfg = BENCHMARKS[group]
        print(f"\n=== {group} ===")
        rows = []
        for method in cfg["methods"]:
            print(f"- {method} ...")
            row = run_method(group, cfg, method, args.input, args.warmup, args.samples)
            tag = "ok" if row["ok"] else "FAIL"
            print(
                f"    [{tag}] fps={row['fps']:.2f} ssim={row['ssim']:.4f} "
                f"psnr={row['psnr']:.2f} mse={row['mse']:.2f} "
                f"vram={row['vram_mb']:.0f}MB cpu={row['cpu_pct']:.0f}%"
            )
            rows.append(row)
        all_results[group] = rows
        if not make_graph(group, rows, RESULTS_DIR / f"{group}.png"):
            graph_ok = False

    json_path = RESULTS_DIR / "results.json"
    json_path.write_text(
        json.dumps(
            {
                "input": str(args.input),
                "timestamp": time.time(),
                "results": all_results,
            },
            indent=2,
        )
    )
    print(f"\nWrote {json_path}")
    if graph_ok:
        print(
            f"Wrote graphs: {', '.join(g + '.png' for g in args.groups)} in {RESULTS_DIR}"
        )
    else:
        print(
            "matplotlib missing -> graphs skipped. "
            "pip install -r benchmarks/requirements.txt"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
