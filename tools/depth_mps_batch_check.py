#!/usr/bin/env python3
"""Run correctness and throughput checks for MPS depth batching.

This script exercises representative MPS image-depth methods at depth_batch=1
vs larger batch sizes, records wall-clock throughput, and compares decoded
output frames against the batch-1 baseline.

It is intended for local Apple Silicon validation after implementing the MPS
depth batching backend.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

DEFAULT_METHODS = [
    "small_v2-mps",
    "distill_small_v2-mps",
    "og_small_v2-mps",
    "small_v3-mps",
]
DEFAULT_QUALITIES = ["low", "medium", "high"]
DEFAULT_BATCHES = [1, 4]


@dataclass
class RunResult:
    method: str
    quality: str
    batch: int
    outpoint: float
    output_path: str
    wall_seconds: float
    frame_count: int
    width: int
    height: int
    fps: float


@dataclass
class CompareResult:
    method: str
    quality: str
    baseline_batch: int
    candidate_batch: int
    frame_count_match: bool
    resolution_match: bool
    compared_frames: int
    max_abs_diff: int
    mean_abs_diff: float


def _video_info(path: Path) -> tuple[int, int, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return frame_count, width, height


def _run_depth(
    repo_root: Path,
    input_path: Path,
    output_path: Path,
    method: str,
    quality: str,
    batch: int,
    outpoint: float,
    encode_method: str,
) -> RunResult:
    cmd = [
        sys.executable,
        "main.py",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--depth",
        "--depth_method",
        method,
        "--depth_quality",
        quality,
        "--depth_batch",
        str(batch),
        "--encode_method",
        encode_method,
        "--bit_depth",
        "8bit",
        "--outpoint",
        str(outpoint),
    ]

    start = time.perf_counter()
    subprocess.run(cmd, cwd=repo_root, check=True)
    wall_seconds = time.perf_counter() - start
    frame_count, width, height = _video_info(output_path)
    fps = frame_count / wall_seconds if wall_seconds > 0 else 0.0
    return RunResult(
        method=method,
        quality=quality,
        batch=batch,
        outpoint=outpoint,
        output_path=str(output_path),
        wall_seconds=wall_seconds,
        frame_count=frame_count,
        width=width,
        height=height,
        fps=fps,
    )


def _compare_videos(
    baseline_path: Path,
    candidate_path: Path,
    method: str,
    quality: str,
    baseline_batch: int,
    candidate_batch: int,
) -> CompareResult:
    base_cap = cv2.VideoCapture(str(baseline_path))
    cand_cap = cv2.VideoCapture(str(candidate_path))
    if not base_cap.isOpened() or not cand_cap.isOpened():
        raise RuntimeError(
            f"Failed to open comparison pair: {baseline_path}, {candidate_path}"
        )

    base_frames = int(base_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cand_frames = int(cand_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    base_w = int(base_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    base_h = int(base_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cand_w = int(cand_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cand_h = int(cand_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    compared = 0
    max_abs_diff = 0
    sum_mean_abs = 0.0
    while True:
        ok_a, frame_a = base_cap.read()
        ok_b, frame_b = cand_cap.read()
        if not ok_a or not ok_b:
            break
        gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
        diff = np.abs(gray_a.astype(np.int16) - gray_b.astype(np.int16))
        max_abs_diff = max(max_abs_diff, int(diff.max()))
        sum_mean_abs += float(diff.mean())
        compared += 1

    base_cap.release()
    cand_cap.release()

    return CompareResult(
        method=method,
        quality=quality,
        baseline_batch=baseline_batch,
        candidate_batch=candidate_batch,
        frame_count_match=base_frames == cand_frames,
        resolution_match=(base_w, base_h) == (cand_w, cand_h),
        compared_frames=compared,
        max_abs_diff=max_abs_diff,
        mean_abs_diff=(sum_mean_abs / compared) if compared else 0.0,
    )


def _print_summary(run_results: list[RunResult], compare_results: list[CompareResult]) -> None:
    print("\nRuns")
    for result in run_results:
        print(
            f"{result.method:24} quality={result.quality:6} "
            f"batch={result.batch:<2} frames={result.frame_count:<4} "
            f"time={result.wall_seconds:6.2f}s fps={result.fps:7.2f}"
        )

    if compare_results:
        print("\nComparisons")
        for result in compare_results:
            print(
                f"{result.method:24} quality={result.quality:6} "
                f"{result.baseline_batch}->{result.candidate_batch} "
                f"frames_ok={result.frame_count_match} res_ok={result.resolution_match} "
                f"max_abs={result.max_abs_diff:<3} mean_abs={result.mean_abs_diff:.4f}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate MPS depth batching")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("benchmarks/bigbugsbunny.mp4"),
        help="Input clip to process",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/results/depth_mps_batch"),
        help="Directory to store outputs and JSON results",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        help="Depth methods to exercise",
    )
    parser.add_argument(
        "--qualities",
        nargs="+",
        default=DEFAULT_QUALITIES,
        choices=["low", "medium", "high"],
        help="Depth quality presets to exercise",
    )
    parser.add_argument(
        "--batches",
        nargs="+",
        type=int,
        default=DEFAULT_BATCHES,
        help="Depth batch sizes to exercise; include 1 as baseline",
    )
    parser.add_argument(
        "--outpoint",
        type=float,
        default=2.0,
        help="Seconds of input to process for each run",
    )
    parser.add_argument(
        "--encode-method",
        type=str,
        default="x264",
        help="Encoder used for output videos",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    if not args.input.is_absolute():
        args.input = repo_root / args.input
    if not args.output_dir.is_absolute():
        args.output_dir = repo_root / args.output_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if 1 not in args.batches:
        raise ValueError("--batches must include 1 so comparisons have a baseline")

    run_results: list[RunResult] = []
    compare_results: list[CompareResult] = []
    outputs: dict[tuple[str, str, int], Path] = {}

    for method in args.methods:
        for quality in args.qualities:
            for batch in args.batches:
                output_path = args.output_dir / f"{method}_{quality}_b{batch}.mp4"
                print(f"Running {method} quality={quality} batch={batch} -> {output_path}")
                result = _run_depth(
                    repo_root=repo_root,
                    input_path=args.input,
                    output_path=output_path,
                    method=method,
                    quality=quality,
                    batch=batch,
                    outpoint=args.outpoint,
                    encode_method=args.encode_method,
                )
                run_results.append(result)
                outputs[(method, quality, batch)] = output_path

            baseline_path = outputs[(method, quality, 1)]
            for batch in args.batches:
                if batch == 1:
                    continue
                candidate_path = outputs[(method, quality, batch)]
                compare_results.append(
                    _compare_videos(
                        baseline_path,
                        candidate_path,
                        method=method,
                        quality=quality,
                        baseline_batch=1,
                        candidate_batch=batch,
                    )
                )

    results_json = {
        "runs": [asdict(result) for result in run_results],
        "comparisons": [asdict(result) for result in compare_results],
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(results_json, indent=2),
        encoding="utf-8",
    )

    _print_summary(run_results, compare_results)
    print(f"\nWrote {args.output_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
