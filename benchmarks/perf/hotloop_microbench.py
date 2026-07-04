"""Short hot-loop microbenchmarks for TAS.

The timed regions auto-calibrate to stay under 10 seconds. Setup/model load is
reported separately so perf claims can stay grounded in steady-state loop time.
"""

from __future__ import annotations

import argparse
import json
import queue
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class Sink:
    def __init__(self) -> None:
        self.count = 0

    def put(self, item) -> None:
        self.count += 1


@dataclass
class TimedRun:
    iterations: int
    elapsed_ms: float
    ms_per_iter: float
    fps_equiv: float


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _event_ms(fn, iterations: int) -> float:
    _sync()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def _wall_ms(fn, iterations: int) -> float:
    _sync()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    _sync()
    return (time.perf_counter() - start) * 1000.0


def calibrate(fn, target_ms: float = 7000.0, max_iterations: int = 240) -> TimedRun:
    iterations = 1
    elapsed = _event_ms(fn, iterations) if torch.cuda.is_available() else _wall_ms(fn, iterations)
    while elapsed < target_ms / 4 and iterations < max_iterations:
        next_iterations = min(max_iterations, iterations * 2)
        if next_iterations == iterations:
            break
        iterations = next_iterations
        elapsed = _event_ms(fn, iterations) if torch.cuda.is_available() else _wall_ms(fn, iterations)
    # If calibration overshot, scale down and remeasure once.
    if elapsed > target_ms and iterations > 1:
        iterations = max(1, int(iterations * target_ms / elapsed))
        elapsed = _event_ms(fn, iterations) if torch.cuda.is_available() else _wall_ms(fn, iterations)
    return TimedRun(
        iterations=iterations,
        elapsed_ms=elapsed,
        ms_per_iter=elapsed / iterations,
        fps_equiv=1000.0 / (elapsed / iterations),
    )


def make_frame(height: int, width: int, dtype: torch.dtype) -> torch.Tensor:
    return (
        torch.rand((1, 3, height, width), device="cuda", dtype=dtype)
        .contiguous()
        .to(memory_format=torch.channels_last)
    )


def bench_rife(args: argparse.Namespace) -> dict:
    from src.interpolate.rife import RifeCuda

    dtype = torch.float16 if args.half else torch.float32
    setup_start = time.perf_counter()
    rife = RifeCuda(
        half=args.half,
        width=args.width,
        height=args.height,
        interpolateMethod=args.method,
        ensemble=False,
        interpolateFactor=args.factor,
        dynamicScale=args.dynamic_scale,
        staticStep=args.static_step,
        compileMode="default",
    )
    frame = make_frame(args.height, args.width, dtype)
    primer = make_frame(args.height, args.width, dtype)
    rife(primer, Sink(), framesToInsert=args.factor - 1)
    _sync()
    setup_s = time.perf_counter() - setup_start

    samples = []
    for _ in range(args.rounds):
        sink = Sink()

        def step(interp_sink=sink) -> None:
            rife(frame, interp_sink, framesToInsert=args.factor - 1)

        result = calibrate(step, target_ms=args.target_ms, max_iterations=args.max_iterations)
        samples.append(result)

    return {
        "bench": "rife_call",
        "method": args.method,
        "precision": "fp16" if args.half else "fp32",
        "width": args.width,
        "height": args.height,
        "factor": args.factor,
        "setup_s": setup_s,
        "timed_runs": [run.__dict__ for run in samples],
        "median_ms_per_iter": statistics.median(run.ms_per_iter for run in samples),
        "median_fps_equiv": statistics.median(run.fps_equiv for run in samples),
        "use_graph": bool(getattr(rife, "useGraph", False)),
    }


def bench_queue(args: argparse.Namespace) -> dict:
    tensor = torch.empty((1, 3, args.height, args.width), device="cuda")

    def queue_step() -> None:
        q = queue.Queue()
        q.put(tensor)
        q.get_nowait()

    sink = Sink()

    def sink_step() -> None:
        sink.put(tensor)

    queue_run = calibrate(queue_step, target_ms=args.target_ms, max_iterations=100_000)
    sink_run = calibrate(sink_step, target_ms=args.target_ms, max_iterations=100_000)
    return {
        "bench": "queue_overhead",
        "queue": queue_run.__dict__,
        "sink": sink_run.__dict__,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="bench", required=True)

    rife = sub.add_parser("rife")
    rife.add_argument("--width", type=int, default=640)
    rife.add_argument("--height", type=int, default=360)
    rife.add_argument("--method", default="rife4.6")
    rife.add_argument("--factor", type=int, default=2)
    rife.add_argument("--half", action=argparse.BooleanOptionalAction, default=True)
    rife.add_argument("--dynamic-scale", action="store_true")
    rife.add_argument("--static-step", action="store_true")
    rife.add_argument("--rounds", type=int, default=3)
    rife.add_argument("--target-ms", type=float, default=7000.0)
    rife.add_argument("--max-iterations", type=int, default=240)
    rife.set_defaults(func=bench_rife)

    q = sub.add_parser("queue")
    q.add_argument("--width", type=int, default=640)
    q.add_argument("--height", type=int, default=360)
    q.add_argument("--target-ms", type=float, default=2000.0)
    q.set_defaults(func=bench_queue)

    args = parser.parse_args()
    result = args.func(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
