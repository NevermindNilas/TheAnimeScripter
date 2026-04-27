"""Benchmark RIFE 4.6 frame interpolation PyTorch CUDA forward pass.

Exercises the per-frame `__call__` path of `RifeCuda` with the `rife4.6`
model at common resolutions. The first call into RifeCuda is a no-op
priming step (it only fills the I0 buffer), so the fixture primes the
model and the benchmark times the steady-state I1/infer/cache cycle.

`framesToInsert=1` mirrors a 2x interpolation factor — one synthesized
midframe per source pair, the most common real workload.
"""

from __future__ import annotations

import queue

import pytest
import torch

from src.unifiedInterpolate import RifeCuda

pytestmark = pytest.mark.cuda

_DEVICE = "cuda"

_RES_720P = (720, 1280)
_RES_1080P = (1080, 1920)


def _make_frame(height: int, width: int, dtype: torch.dtype) -> torch.Tensor:
    """Allocate a random RGB frame for rife. Channels-last is what padFrame produces."""
    return (
        torch.rand((1, 3, height, width), device=_DEVICE, dtype=dtype)
        .contiguous()
        .to(memory_format=torch.channels_last)
    )


def _build_rife(width: int, height: int, half: bool) -> RifeCuda:
    rife = RifeCuda(
        half=half,
        width=width,
        height=height,
        interpolateMethod="rife4.6",
        ensemble=False,
        interpolateFactor=2,
        dynamicScale=False,
        staticStep=False,
        compileMode="default",
    )
    # Prime: first call only fills I0 and returns None (firstRun gate).
    dtype = torch.float16 if half else torch.float32
    primer = _make_frame(height, width, dtype)
    rife(primer, queue.Queue(), framesToInsert=1)
    return rife


@pytest.fixture(scope="module")
def rife_720p_fp16() -> RifeCuda:
    h, w = _RES_720P
    return _build_rife(width=w, height=h, half=True)


@pytest.fixture(scope="module")
def rife_1080p_fp16() -> RifeCuda:
    h, w = _RES_1080P
    return _build_rife(width=w, height=h, half=True)


@pytest.fixture(scope="module")
def rife_1080p_fp32() -> RifeCuda:
    h, w = _RES_1080P
    return _build_rife(width=w, height=h, half=False)


def _run_interp(rife: RifeCuda, frame: torch.Tensor, q: "queue.Queue") -> None:
    """One I1/infer/cache cycle producing `framesToInsert` synthesized frames."""
    rife(frame, q, framesToInsert=1)
    while not q.empty():
        q.get_nowait()


def test_benchmark_rife46_720p_fp16(benchmark, rife_720p_fp16) -> None:
    """1280x720 fp16, 1 inserted midframe per call (2x interpolation)."""
    h, w = _RES_720P
    frame = _make_frame(h, w, torch.float16)
    q: "queue.Queue" = queue.Queue()
    benchmark.pedantic(
        _run_interp,
        args=(rife_720p_fp16, frame, q),
        iterations=5,
        rounds=20,
        warmup_rounds=3,
    )


def test_benchmark_rife46_1080p_fp16(benchmark, rife_1080p_fp16) -> None:
    """1920x1080 fp16, 1 inserted midframe per call (2x interpolation)."""
    h, w = _RES_1080P
    frame = _make_frame(h, w, torch.float16)
    q: "queue.Queue" = queue.Queue()
    benchmark.pedantic(
        _run_interp,
        args=(rife_1080p_fp16, frame, q),
        iterations=3,
        rounds=15,
        warmup_rounds=3,
    )


def test_benchmark_rife46_1080p_fp32(benchmark, rife_1080p_fp32) -> None:
    """1920x1080 fp32 — slow-path baseline for relative comparison."""
    h, w = _RES_1080P
    frame = _make_frame(h, w, torch.float32)
    q: "queue.Queue" = queue.Queue()
    benchmark.pedantic(
        _run_interp,
        args=(rife_1080p_fp32, frame, q),
        iterations=3,
        rounds=10,
        warmup_rounds=3,
    )


def test_benchmark_rife46_720p_fp16_4x(benchmark, rife_720p_fp16) -> None:
    """1280x720 fp16, 3 inserted midframes per call (4x interpolation factor)."""
    h, w = _RES_720P
    frame = _make_frame(h, w, torch.float16)
    q: "queue.Queue" = queue.Queue()

    def _run_4x() -> None:
        rife_720p_fp16(frame, q, framesToInsert=3)
        while not q.empty():
            q.get_nowait()

    benchmark.pedantic(_run_4x, iterations=3, rounds=15, warmup_rounds=3)
