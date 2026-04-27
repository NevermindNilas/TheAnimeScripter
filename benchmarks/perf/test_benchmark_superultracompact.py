"""Benchmark SuperUltraCompact 2x upscaler PyTorch CUDA forward pass.

Exercises the per-frame `__call__` path of `UniversalPytorch` with the
`superultracompact` model at common input resolutions and dtypes.

Model load (weight download + warmup + CUDA-graph capture) is amortized
across iterations via a module-scoped fixture so the benchmark only times
the steady-state inference call.
"""

from __future__ import annotations

import pytest
import torch

from src.unifiedUpscale import UniversalPytorch

pytestmark = pytest.mark.cuda

_DEVICE = "cuda"

# (height, width) test matrix — covers SD/HD/FHD on the same arch.
_RES_720P = (720, 1280)
_RES_1080P = (1080, 1920)
_RES_540P = (540, 960)


def _make_frame(height: int, width: int, dtype: torch.dtype) -> torch.Tensor:
    """Allocate a random RGB frame in the layout the upscaler expects."""
    return (
        torch.rand((1, 3, height, width), device=_DEVICE, dtype=dtype)
        .contiguous()
        .to(memory_format=torch.channels_last)
    )


def _build_upscaler(width: int, height: int, half: bool) -> UniversalPytorch:
    return UniversalPytorch(
        upscaleMethod="superultracompact",
        upscaleFactor=2,
        half=half,
        width=width,
        height=height,
        compileMode="default",
    )


@pytest.fixture(scope="module")
def upscaler_540p_fp16() -> UniversalPytorch:
    h, w = _RES_540P
    return _build_upscaler(width=w, height=h, half=True)


@pytest.fixture(scope="module")
def upscaler_720p_fp16() -> UniversalPytorch:
    h, w = _RES_720P
    return _build_upscaler(width=w, height=h, half=True)


@pytest.fixture(scope="module")
def upscaler_1080p_fp16() -> UniversalPytorch:
    h, w = _RES_1080P
    return _build_upscaler(width=w, height=h, half=True)


@pytest.fixture(scope="module")
def upscaler_1080p_fp32() -> UniversalPytorch:
    h, w = _RES_1080P
    return _build_upscaler(width=w, height=h, half=False)


def _run_upscale(upscaler: UniversalPytorch, frame: torch.Tensor) -> None:
    """One end-to-end inference call. The class syncs internally."""
    upscaler(frame, None)


def test_benchmark_superultracompact_540p_fp16(benchmark, upscaler_540p_fp16) -> None:
    """960x540 fp16 -> 1920x1080 fp16."""
    h, w = _RES_540P
    frame = _make_frame(h, w, torch.float16)
    benchmark.pedantic(
        _run_upscale,
        args=(upscaler_540p_fp16, frame),
        iterations=5,
        rounds=20,
        warmup_rounds=3,
    )


def test_benchmark_superultracompact_720p_fp16(benchmark, upscaler_720p_fp16) -> None:
    """1280x720 fp16 -> 2560x1440 fp16."""
    h, w = _RES_720P
    frame = _make_frame(h, w, torch.float16)
    benchmark.pedantic(
        _run_upscale,
        args=(upscaler_720p_fp16, frame),
        iterations=5,
        rounds=20,
        warmup_rounds=3,
    )


def test_benchmark_superultracompact_1080p_fp16(benchmark, upscaler_1080p_fp16) -> None:
    """1920x1080 fp16 -> 3840x2160 fp16 (the typical 2x anime upscale path)."""
    h, w = _RES_1080P
    frame = _make_frame(h, w, torch.float16)
    benchmark.pedantic(
        _run_upscale,
        args=(upscaler_1080p_fp16, frame),
        iterations=3,
        rounds=15,
        warmup_rounds=3,
    )


def test_benchmark_superultracompact_1080p_fp32(benchmark, upscaler_1080p_fp32) -> None:
    """1920x1080 fp32 -> 3840x2160 fp32 — the slow-path baseline."""
    h, w = _RES_1080P
    frame = _make_frame(h, w, torch.float32)
    benchmark.pedantic(
        _run_upscale,
        args=(upscaler_1080p_fp32, frame),
        iterations=3,
        rounds=10,
        warmup_rounds=3,
    )
