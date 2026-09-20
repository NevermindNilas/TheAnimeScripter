"""AMD ROCm (HIP) depth backends. Eager, no CUDA graphs.

Each class strips the "-rocm" suffix and delegates to its CUDA parent on the
torch.cuda (HIP) device. Depth CUDA paths use private streams but no graph
capture, so the parents already run eager; HIP streams are functional, and the
drivers synchronize the same way. compileMode is ignored like the MPS backends
(torch.compile on ROCm is unstable).
"""

import logging

from src.depth.backends.cuda import DepthCuda, LimboCuda, OGDepthV2CUDA, OGDepthV3Cuda
from src.depth.backends.da3_streaming import DA3StreamingCuda, LimboStreamingCuda
from src.depth.backends.video import VideoDepthAnythingCUDA, VideoDepthAnythingTorch
from src.infra.logAndPrint import logAndPrint


def _stripRocm(method: str) -> str:
    return method.replace("-rocm", "")


def _ignoreCompileMode(compileMode: str, method: str) -> str:
    if compileMode != "default":
        logAndPrint(
            f"compileMode '{compileMode}' ignored on ROCm depth backend ({method}).",
            "yellow",
        )
        return "default"
    return compileMode


class DepthROCm(DepthCuda):
    """small_v2-rocm."""

    def __init__(
        self,
        input,
        output,
        width,
        height,
        fps,
        half,
        inpoint=0,
        outpoint=0,
        encode_method="x264",
        depth_method="small_v2-rocm",
        benchmark=False,
        totalFrames=0,
        bitDepth="16bit",
        depthQuality="high",
        compileMode="default",
        depth_batch=1,
    ):
        super().__init__(
            input,
            output,
            width,
            height,
            fps,
            half,
            inpoint,
            outpoint,
            encode_method,
            _stripRocm(depth_method),
            benchmark,
            totalFrames,
            bitDepth,
            depthQuality,
            _ignoreCompileMode(compileMode, depth_method),
            depth_batch,
        )


class OGDepthV2ROCm(OGDepthV2CUDA):
    """og_small_v2-rocm."""

    def __init__(
        self,
        input,
        output,
        width,
        height,
        fps,
        half,
        inpoint=0,
        outpoint=0,
        encode_method="x264",
        depth_method="og_small_v2-rocm",
        benchmark=False,
        totalFrames=0,
        bitDepth="16bit",
        depthQuality="high",
        compileMode="default",
        depth_batch=1,
    ):
        super().__init__(
            input,
            output,
            width,
            height,
            fps,
            half,
            inpoint,
            outpoint,
            encode_method,
            _stripRocm(depth_method),
            benchmark,
            totalFrames,
            bitDepth,
            depthQuality,
            _ignoreCompileMode(compileMode, depth_method),
            depth_batch,
        )


class VideoDepthAnythingROCm(VideoDepthAnythingCUDA):
    """og_video_small_v2-rocm."""

    def __init__(
        self,
        input,
        output,
        width,
        height,
        fps,
        half,
        inpoint=0,
        outpoint=0,
        encode_method="x264",
        depth_method="og_video_small_v2-rocm",
        benchmark=False,
        totalFrames=0,
        bitDepth="16bit",
        depthQuality="high",
        compileMode="default",
    ):
        super().__init__(
            input,
            output,
            width,
            height,
            fps,
            half,
            inpoint,
            outpoint,
            encode_method,
            _stripRocm(depth_method),
            benchmark,
            totalFrames,
            bitDepth,
            depthQuality,
            _ignoreCompileMode(compileMode, depth_method),
        )


class VideoDepthTorchROCm(VideoDepthAnythingTorch):
    """video_small_v2-rocm."""

    def __init__(
        self,
        input,
        output,
        width,
        height,
        fps,
        half,
        inpoint=0,
        outpoint=0,
        encode_method="x264",
        depth_method="video_small_v2-rocm",
        benchmark=False,
        totalFrames=0,
        bitDepth="16bit",
        depthQuality="high",
        compileMode="default",
        depth_window=32,
    ):
        super().__init__(
            input,
            output,
            width,
            height,
            fps,
            half,
            inpoint,
            outpoint,
            encode_method,
            _stripRocm(depth_method),
            benchmark,
            totalFrames,
            bitDepth,
            depthQuality,
            _ignoreCompileMode(compileMode, depth_method),
            depth_window,
        )


class DA3StreamingROCm(DA3StreamingCuda):
    """video_small_v3-rocm / video_base_v3-rocm."""

    def __init__(self, *args, depth_window=32, **kwargs):
        method = kwargs.get("depth_method", "")
        if method:
            kwargs["depth_method"] = _stripRocm(method)
            kwargs["compileMode"] = _ignoreCompileMode(
                kwargs.get("compileMode", "default"), method
            )
        else:
            args = tuple(
                _stripRocm(a) if isinstance(a, str) and a.endswith("-rocm") else a
                for a in args
            )
        super().__init__(*args, depth_window=depth_window, **kwargs)


class LimboStreamingROCm(LimboStreamingCuda):
    """video_limbo-rocm / video_limbo_v2-rocm."""

    def __init__(self, *args, depth_window=32, **kwargs):
        method = kwargs.get("depth_method", "")
        if method:
            kwargs["depth_method"] = _stripRocm(method)
            kwargs["compileMode"] = _ignoreCompileMode(
                kwargs.get("compileMode", "default"), method
            )
        else:
            args = tuple(
                _stripRocm(a) if isinstance(a, str) and a.endswith("-rocm") else a
                for a in args
            )
        super().__init__(*args, depth_window=depth_window, **kwargs)


class OGDepthV3ROCm(OGDepthV3Cuda):
    """small_v3-rocm / base_v3-rocm / large_v3-rocm / og_large_v3-rocm."""

    def __init__(
        self,
        input,
        output,
        width,
        height,
        fps,
        half,
        inpoint=0,
        outpoint=0,
        encode_method="x264",
        depth_method="small_v3-rocm",
        benchmark=False,
        totalFrames=0,
        bitDepth="16bit",
        depthQuality="high",
        compileMode="default",
        depth_batch=1,
    ):
        super().__init__(
            input,
            output,
            width,
            height,
            fps,
            half,
            inpoint,
            outpoint,
            encode_method,
            _stripRocm(depth_method),
            benchmark,
            totalFrames,
            bitDepth,
            depthQuality,
            _ignoreCompileMode(compileMode, depth_method),
            depth_batch,
        )


class LimboROCm(LimboCuda):
    """limbo-rocm / limbo_v2-rocm."""

    def __init__(
        self,
        input,
        output,
        width,
        height,
        fps,
        half,
        inpoint=0,
        outpoint=0,
        encode_method="x264",
        depth_method="limbo-rocm",
        benchmark=False,
        totalFrames=0,
        bitDepth="16bit",
        depthQuality="high",
        compileMode="default",
        depth_batch=1,
    ):
        # LimboCuda.handleModels derives the checkpoint from the base name;
        # strip first so "limbo-rocm" resolves to the "limbo" weights.
        base = _stripRocm(depth_method)
        logging.info(f"ROCm depth {depth_method} resolving to {base} weights")
        super().__init__(
            input,
            output,
            width,
            height,
            fps,
            half,
            inpoint,
            outpoint,
            encode_method,
            base,
            benchmark,
            totalFrames,
            bitDepth,
            depthQuality,
            _ignoreCompileMode(compileMode, depth_method),
            depth_batch,
        )
