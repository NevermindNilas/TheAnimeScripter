"""Frame reading shared by the autoclip readers."""

import torch


def eightBit(frames):
    """Decoded RGB frames as uint8, the input every autoclip model is written for.

    nelux decodes a >8-bit source (Hi10P, HEVC Main10, ProRes) to uint16:
    TransNetV2's forward asserts uint8 (and its CUDA graph's uint8 input
    wrapped the values), and MaxxVIT's preprocessing divides by 255. Torch has
    no uint16 shift, hence the int32 hop; nelux's uint16 white (65283..65535)
    lands on 255.
    """
    if frames.dtype == torch.uint8:
        return frames
    return (frames.to(torch.int32) >> 8).to(torch.uint8)


def openResizedReader(path, width, height, inPoint, outPoint, resizeFilter=None):
    """A nelux frame iterator at ``width`` x ``height`` over the trimmed range.

    Returns ``(iterator, fps, startFrame, endFrame)``; ``endFrame`` comes from
    the container's frame count when no out point is set, so callers that need
    the exact frame total count what the iterator yields. ``resizeFilter`` is
    nelux's scaler name; None keeps nelux's default (bilinear).

    The iterator with prefetch is the fastest access pattern available:
    get_batch_range is blocked when resize is configured. Do NOT switch to
    ``reader.frame_at(i)`` in a loop: that path calls seekToNearestKeyframe +
    decode-forward per call, giving O(N^2)-ish behavior for sequential
    indices. ``__iter__`` (and the range-slice form below) reuse the
    persistent decoder and stay O(1) per frame.

    ``prefetch=True`` at construction is mandatory: nelux 0.10.x has a race
    between FF_THREAD_FRAME codec ctx and post-hoc ``start_prefetch()`` (the
    prefetch flag never flips, decodeFrame keeps routing to the sync path
    while the producer thread also decodes -> ``av_assert0(!fctx->async_lock)``
    at pthread_frame.c:174). Constructing with prefetch=True from the start
    avoids the dual-path race entirely.
    """
    import nelux  # after torch (module level), which nelux requires

    from src.io.getVideoMetadata import (
        isTrimUnset,
        storedResizeTarget,
        trimPointToFrames,
    )

    scaler = {} if resizeFilter is None else {"resize_filter": resizeFilter}
    reader = nelux.VideoReader(
        path,
        backend="pytorch",
        decode_accelerator="cpu",
        # Stored orientation: nelux rotates a portrait clip after scaling.
        resize=storedResizeTarget(path, width, height),
        prefetch=True,
        **scaler,
    )
    fps = reader.fps
    if not fps or fps <= 0:
        raise RuntimeError(f"Invalid FPS reported for video: {path}")

    startFrame = trimPointToFrames(fps, inPoint) if not isTrimUnset(inPoint) else 0
    endFrame = (
        trimPointToFrames(fps, outPoint)
        if not isTrimUnset(outPoint)
        else reader.frame_count
    )
    if endFrame <= startFrame:
        raise RuntimeError(f"Empty range: inPoint={inPoint}, outPoint={outPoint}")

    if startFrame > 0 or endFrame < reader.frame_count:
        iterable = reader([startFrame / fps, endFrame / fps])
    else:
        iterable = reader

    return iter(iterable), fps, startFrame, endFrame


def isConstantFrameRate(path) -> bool:
    """CFR check: r_frame_rate must equal avg_frame_rate.

    Cut timestamps derived from frame_number / average fps are only truthful
    on constant-frame-rate containers. nelux.probe() returns ffprobe's own
    rational strings for both rates. Any probe failure (unreadable file,
    missing rate) is "not provably CFR" and returns False.
    """
    import logging
    import os

    try:
        import nelux  # after torch (module level), which nelux requires

        stream = nelux.probe(os.path.normpath(path))

        def rate(expr: str) -> float:
            num, _, den = expr.partition("/")
            return float(num) / float(den) if den and float(den) else 0.0

        r, avg = rate(stream["r_frame_rate"]), rate(stream["avg_frame_rate"])
        return r > 0 and abs(r - avg) < 0.01
    except Exception as e:
        logging.info(f"CFR probe failed ({e}); treating input as VFR")
        return False


def warnIfVariableRate(path, fps):
    """Warn that frame-index cut times drift on a variable-frame-rate input."""
    if isConstantFrameRate(path):
        return
    from src.infra.logAndPrint import logAndPrint

    logAndPrint(
        f"Input is not provably constant frame rate; cut times assume "
        f"{fps:.3f} fps and drift on a variable-frame-rate source "
        "(--autoclip_method pyscenedetect reads real timestamps).",
        "yellow",
    )
