"""One definition of "did this run actually produce its output?".

`main.py:_videoFailed` and `src/server/aeComms.py:reportTerminalStatus` both
answer that question -- for the batch exit code and for the After Effects
panel -- and they have to agree, or a run is reported one way to the shell and
the other way to the UI.
"""

import logging
import os


def sequenceWrote(patternPath: str) -> bool:
    """Whether an image2 pattern like ``frames_%05d.png`` produced any frame.

    FFmpeg expands the ``%05d`` itself, so the pattern is never a real path and
    ``os.path.getsize`` on it always raises. Look in its directory instead.
    """
    directory = os.path.dirname(patternPath) or "."
    prefix, _, suffix = os.path.basename(patternPath).partition("%")
    suffix = os.path.splitext(suffix)[1]
    try:
        entries = os.listdir(directory)
    except OSError:
        return False
    for name in entries:
        if not name.startswith(prefix) or not name.endswith(suffix):
            continue
        try:
            if os.path.getsize(os.path.join(directory, name)) > 0:
                return True
        except OSError:
            continue
    return False


def truncatedDecodeError(readBuffer, expectedFrames: int):
    """The decode error to blame for a short output, or ``None``.

    A decode that dies part-way can only signal it by putting its end-of-stream
    sentinel, which reads exactly like a clean EOF, so every consumer used to
    finish "successfully" on a truncated file. But an error raised *after* the
    last frame was queued -- a decoder teardown error, a corrupt trailing
    packet -- leaves a complete, correct output, and failing that run would be
    a lie in the other direction. Only frames actually missing count.
    """
    error = getattr(readBuffer, "decodeError", None)
    if error is None:
        return None
    delivered = getattr(readBuffer, "_emittedFrames", 0)
    if expectedFrames and delivered >= expectedFrames:
        logging.warning(
            f"Decoder raised after delivering {delivered}/{expectedFrames} "
            f"frames; output is complete: {error}"
        )
        return None
    return error


def outputWasWritten(outputPath: str) -> bool:
    """Whether the run left something at ``outputPath``.

    Handles the image-sequence case, where the path is an FFmpeg pattern
    rather than a file.
    """
    if "%" in os.path.basename(outputPath or ""):
        return sequenceWrote(outputPath)
    try:
        return os.path.getsize(outputPath) > 0
    except OSError, TypeError:
        return False
