"""One definition of "did this run actually produce its output?".

`main.py:_videoFailed` and `src/server/aeComms.py:reportTerminalStatus` both
answer that question -- for the batch exit code and for the After Effects
panel -- and they have to agree, or a run is reported one way to the shell and
the other way to the UI.
"""

import logging
import os
import re

# The image2 muxer's frame counter: ``%d`` or a zero-padded ``%05d``.
_FRAME_COUNTER = re.compile(r"%(?:0(\d+))?d")


def sequenceWrote(patternPath: str) -> bool:
    """Whether an image2 pattern like ``frames_%05d.png`` produced any frame.

    FFmpeg expands the ``%05d`` itself, so the pattern is never a real path and
    ``os.path.getsize`` on it always raises. Look in its directory instead --
    but match the counter, not just the prefix and the extension, or a stray
    ``frames_old.png`` the user left in the folder reads as a written frame.
    """
    basename = os.path.basename(patternPath or "")
    counter = _FRAME_COUNTER.search(basename)
    if counter is None:
        return False
    # ``%05d`` is a minimum width rather than a fixed one: FFmpeg writes a
    # sixth digit once the counter passes 99999.
    width = int(counter.group(1) or 1)
    frame = re.compile(
        re.escape(basename[: counter.start()])
        + rf"\d{{{width},}}"
        + re.escape(basename[counter.end() :])
    )
    directory = os.path.dirname(patternPath) or "."
    try:
        entries = os.listdir(directory)
    except OSError:
        return False
    for name in entries:
        if not frame.fullmatch(name):
            continue
        try:
            if os.path.getsize(os.path.join(directory, name)) > 0:
                return True
        except OSError:
            continue
    return False


def truncatedDecodeError(readBuffer, expectedFrames: int, writeBuffer=None):
    """The pipeline error to blame for a bad output, or ``None``.

    A decode that dies part-way can only signal it by putting its end-of-stream
    sentinel, which reads exactly like a clean EOF, so every consumer used to
    finish "successfully" on a truncated file. But an error raised *after* the
    last frame was queued -- a decoder teardown error, a corrupt trailing
    packet -- leaves a complete, correct output, and failing that run would be
    a lie in the other direction. Only frames actually missing count.

    The writer has the same shape: it catches every encoding exception, logs
    it, and leaves the output file's size as the only signal. That is enough
    for a single file, which ends up 0 bytes, but not for an image sequence --
    one written frame out of five hundred looks exactly like success.
    """
    error = getattr(writeBuffer, "encodeError", None)
    if error is not None:
        return error

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
    rather than a file. A bare ``%`` does not make one: ``50%_off.mp4`` is a
    perfectly ordinary file name, and skipping its size check for a directory
    listing would report the wrong outcome for it.
    """
    if _FRAME_COUNTER.search(os.path.basename(outputPath or "")):
        return sequenceWrote(outputPath)
    try:
        return os.path.getsize(outputPath) > 0
    except OSError, TypeError:
        return False
