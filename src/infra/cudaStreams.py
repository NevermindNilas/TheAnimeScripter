"""Ordering between the caller's CUDA stream and a stage's private streams."""

import torch


def waitForCaller(stream) -> None:
    """Make a private CUDA ``stream`` wait for the work already queued on the caller's
    current stream before it runs anything more.

    A stage that copies or reads its input on its own stream otherwise races the work
    that produced that input: a frame made on the default stream (a tensor op, a
    decoder upload, the previous stage's last kernel) can still be in flight when the
    private stream reads it, and the stage sees a half-written frame. Every stage
    happens to synchronise its own streams before handing a frame on today, which is
    why renders did not show it, but any caller that does not (the tests did not) got
    garbage. ``wait_stream`` only orders the GPU work; the host does not block.

    No-op for a stream that is None (the ROCm paths drop their private streams), or
    without CUDA.
    """
    if stream is None or not torch.cuda.is_available():
        return
    current = torch.cuda.current_stream()
    if current != stream:
        stream.wait_stream(current)
