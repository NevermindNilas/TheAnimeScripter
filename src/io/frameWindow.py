"""Push-fed lag ring for one edge of the frame pipeline.

Every stage in the processing chain receives its input as a stream of
``FrameSlot`` objects pushed from upstream. A temporal driver needs neighbouring
frames handed to it, so each stage owns a ``LagRing``: it holds the last ``past``
slots plus the ``future`` slots that have not yet been emitted, and runs a
*centre* only once its future neighbours have arrived. Drivers declare their
demand as ``temporalWindow = (past, future)`` on the class; ``temporalDemand``
reads it, and a stage with no declaration is ``(0, 0)`` and runs as soon as its
frame arrives.

Three properties beyond the window size:

**Domain.** A neighbour is only meaningful in the same processing domain as the
frame it neighbours. Each edge carries the frames of its own stream, so a
driver's neighbour is the next item on that stream: the restore-domain source
frame for a stage before interpolation, the next interpolated output for a stage
after it (see ``src/pipeline/executor.py``, option A).

**Validity.** A neighbour across a hard scene cut is not context, it is a
different shot. ``LagRing.successor`` returns ``None`` there, exactly as it does
at the end of the stream, and every temporal driver already treats
``nextFrame=None`` as "no future context".

**Order.** A centre is run exactly once, in increasing index order, so recurrent
drivers (AnimeSR carries ``prevFrame`` and a hidden ``state``) see frames in
sequence.
"""

from collections import namedtuple

Window = namedtuple("Window", "centre past following")
"""One run of a stage: the slot being processed, the slots before it (oldest
first), and the slot after it (``None`` at the end of the stream or across a
scene cut)."""


class FrameSlot:
    """One frame on a stream, plus the metadata later stages depend on.

    ``frame`` is the frame in the stream's own domain. ``isCut`` marks a hard scene
    cut *at* this frame, i.e. between it and the frame before it on the same
    stream. ``dupsBefore`` is how many source frames the smooth-dedup detector
    dropped between the previous admitted frame and this one, so the interpolation
    gap can be widened to cover them. ``sourcePos`` is the source position the frame
    was produced for. ``isCentre`` marks a frame that belongs to a source position
    (as opposed to an interpolated or held frame) and so is what a trailing
    duplicate is cloned from. ``cacheKey`` names the source slot a held copy was
    copied from; a downstream stateless upscaler reuses its output for copies that
    share a key within one interpolation gap.
    """

    __slots__ = ("frame", "isCut", "dupsBefore", "sourcePos", "isCentre", "cacheKey")

    def __init__(
        self,
        frame,
        isCut: bool = False,
        dupsBefore: int = 0,
        sourcePos: int = 0,
        isCentre: bool = False,
        cacheKey=None,
    ):
        self.frame = frame
        self.isCut = isCut
        self.dupsBefore = dupsBefore
        self.sourcePos = sourcePos
        self.isCentre = isCentre
        self.cacheKey = cacheKey


class LagRing:
    """Bounded push-fed window that releases each centre once its future is known.

    ``push`` takes the next slot of the stream and returns the ``Window`` s that
    became runnable, in order. ``flush`` is the end of the stream: every remaining
    centre is released with whatever future it has (``None`` at the very end).
    At most ``past + 1 + future`` slots are retained.
    """

    __slots__ = ("past", "future", "_buf", "_base", "_next", "_seen")

    def __init__(self, past: int = 0, future: int = 0):
        if past < 0 or future < 0:
            raise ValueError(f"window bounds must be non-negative, got {past, future}")
        self.past = past
        self.future = future
        self._buf: list = []  # slots still needed, oldest first
        self._base = 0  # absolute stream index of _buf[0]
        self._next = 0  # absolute index of the next centre to release
        self._seen = 0  # slots pushed so far

    @property
    def capacity(self) -> int:
        """Slots this edge keeps alive at once: past, centre and future."""
        return self.past + 1 + self.future

    def push(self, slot) -> list:
        self._buf.append(slot)
        self._seen += 1
        ready = []
        while self._next + self.future < self._seen:
            ready.append(self._take())
        return ready

    def flush(self) -> list:
        ready = []
        while self._next < self._seen:
            ready.append(self._take())
        return ready

    def _take(self) -> Window:
        index = self._next
        local = index - self._base
        centre = self._buf[local]
        past = tuple(self._buf[max(0, local - self.past) : local])
        following = self._successor(index)
        self._next += 1
        # Slots older than the new centre's past are no longer reachable.
        keepFrom = self._next - self.past - self._base
        if keepFrom > 0:
            del self._buf[:keepFrom]
            self._base += keepFrom
        return Window(centre, past, following)

    def _successor(self, index: int):
        """The slot after ``index``, unless the stream ended or a cut separates them."""
        following = index + 1
        if following >= self._seen:
            return None
        slot = self._buf[following - self._base]
        if slot.isCut:
            return None
        return slot


def temporalDemand(*stages) -> tuple[int, int]:
    """Widest ``(past, future)`` window any of ``stages`` asks to be handed.

    Stages that are disabled (``None``) or non-temporal contribute ``(0, 0)``.
    """
    past = future = 0
    for stage in stages:
        if stage is None:
            continue
        stagePast, stageFuture = getattr(stage, "temporalWindow", (0, 0))
        past = max(past, stagePast)
        future = max(future, stageFuture)
    return past, future
