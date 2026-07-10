"""Bounded sliding window over the frame stream a driver actually consumes.

Temporal drivers (AnimeSR, DistilDRBA) need neighbouring frames, not just the
one being processed. Before this existed, ``main.py`` kept a bespoke
``self.nextFrame`` attribute that was populated behind a hardcoded method-name
check, so every backend of a temporal model had to be re-listed there by hand
(``animesr-tensorrt`` never was, and silently ran with ``next == current``).

Drivers declare ``temporalWindow = (past, future)`` and the pipeline sizes this
ring to the widest demand along its stage chain. The declaration covers only the
neighbours a driver needs *handed to it*; frames it caches internally (RIFE's
``I0``, AnimeSR's padded ``prevFrame``) are not part of it.

A window has three properties beyond its size:

**Domain.** A neighbour is only meaningful in the same processing domain as the
frame it neighbours. Frames enter the ring through the pipeline's *entry*
callback (dedup, then restore), so every slot holds a restore-domain frame, and
stages further down the chain memoize their own output per slot (see
``FrameSlot.cache``). Handing a driver a raw decoded frame as the neighbour of a
restored or upscaled one is what this replaces.

**Validity.** A neighbour across a hard scene cut is not context, it is a
different shot. ``successorFrame`` returns ``None`` there, exactly as it does at
the end of the stream. Every temporal driver already treats ``nextFrame=None`` as
"no future context" and substitutes the current frame.

**Order.** Memoized stage values are computed on the slot the first time they are
asked for, and callers only ever look forward, so computation runs in strictly
increasing frame order. Recurrent drivers (AnimeSR carries ``prevFrame`` and a
hidden ``state``) therefore still see frames in sequence.
"""


class FrameSlot:
    """One frame in the window, plus whatever later stages have computed for it.

    ``frame`` is the frame in the window's own domain (post-dedup, post-restore).
    ``isCut`` marks a hard scene cut *at* this frame, i.e. between it and the
    frame before it. ``cache`` memoizes downstream stage outputs keyed by name.
    """

    __slots__ = ("frame", "isCut", "cache")

    def __init__(self, frame, isCut: bool = False):
        self.frame = frame
        self.isCut = isCut
        self.cache: dict = {}


class FrameWindow:
    """A ring of at most ``past + 1 + future`` slots pulled from ``source``.

    ``source`` is a zero-arg callable returning the next raw frame, or ``None``
    once the stream is exhausted (``BuildBuffer.read``). It is called exactly
    once past the final frame: the reader emits a single ``None`` sentinel, so
    the window never reads again after seeing it.

    ``enter`` maps a raw frame to a ``FrameSlot``, or to ``None`` to drop it
    (dedup). It runs in decode order, once per frame, so the stateful detectors
    it wraps see exactly the sequence they would have seen inline.

    Frames are held by reference. ``BuildBuffer.processFrameToTorch`` always
    allocates (the dtype cast from uint8/uint16 is never a view), and every
    restore/upscale backend returns a fresh tensor -- it has to, since its output
    already outlives the call in the 32-deep writer queue -- so retaining slots
    cannot alias a reused buffer.
    """

    __slots__ = (
        "_source",
        "_enter",
        "_past",
        "_future",
        "_slots",
        "_centre",
        "_exhausted",
        "consumed",
        "dropped",
    )

    def __init__(self, source, past: int = 0, future: int = 0, enter=None):
        if past < 0 or future < 0:
            raise ValueError(f"window bounds must be non-negative, got {past, future}")
        self._source = source
        self._enter = enter if enter is not None else FrameSlot
        self._past = past
        self._future = future
        self._slots: list = []
        self._centre = -1  # not yet primed
        self._exhausted = False
        # Raw frames pulled from the source, and how many `enter` rejected.
        # The frame loop reports both, so dedup accounting stays decode-based.
        self.consumed = 0
        self.dropped = 0

    def _fill(self) -> None:
        while (
            not self._exhausted and len(self._slots) - self._centre - 1 < self._future
        ):
            frame = self._source()
            if frame is None:
                self._exhausted = True
                return
            self.consumed += 1
            slot = self._enter(frame)
            if slot is None:
                self.dropped += 1
                continue
            self._slots.append(slot)

    def advance(self) -> bool:
        """Move the centre to the next admitted slot. False once the stream is spent."""
        if self._centre < 0:
            self._centre = 0
        else:
            self._centre += 1
        self._fill()

        if self._centre >= len(self._slots):
            return False

        # Release slots that have fallen out of the trailing edge.
        drop = self._centre - self._past
        if drop > 0:
            del self._slots[:drop]
            self._centre -= drop
        return True

    @property
    def centre(self) -> FrameSlot:
        """The slot currently being processed."""
        return self._slots[self._centre]

    def at(self, offset: int) -> FrameSlot | None:
        """Slot at ``offset`` from the centre, or ``None`` past an edge."""
        index = self._centre + offset
        if index < 0 or index >= len(self._slots):
            return None
        return self._slots[index]

    def successor(self, offset: int = 0) -> FrameSlot | None:
        """Slot following the one at ``offset``, unless a scene cut separates them.

        ``None`` means "no future context": either the stream ended, or the next
        frame opens a new shot. Drivers already handle that by falling back to
        the current frame.
        """
        following = self.at(offset + 1)
        if following is None or following.isCut:
            return None
        return following

    def successorFrame(self, offset: int = 0):
        """``successor(offset).frame``, in the window's own domain."""
        following = self.successor(offset)
        return None if following is None else following.frame

    def staged(self, offset: int, key: str, compute):
        """Memoized downstream stage output for the slot at ``offset``.

        ``compute(offset)`` runs at most once per slot. Callers only look
        forward, so slots are computed in increasing frame order and recurrent
        drivers stay in sequence.
        """
        slot = self.at(offset)
        if slot is None:
            return None
        value = slot.cache.get(key)
        if value is None:
            value = compute(offset)
            slot.cache[key] = value
        return value


def temporalDemand(*stages) -> tuple[int, int]:
    """Widest ``(past, future)`` window any of ``stages`` asks to be handed.

    Stages that are disabled (``None``) or non-temporal contribute ``(0, 0)``.
    Use this per stage; the pipeline decides whether demands compose by ``max``
    (parallel consumers of the same stream) or by sum (chained stages, where a
    stage's lookahead is itself resolved through its upstream's lookahead).
    """
    past = future = 0
    for stage in stages:
        if stage is None:
            continue
        stagePast, stageFuture = getattr(stage, "temporalWindow", (0, 0))
        past = max(past, stagePast)
        future = max(future, stageFuture)
    return past, future
