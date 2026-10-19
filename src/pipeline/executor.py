"""Runs a planned chain over the decoded frame stream.

Each unit of the plan becomes a node with its own input ring (``LagRing``,
see ``src/io/frameWindow.py``). Frames are pushed from the source into the head
node; a node runs a centre as soon as its declared future has arrived, and
pushes its output into the next node inside ``put``. So the order of driver
calls is the order the old engine made them in, and a temporal stage after
interpolation sees its true neighbours on its own stream (option A: the lag ring
persists across interpolation gaps, and ``end`` drains it with ``None``).

Stream contract. Every stage returns a tensor whose producing work is finished:
the adapters that cast, transfer or clone a frame synchronize the stream they
used before the frame leaves them. The writer copies on its own private stream
and never waits on the pipeline, so no writer-edge sync exists.

Cuts. A hard cut is ``FrameSlot.isCut``. A filter that drops a slot ORs its cut
into the next admitted slot. An interpolation gap across a cut is held: the
slots before the cut's final source step are copies of the previous frame, the
rest are copies of the new one, and the first copy of the new shot carries
``isCut`` so a downstream temporal stage gets no neighbour across it.
"""

import torch

from src.interpolate._timesteps import cutHoldSplit, gapPlan
from src.io.frameWindow import FrameSlot, LagRing, Window, temporalDemand
from src.pipeline.stages import DROP, isStatelessUpscale

_DEDUP_LIKE = ("dedup", "smooth-dedup")


def _syncAfter(frame) -> None:
    """Finish the work that produced ``frame`` before another stream reads it."""
    device = getattr(frame, "device", None)
    if device is None:
        return
    if device.type == "cuda":
        torch.cuda.current_stream(device).synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _defaultStream():
    """The default CUDA stream, or ``None`` when this run never initialized CUDA.

    Resolved once per pipeline: entering a stream context costs ~8 us, and a run
    on DirectML/OpenVINO/NCNN must not create a CUDA context just to ask.
    """
    if torch.cuda.is_initialized():
        return torch.cuda.default_stream()
    return None


def _toDevice(device: str):
    def move(frame):
        moved = frame.to(device)
        _syncAfter(moved)
        return moved

    return move


class _Node:
    """A stage of the chain: owns its input ring and forwards what it produces."""

    def __init__(self, name: str, window, downstream, outShape=(0, 0), inShape=None):
        past, future = window
        self.name = name
        self.ring = LagRing(past, future)
        self.downstream = downstream
        self.outShape = outShape
        # The ring holds this node's INPUT frames.
        self.inShape = outShape if inShape is None else inShape
        # A stage that sees no neighbours runs each frame as it arrives; skipping
        # the ring bookkeeping keeps the common chains at the old per-frame cost.
        self._direct = past == 0 and future == 0

    def put(self, slot) -> None:
        if self._direct:
            self._run(Window(slot, (), None))
            return
        for window in self.ring.push(slot):
            self._run(window)

    def end(self) -> None:
        for window in self.ring.flush():
            self._run(window)
        self.downstream.end()

    def _run(self, window) -> None:
        raise NotImplementedError


class FilterNode(_Node):
    """Drops the slots its detector calls duplicates.

    ``carriesDups`` (smooth dedup) hands the drop count to the next admitted slot
    as ``dupsBefore``, so the interpolation gap that follows is widened to cover
    the source frames that were dropped.
    """

    def __init__(
        self, name, detector, downstream, *, carriesDups=False, outShape=(0, 0)
    ):
        super().__init__(name, (0, 0), downstream, outShape)
        self.detector = detector
        self.carriesDups = carriesDups
        self.drops = 0
        self.pendingDups = 0
        self._pendingCut = False

    def _run(self, window) -> None:
        slot = window.centre
        verdict = self.detector(slot.frame)
        if verdict is DROP or bool(verdict):
            self.drops += 1
            if self.carriesDups:
                self.pendingDups += 1
            self._pendingCut = self._pendingCut or slot.isCut
            return
        dups = self.pendingDups if self.carriesDups else 0
        isCut = slot.isCut or self._pendingCut
        self.pendingDups = 0
        self._pendingCut = False
        self.downstream.put(
            FrameSlot(
                slot.frame,
                isCut,
                dups,
                slot.sourcePos,
                slot.isCentre,
                slot.cacheKey,
            )
        )


class MapNode(_Node):
    """Replaces each frame with the driver's output. The input is never modified.

    ``takesNext`` gives the driver its neighbour (upscalers always take one). A
    stateless upscaler with ``memo`` set serves every copy of one held source slot
    from one computation within the current interpolation gap.
    """

    def __init__(
        self, name, fn, window, downstream, *, takesNext=False, outShape=(0, 0)
    ):
        super().__init__(name, window, downstream, outShape)
        self.fn = fn
        self.takesNext = takesNext
        self.memo = None

    def _run(self, window) -> None:
        slot = window.centre
        nextFrame = None
        if self.takesNext and window.following is not None:
            nextFrame = window.following.frame
        if self.memo is not None and slot.cacheKey is not None:
            frame = self._reuse(slot, nextFrame)
        else:
            frame = self._apply(slot.frame, nextFrame)
        self.downstream.put(
            FrameSlot(frame, slot.isCut, slot.dupsBefore, slot.sourcePos, slot.isCentre)
        )

    def _apply(self, frame, nextFrame):
        out = self.fn(frame, nextFrame) if self.takesNext else self.fn(frame)
        if out is None:
            raise RuntimeError(f"the {self.name} stage returned no frame")
        return out

    def _reuse(self, slot, nextFrame):
        cached = self.memo.get(slot.cacheKey)
        if cached is None:
            cached = self._apply(slot.frame, nextFrame)
            self.memo[slot.cacheKey] = cached
        cloned = cached.clone()
        _syncAfter(cloned)
        return cloned


class AnnotateNode(_Node):
    """Scene-cut detector: marks the slot whose frame opens a new shot."""

    def __init__(
        self, name, detector, downstream, *, scoreSource=None, outShape=(0, 0)
    ):
        super().__init__(name, (0, 0), downstream, outShape)
        self.detector = detector
        self.scoreSource = scoreSource
        self.cuts = 0

    def _run(self, window) -> None:
        slot = window.centre
        score = None
        if self.scoreSource is not None:
            score = getattr(self.scoreSource, "lastReusableScore", None)
        if score is not None and getattr(self.detector, "supportsScoreReuse", False):
            isCut = bool(self.detector(slot.frame, score))
        else:
            isCut = bool(self.detector(slot.frame))
        if isCut:
            self.cuts += 1
        self.downstream.put(
            FrameSlot(
                slot.frame,
                slot.isCut or isCut,
                slot.dupsBefore,
                slot.sourcePos,
                slot.isCentre,
                slot.cacheKey,
            )
        )


class _DriverSink:
    """What an interpolation driver puts its intermediate frames into."""

    __slots__ = ("node",)

    def __init__(self, node):
        self.node = node

    def put(self, frame) -> None:
        self.node.emitFrame(frame)


class ExpandNode(_Node):
    """Interpolation: emits the frames owed to each source interval, then the centre.

    The gap for source frame ``k`` covers ``(previous, k]``, so a smooth-dedup
    gap widened by dropped duplicates gets proportionally more slots. A gap over a
    hard cut, or a hold longer than ``smoothDedupMaxSpan``, is filled with copies
    instead of inference (see ``_emitHeldGap``).
    """

    def __init__(
        self,
        name,
        driver,
        factorNum,
        factorDen,
        downstream,
        *,
        smoothDedup=False,
        smoothDedupMaxSpan=0,
        maskedSink=None,
        outShape=(0, 0),
    ):
        past, future = getattr(driver, "temporalWindow", (0, 0))
        super().__init__(name, (past, future), downstream, outShape)
        self.driver = driver
        self.factorNum = int(factorNum)
        self.factorDen = int(factorDen)
        self.smoothDedup = smoothDedup
        self.maxSpan = smoothDedupMaxSpan
        self.maskedSink = maskedSink
        # A driver that declares a future neighbour is handed it as an argument.
        self.takesNext = future > 0
        self.memos: list = []
        # A gap's frames are collected and handed on after the driver returns,
        # as the old engine's _FrameCollector did: writing each frame mid-call
        # wakes the writer thread between RIFE's timestep launches and cost
        # 1-1.3% at x4/360p. ``collectLimit`` (the writer's queue depth) bounds
        # what one gap keeps alive, so a long --smooth_dedup hold stays flat.
        self.collectLimit = 32  # set by Pipeline
        self._collected: list = []
        self.defaultStream = None  # set by Pipeline; see _flushCollected
        self._driverPin = None  # see _flushCollected
        self.sourcePos = 0
        self._lastFed = None
        self._maskAnchor = None
        self._pos = 0
        self._sink = _DriverSink(self)

    def _run(self, window) -> None:
        centre = window.centre
        nextFrame = None
        if window.following is not None:
            nextFrame = window.following.frame

        prevPos = self.sourcePos
        dups = centre.dupsBefore if self.smoothDedup else 0
        curPos = prevPos + 1 + dups
        span = curPos - prevPos
        holdGap = bool(self.smoothDedup and self.maxSpan and span > self.maxSpan)
        framesToInsert, timesteps = gapPlan(
            prevPos, curPos, self.factorNum, self.factorDen
        )
        if self.factorDen == 1 and span == 1:
            # Identical values, but the driver's own ladder keeps its timestep
            # buffer cached on the common path.
            timesteps = None
        holdFromPrev = (
            cutHoldSplit(prevPos, curPos, self.factorNum, self.factorDen)
            if centre.isCut
            else framesToInsert
        )
        self.sourcePos = curPos
        self._pos = curPos

        self._clearMemos()
        try:
            if centre.isCut or holdGap:
                fromCurrent = self._emitHeldGap(centre, framesToInsert, holdFromPrev)
                self.driver.cacheFrameReset(centre.frame)
            else:
                self._drive(centre, nextFrame, framesToInsert, timesteps)
                fromCurrent = 0
            self._emitCentre(centre, curPos, fromCurrent)
            self._flushCollected()
        finally:
            self._collected.clear()
            self._clearMemos()

        self._maskAnchor = centre.frame
        self._lastFed = centre

    def _emitHeldGap(self, centre, framesToInsert, holdFromPrev) -> int:
        """Fill the gap with copies: from the previous frame first, then the current one.

        Returns how many copies came from the current frame (the centre's own
        cache key only matters when there are some).
        """
        prev = self._lastFed
        fromPrev = min(holdFromPrev, framesToInsert) if prev is not None else 0
        for _ in range(fromPrev):
            self._emitHeld(prev.frame, prev, cut=False)
        fromCurrent = framesToInsert - fromPrev
        for index in range(fromCurrent):
            self._emitHeld(centre.frame, centre, cut=centre.isCut and index == 0)
        return fromCurrent

    def _drive(self, centre, nextFrame, framesToInsert, timesteps) -> None:
        sink = self._sink
        if self.maskedSink is not None:
            anchor = centre.frame if self.takesNext else self._maskAnchor
            sink = self.maskedSink.bind(sink, anchor)
        if self.takesNext:
            self.driver(centre.frame, nextFrame, sink, framesToInsert, timesteps)
        else:
            self.driver(centre.frame, sink, framesToInsert, timesteps)

    def _emitCentre(self, centre, curPos, fromCurrent) -> None:
        # The centre opens the new shot only when no copy of the new frame came
        # before it; otherwise the first copy did.
        isCut = centre.isCut and fromCurrent == 0
        cacheKey = centre if fromCurrent else None
        self._emit(
            FrameSlot(
                centre.frame,
                isCut,
                centre.dupsBefore,
                curPos,
                True,
                cacheKey,
            )
        )

    def _emitHeld(self, frame, key, *, cut: bool) -> None:
        held = frame.clone()
        _syncAfter(held)
        self._emit(FrameSlot(held, cut, 0, self._pos, False, key))

    def emitFrame(self, frame) -> None:
        """A frame a driver produced (not held, not the centre)."""
        self._collected.append(FrameSlot(frame, False, 0, self._pos, False, None))
        if len(self._collected) >= self.collectLimit:
            self._flushCollected(fromDriver=True)

    def _emit(self, slot) -> None:
        self._collected.append(slot)
        if len(self._collected) >= self.collectLimit:
            self._flushCollected()

    def _flushCollected(self, fromDriver: bool = False) -> None:
        """Hand the collected frames on, in order.

        The stages after interpolation run here, and they expect the default
        stream. At the end of a gap this runs on the main thread, which is
        already on it. A mid-gap flush (the collect limit) runs inside the
        driver's put, and gmfss puts from inside its private stream, so that
        case is pinned -- checked once per driver, because entering a stream
        context costs ~8 us.
        """
        collected = self._collected
        if not collected:
            return
        self._collected = []
        if fromDriver:
            pin = self._driverPin
            if pin is None:
                stream = self.defaultStream
                pin = stream is not None and torch.cuda.current_stream() != stream
                self._driverPin = pin
            if pin:
                with torch.cuda.stream(self.defaultStream):
                    for slot in collected:
                        self.downstream.put(slot)
                return
        for slot in collected:
            self.downstream.put(slot)

    def _clearMemos(self) -> None:
        for memo in self.memos:
            memo.clear()

    def trailingOwed(self, trailing: int) -> int:
        """Output slots owed to ``trailing`` duplicates that closed the stream."""
        endPos = self.sourcePos + trailing
        owed = (endPos * self.factorNum) // self.factorDen - (
            self.sourcePos * self.factorNum
        ) // self.factorDen
        self.sourcePos = endPos
        return owed


class _Terminal:
    """The writer. Remembers the last centre frame, which trailing duplicates copy."""

    def __init__(self, writer):
        self.writer = writer
        self.lastCentre = None

    def put(self, slot) -> None:
        if slot.isCentre:
            self.lastCentre = slot.frame
        self.writer.write(slot.frame)

    def end(self) -> None:
        pass

    def payout(self, owed: int) -> None:
        for _ in range(owed):
            held = self.lastCentre.clone()
            _syncAfter(held)
            self.writer.write(held)


def _frameBytes(shape, half: bool) -> int:
    """Bytes of one 3-channel frame of ``shape`` (width, height) at fp16 or fp32."""
    width, height = shape
    return 3 * int(width) * int(height) * (2 if half else 4)


class Pipeline:
    """The chain of nodes for one plan, fed one decoded frame at a time."""

    def __init__(
        self,
        plan,
        drivers,
        writer,
        *,
        factorNum: int = 1,
        factorDen: int = 1,
        smoothDedup: bool = False,
        smoothDedupMaxSpan: int = 0,
        maskedSink=None,
        customModel=None,
        half: bool = False,
        collectLimit: int = 32,
    ):
        self.plan = plan
        self.half = half
        self.consumed = 0
        self.terminal = _Terminal(writer)
        self.smoothDedup = smoothDedup

        downstream = self.terminal
        nodes = []
        for index in range(len(plan.units) - 1, -1, -1):
            unit = plan.units[index]
            precedingDetector = None
            if index > 0 and plan.units[index - 1].name in _DEDUP_LIKE:
                precedingDetector = drivers.get(plan.units[index - 1].name)
            node = self._build(
                unit,
                drivers,
                downstream,
                factorNum,
                factorDen,
                smoothDedup,
                smoothDedupMaxSpan,
                maskedSink,
                precedingDetector,
            )
            node.inShape = (unit.inWidth, unit.inHeight)
            nodes.append(node)
            downstream = node
        self.head = downstream
        self.nodes = list(reversed(nodes))
        defaultStream = _defaultStream()
        for node in self.nodes:
            if isinstance(node, ExpandNode):
                node.defaultStream = defaultStream
                node.collectLimit = max(1, int(collectLimit))
                if plan.upscaleBefore("interpolate"):
                    # On upscaled frames the opposite wins: the writer's copy
                    # of each large frame overlaps the driver's next timestep
                    # only if it is handed on at once (the old interpolate-last
                    # path wrote straight to the writer; collecting there cost
                    # 0.4% at 720p->1440p).
                    node.collectLimit = 1
        self._wireReuse(drivers, customModel)

    @staticmethod
    def _build(
        unit,
        drivers,
        downstream,
        factorNum,
        factorDen,
        smoothDedup,
        smoothDedupMaxSpan,
        maskedSink,
        precedingDetector,
    ):
        name = unit.name
        outShape = (unit.outWidth, unit.outHeight)
        if name in _DEDUP_LIKE:
            return FilterNode(
                name,
                drivers[name],
                downstream,
                carriesDups=(name == "smooth-dedup"),
                outShape=outShape,
            )
        if name == "scenechange":
            return AnnotateNode(
                name,
                drivers[name],
                downstream,
                scoreSource=precedingDetector,
                outShape=outShape,
            )
        if name == "interpolate":
            return ExpandNode(
                name,
                drivers[name],
                factorNum,
                factorDen,
                downstream,
                smoothDedup=smoothDedup,
                smoothDedupMaxSpan=smoothDedupMaxSpan,
                maskedSink=maskedSink,
                outShape=outShape,
            )
        if name == "hop":
            return MapNode(
                name,
                _toDevice(unit.method),
                (0, 0),
                downstream,
                outShape=outShape,
            )
        driver = drivers[name]
        if name == "upscale":
            return MapNode(
                name,
                driver,
                temporalDemand(driver),
                downstream,
                takesNext=True,
                outShape=outShape,
            )
        return MapNode(
            name,
            driver,
            temporalDemand(driver),
            downstream,
            takesNext=False,
            outShape=outShape,
        )

    def _wireReuse(self, drivers, customModel) -> None:
        """Let the first stateless upscaler after interpolation serve held copies once."""
        expander = None
        for node in self.nodes:
            if isinstance(node, ExpandNode):
                expander = node
                continue
            if expander is None or node.name != "upscale" or node.memo is not None:
                continue
            unit = self.plan.unit("upscale")
            if unit is None or not isStatelessUpscale(
                drivers["upscale"], unit.method, customModel
            ):
                continue
            node.memo = {}
            expander.memos.append(node.memo)

    # Counters ----------------------------------------------------------

    @property
    def dropped(self) -> int:
        return sum(node.drops for node in self.nodes if isinstance(node, FilterNode))

    @property
    def cuts(self) -> int:
        return sum(node.cuts for node in self.nodes if isinstance(node, AnnotateNode))

    def ringFrames(self) -> int:
        """Frames the rings keep alive beyond the one each stage is working on."""
        return sum(node.ring.capacity - 1 for node in self.nodes)

    def ringBytes(self) -> int:
        """Bytes of ``ringFrames``, each at its ring's (input) frame size."""
        return sum(
            (node.ring.capacity - 1) * _frameBytes(node.inShape, self.half)
            for node in self.nodes
        )

    # Driving -----------------------------------------------------------

    def feed(self, rawFrame) -> None:
        self.consumed += 1
        self.head.put(FrameSlot(rawFrame, False, 0, self.consumed))

    def pull(self, read) -> bool:
        """Read one decoded frame and run it through the chain. False at the end.

        Reading and feeding happen in one call, as the old FrameWindow did: the
        frame goes into the head stage the moment the decode queue hands it
        over. Returning to the caller's loop in between cost ~40 us per frame
        at 360p (measured), because the decode thread, woken by the queue,
        takes the GIL right there.
        """
        rawFrame = read()
        if rawFrame is None:
            return False
        self.consumed += 1
        self.head.put(FrameSlot(rawFrame, False, 0, self.consumed))
        return True

    def drain(self) -> None:
        """End of stream: every ring releases its remaining centres, in chain order."""
        self.head.end()

    def payoutTrailing(self) -> None:
        """Pay out the duplicates that closed the stream, as clones of the last centre.

        They are never routed back through upscale: a recurrent upscaler would
        change its output, and a stateless one would waste work on the copies.
        """
        if not self.smoothDedup or not self.nodes:
            return
        smooth = self.nodes[0]
        expander = next((n for n in self.nodes if isinstance(n, ExpandNode)), None)
        if expander is None or not isinstance(smooth, FilterNode):
            return
        trailing = smooth.pendingDups
        if not trailing or self.terminal.lastCentre is None:
            return
        owed = expander.trailingOwed(trailing)
        smooth.pendingDups = 0
        self.terminal.payout(owed)


__all__ = [
    "DROP",
    "ExpandNode",
    "FilterNode",
    "MapNode",
    "AnnotateNode",
    "Pipeline",
]
