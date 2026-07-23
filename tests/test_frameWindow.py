"""Tests for the bounded frame window that replaced main.py's self.nextFrame."""

import ast
import pathlib

import pytest

from src.io.frameWindow import FrameSlot, FrameWindow, temporalDemand


class _Reader:
    """Mimics BuildBuffer.read(): yields frames, then a single None sentinel.

    Reading past the sentinel is a bug (the decode queue only ever receives one
    None), so this raises instead of quietly returning None again.
    """

    def __init__(self, frames):
        self._frames = list(frames)
        self._index = 0
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self._index > len(self._frames):
            raise AssertionError("read() called after the None sentinel")
        if self._index == len(self._frames):
            self._index += 1
            return None
        frame = self._frames[self._index]
        self._index += 1
        return frame


def _walk(window):
    """Drive the window to exhaustion, collecting (centre, prev, successor)."""
    seen = []
    while window.advance():
        previous = window.at(-1)
        seen.append(
            (
                window.centre.frame,
                None if previous is None else previous.frame,
                window.successorFrame(),
            )
        )
    return seen


def test_noLookaheadReadsOneFrameAtATime():
    reader = _Reader([1, 2, 3])
    window = FrameWindow(reader, past=0, future=0)

    assert window.advance() is True
    assert window.centre.frame == 1
    # A non-temporal pipeline must not pin an extra decoded frame.
    assert reader.calls == 1
    assert window.successorFrame() is None


def test_lookaheadOneExposesNextFrame():
    window = FrameWindow(_Reader([1, 2, 3]), past=0, future=1)
    assert _walk(window) == [
        (1, None, 2),
        (2, None, 3),
        (3, None, None),  # final frame: no future context
    ]


def test_lookbehindExposesPreviousFrame():
    window = FrameWindow(_Reader([1, 2, 3]), past=1, future=1)
    assert _walk(window) == [
        (1, None, 2),
        (2, 1, 3),
        (3, 2, None),
    ]


def test_ringIsBounded():
    window = FrameWindow(_Reader(range(100)), past=1, future=2)
    while window.advance():
        assert len(window._slots) <= 1 + 1 + 2


def test_singleFrameStream():
    window = FrameWindow(_Reader([7]), past=1, future=1)
    assert _walk(window) == [(7, None, None)]


def test_emptyStream():
    window = FrameWindow(_Reader([]), past=0, future=1)
    assert _walk(window) == []


def test_sentinelIsReadExactlyOnce():
    """BuildBuffer emits one None; a second read() would block forever."""
    reader = _Reader([1, 2, 3])
    window = FrameWindow(reader, past=0, future=1)
    _walk(window)  # _Reader raises if the window reads past the sentinel
    assert reader.calls == 4  # 3 frames + 1 sentinel


def test_advanceAfterExhaustionStaysFalse():
    window = FrameWindow(_Reader([1]), past=0, future=1)
    _walk(window)
    assert window.advance() is False
    assert window.advance() is False


def test_negativeBoundsRejected():
    with pytest.raises(ValueError):
        FrameWindow(_Reader([]), past=-1, future=0)


# --- entry pipeline: domain + dedup -----------------------------------------


def test_enterTransformsIntoTheWindowDomain():
    """Slots hold what `enter` produced, so neighbours share the centre's domain."""
    window = FrameWindow(
        _Reader([1, 2, 3]), past=0, future=1, enter=lambda raw: FrameSlot(raw * 10)
    )
    assert _walk(window) == [(10, None, 20), (20, None, 30), (30, None, None)]


def test_droppedFramesNeverEnterTheWindow():
    """A dedup'd frame is not somebody's `next frame`; the next kept one is."""
    reader = _Reader([1, 2, 2, 3])
    dropDuplicates = []

    def enter(raw):
        if dropDuplicates and dropDuplicates[-1] == raw:
            return None
        dropDuplicates.append(raw)
        return FrameSlot(raw)

    window = FrameWindow(reader, past=0, future=1, enter=enter)
    assert _walk(window) == [(1, None, 2), (2, None, 3), (3, None, None)]
    assert window.consumed == 4
    assert window.dropped == 1


def test_enterRunsOncePerFrameInDecodeOrder():
    """Stateful detectors (dedup, scene-cut) must see the stream in sequence."""
    order = []

    def enter(raw):
        order.append(raw)
        return FrameSlot(raw)

    window = FrameWindow(_Reader([1, 2, 3, 4]), past=0, future=2, enter=enter)
    while window.advance():
        pass
    assert order == [1, 2, 3, 4]


def test_countersTrackDecodedFramesNotKeptOnes():
    window = FrameWindow(
        _Reader([1, 2, 3, 4]),
        past=0,
        future=0,
        enter=lambda raw: None if raw % 2 == 0 else FrameSlot(raw),
    )
    while window.advance():
        pass
    assert (window.consumed, window.dropped) == (4, 2)


# --- validity: no neighbour across a scene cut -------------------------------


def test_successorIsNoneAcrossASceneCut():
    """A frame from the next shot is not future context. Drivers see None."""
    window = FrameWindow(
        _Reader([1, 2, 3]),
        past=0,
        future=1,
        enter=lambda raw: FrameSlot(raw, isCut=(raw == 3)),
    )
    assert _walk(window) == [
        (1, None, 2),
        (2, None, None),  # frame 3 opens a new shot
        (3, None, None),
    ]


def test_cutOnTheCentreDoesNotHideItsOwnSuccessor():
    """isCut on the centre is the caller's hold signal, not a successor veto."""
    window = FrameWindow(
        _Reader([1, 2, 3]),
        past=0,
        future=1,
        enter=lambda raw: FrameSlot(raw, isCut=(raw == 2)),
    )
    seen = []
    while window.advance():
        seen.append((window.centre.frame, window.centre.isCut, window.successorFrame()))
    assert seen == [(1, False, None), (2, True, 3), (3, False, None)]


# --- memoized downstream stages ----------------------------------------------


def test_stagedComputesOncePerSlot():
    window = FrameWindow(_Reader([1, 2, 3]), past=0, future=1)
    calls = []

    def compute(offset):
        calls.append(window.at(offset).frame)
        return window.at(offset).frame * 100

    while window.advance():
        assert window.staged(0, "up", compute) == window.centre.frame * 100
        if window.successor() is not None:
            window.staged(1, "up", compute)

    # Each frame upscaled exactly once even though two stages read it...
    assert sorted(calls) == [1, 2, 3]
    # ...and always in increasing order, so recurrent drivers stay in sequence.
    assert calls == sorted(calls)


def test_stagedIsNoneOffTheEdge():
    window = FrameWindow(_Reader([1]), past=0, future=1)
    window.advance()
    assert window.staged(1, "up", lambda offset: 999) is None


# --- driver declarations -----------------------------------------------------


class _Temporal:
    temporalWindow = (0, 1)


class _Wide:
    temporalWindow = (2, 3)


class _Plain:
    pass


def test_temporalDemandDefaultsToZero():
    assert temporalDemand(None, _Plain(), None) == (0, 0)


def test_temporalDemandTakesWidestPerAxis():
    assert temporalDemand(_Temporal(), _Wide(), _Plain(), None) == (2, 3)


def test_temporalDemandIgnoresDisabledStages():
    assert temporalDemand(_Temporal(), None) == (0, 1)


# Source-level, so it holds without torch/tensorrt/onnxruntime installed.
_TEMPORAL_BACKEND_FILES = [
    "src/upscale/misc.py",
    "src/upscale/tensorrt.py",
    "src/upscale/directml.py",
    "src/interpolate/distildrba.py",
]


def _declaresTemporalWindow(classNode) -> bool:
    return any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "temporalWindow"
            for target in node.targets
        )
        for node in classNode.body
    )


def test_everyTemporalBackendDeclaresItsWindow():
    """A new AnimeSR/DistilDRBA backend must declare temporalWindow.

    Before FrameWindow, the lookahead was gated on `upscaleMethod == "animesr"`,
    so animesr-tensorrt/-directml/-openvino silently ran with next == current.
    The declaration now lives on the class, but only if someone writes it.
    """
    repoRoot = pathlib.Path(__file__).resolve().parent.parent
    missing = []
    for relativePath in _TEMPORAL_BACKEND_FILES:
        source = (repoRoot / relativePath).read_text(encoding="utf-8")
        for node in ast.parse(source).body:
            if not isinstance(node, ast.ClassDef):
                continue
            if not node.name.startswith(("AnimeSR", "DistilDRBA")):
                continue
            if not _declaresTemporalWindow(node):
                missing.append(f"{relativePath}::{node.name}")

    assert not missing, f"temporal backends missing temporalWindow: {missing}"


def test_distilDrbaRejectsACrossDomainNeighbour():
    """The silent bilinear resize of I2 is gone; a mismatch is now a hard error.

    Asserted per class rather than as a total count, so adding a backend means
    adding the guard rather than bumping a magic number.
    """
    guard = "must supply the neighbour in the same domain"
    source = pathlib.Path("src/interpolate/distildrba.py").read_text(encoding="utf-8")

    unguarded = [
        node.name
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef)
        and node.name.startswith("DistilDRBA")
        and guard not in (ast.get_source_segment(source, node) or "")
    ]
    assert not unguarded, (
        f"DistilDRBA backends silently accepting a resized I2: {unguarded}"
    )
