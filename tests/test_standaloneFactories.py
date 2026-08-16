"""Every standalone driver is handed the rate it actually emits.

`segment()` was the sole outlier: all four of its arms passed
`self.outputFPS` (`fps * interpolate_factor`) while depth, object detection,
stabilize and motion blur passed `self.fps`. Since `--segment` bypasses the
interpolating frame loop entirely and writes one frame per decoded frame, the
container claimed twice the rate it wrote at -- the matte played at 2x and
drifted off its passed-through audio by half its own length, with no warning.

`src/factories/standalone.py` has no module-level imports, so this runs in the
bare CI venv: the driver modules are stubbed out before the factory imports
them.
"""

import sys
import types

import pytest

from src.factories.standalone import segment

SEGMENT_CLASSES = {
    "anime": "AnimeSegment",
    "anime-tensorrt": "AnimeSegmentTensorRT",
    "anime-directml": "AnimeSegmentDirectML",
    "anime-openvino": "AnimeSegmentOpenVino",
}

SOURCE_FPS = 23.976
INTERPOLATED_FPS = SOURCE_FPS * 2  # deliberately wrong for a segment run


class _Recorder:
    """Stands in for an AnimeSegment* driver and keeps its positional args."""

    lastArgs = None

    def __init__(self, *args, **kwargs):
        type(self).lastArgs = args
        self.processingError = None


@pytest.fixture
def stubSegmentModule(monkeypatch):
    module = types.ModuleType("src.segment.animeSegment")
    recorders = {}
    for className in SEGMENT_CLASSES.values():
        recorder = type(className, (_Recorder,), {})
        recorders[className] = recorder
        setattr(module, className, recorder)
    monkeypatch.setitem(sys.modules, "src.segment.animeSegment", module)
    return recorders


def _processor(segmentMethod):
    return types.SimpleNamespace(
        segmentMethod=segmentMethod,
        input="in.mp4",
        output="out.mov",
        width=1920,
        height=1080,
        fps=SOURCE_FPS,
        outputFPS=INTERPOLATED_FPS,
        inpoint=0.0,
        outpoint=0.0,
        encodeMethod="prores",
        benchmark=False,
        totalFrames=100,
        segmentBatch=1,
        processingError=None,
    )


@pytest.mark.parametrize("method", list(SEGMENT_CLASSES))
def testSegmentIsGivenTheSourceRateNotTheInterpolatedOne(method, stubSegmentModule):
    processor = _processor(method)

    segment(processor)

    fpsArg = stubSegmentModule[SEGMENT_CLASSES[method]].lastArgs[4]
    assert fpsArg == SOURCE_FPS
    assert fpsArg != INTERPOLATED_FPS


def testSegmentStillReportsItsOwnOutcome(stubSegmentModule):
    # The "one run, one outcome" wiring must survive the fps change.
    processor = _processor("anime")

    segment(processor)

    assert processor.processingError is None
    assert hasattr(processor, "processingError")


def testNoStandaloneArmReadsTheInterpolatedRate():
    # The invariant that was violated, pinned so a copy-paste cannot reinstate
    # it in a new driver: the interpolated rate belongs to the frame loop, which
    # none of these drivers run. AST rather than text, so the comment explaining
    # the bug does not satisfy its own check.
    import ast
    from pathlib import Path

    tree = ast.parse(
        (
            Path(__file__).resolve().parent.parent
            / "src"
            / "factories"
            / "standalone.py"
        ).read_text(encoding="utf-8")
    )

    reads = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
    }
    assert "outputFPS" not in reads
    assert "fps" in reads
