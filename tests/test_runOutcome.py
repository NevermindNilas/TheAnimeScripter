"""Pin the "a run must not lie about its outcome" contract.

Every capability that bypasses ``main.py:start()`` has to report for itself, or
the After Effects panel sits on the last progress string and the batch loop
judges the run purely by the size of a possibly half-written file. ``--segment``,
``--obj_detect`` and ``--stabilize`` already did; ``--depth`` and ``--moblur``
did not.
"""

import ast
from pathlib import Path

import pytest

from src.io.runOutcome import outputWasWritten, truncatedDecodeError

REPO_ROOT = Path(__file__).resolve().parent.parent
STANDALONE = REPO_ROOT / "src" / "factories" / "standalone.py"
BACKENDS = REPO_ROOT / "src" / "depth" / "backends"

# Every standalone capability that writes a video file, swallows its own
# errors, and therefore owes main.py a processingError. --autoclip writes
# autoclipresults.txt instead, and --moblur re-raises rather than swallowing,
# so main()'s per-video handler already counts it as a failure.
VIDEO_CAPABILITIES = [
    "objectDetection",
    "segment",
    "depth",
    "stabilize",
]


# --------------------------------------------------------------------------- #
# outputWasWritten
# --------------------------------------------------------------------------- #


def testOutputWasWrittenForAPlainFile(tmp_path):
    written = tmp_path / "out.mp4"
    written.write_bytes(b"x" * 32)
    assert outputWasWritten(str(written)) is True
    assert outputWasWritten(str(tmp_path / "missing.mp4")) is False


def testOutputWasWrittenForAnImageSequence(tmp_path):
    pattern = str(tmp_path / "frames_%05d.png")
    assert outputWasWritten(pattern) is False
    (tmp_path / "frames_00001.png").write_bytes(b"x" * 32)
    assert outputWasWritten(pattern) is True


def testOutputWasWrittenIgnoresUnrelatedFilesInTheSequenceFolder(tmp_path):
    (tmp_path / "notes.txt").write_bytes(b"x" * 32)
    assert outputWasWritten(str(tmp_path / "frames_%05d.png")) is False


# --------------------------------------------------------------------------- #
# truncatedDecodeError
# --------------------------------------------------------------------------- #


class _Reader:
    def __init__(self, error=None, delivered=0):
        self.decodeError = error
        self._emittedFrames = delivered


class _Writer:
    def __init__(self, error=None):
        self.encodeError = error


def testShortDecodeIsBlamed():
    error = RuntimeError("decoder died")
    assert truncatedDecodeError(_Reader(error, 5), 20, _Writer()) is error


def testDecodeErrorAfterTheLastFrameIsNot():
    """A teardown error or a corrupt trailing packet leaves a complete output;
    failing that run would be a lie in the other direction."""
    error = RuntimeError("decoder teardown")
    assert truncatedDecodeError(_Reader(error, 20), 20, _Writer()) is None


def testCleanRunBlamesNothing():
    assert truncatedDecodeError(_Reader(None, 20), 20, _Writer()) is None


def testEncodeErrorIsBlamedEvenWithAFullDecode():
    """The writer logged every encoding exception and left the output file's
    size as the only signal -- enough for one file, which ends up 0 bytes, but
    an image sequence that died after one frame looks exactly like success."""
    error = RuntimeError("encoder died")
    assert truncatedDecodeError(_Reader(None, 20), 20, _Writer(error)) is error


# --------------------------------------------------------------------------- #
# standalone factory propagation
# --------------------------------------------------------------------------- #


def _functionNode(name):
    tree = ast.parse(STANDALONE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name}() not found in standalone.py")


@pytest.mark.parametrize("capability", VIDEO_CAPABILITIES)
def testStandaloneCapabilityPropagatesItsError(capability):
    """Without this main.py falls back to the output file's size and counts a
    failed run as a success."""
    node = _functionNode(capability)
    assigns = [
        target
        for stmt in ast.walk(node)
        if isinstance(stmt, ast.Assign)
        for target in stmt.targets
        if isinstance(target, ast.Attribute) and target.attr == "processingError"
    ]
    assert assigns, (
        f"standalone.{capability}() never sets self.processingError, so a failed "
        "run reports success"
    )


# --------------------------------------------------------------------------- #
# depth backends
# --------------------------------------------------------------------------- #


def _depthClasses():
    for path in sorted(BACKENDS.glob("*.py")):
        if path.name.startswith("_"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                yield path.name, node


def testEveryDepthBackendCarriesTheOutcomeContract():
    """A depth run reports for itself: it bypasses start()/_notifyAdobe."""
    missing = []
    for fileName, node in _depthClasses():
        bases = {b.id for b in node.bases if isinstance(b, ast.Name)}
        inheritsFromSibling = any(
            base.endswith(("CUDA", "MPS", "Cuda", "Mps")) for base in bases
        )
        if "DepthRunOutcome" not in bases and not inheritsFromSibling:
            missing.append(f"{fileName}:{node.name}")
    assert not missing, f"depth backends without the outcome contract: {missing}"


def testDepthFrameLoopsRunUnderTheGuard():
    """A raise inside process() otherwise leaves the reader blocked on put()
    and the writer waiting for a sentinel, hanging the executor join."""
    unguarded = []
    for path in sorted(BACKENDS.glob("*.py")):
        if path.name.startswith("_"):
            continue
        text = path.read_text(encoding="utf-8")
        if "executor.submit(self.process)" in text:
            unguarded.append(path.name)
        if "executor.submit(self.process_nelux)" in text:
            unguarded.append(path.name)
    assert not unguarded, (
        f"depth frame loops submitted without guardedProcess: {unguarded}"
    )
