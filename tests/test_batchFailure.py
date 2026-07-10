"""Tests for main.py failure reporting.

Two bugs are covered:

BUG A -- a failed run used to exit 0 and count failures as successes, so
scripts/CI could not tell a batch fell over. main.py now counts succeeded vs
failed videos and calls ``sys.exit(1)`` when anything failed.

BUG B -- a failed run used to tell the After Effects panel ``setCompleted``,
which then hunted for an output file that was never written (issues #269,
#236). ``VideoProcessor._notifyAdobe`` now sends ``setFailed`` instead.

The failure decision is factored into the pure helper ``main._videoFailed`` so
it can be unit-tested without spinning up the full pipeline; the batch-exit and
AE-notification behaviours are exercised against it.
"""

from types import SimpleNamespace

import pytest

import main
import src.constants as cs


class _RecordingProgressState:
    """Stand-in for aeComms.progressState that records terminal calls."""

    def __init__(self):
        self.completedCalls = []
        self.failedCalls = []

    def setCompleted(self, outputPath=None):
        self.completedCalls.append(outputPath)

    def setFailed(self, error=None):
        self.failedCalls.append(error)


# --------------------------------------------------------------------------- #
# _videoFailed pure helper
# --------------------------------------------------------------------------- #


def test_videoFailed_true_when_processing_error(tmp_path):
    # A stored frame-loop exception is a failure even if an output file exists.
    outputPath = tmp_path / "out.mp4"
    outputPath.write_bytes(b"x" * 1024)
    assert main._videoFailed(RuntimeError("boom"), str(outputPath)) is True


def test_videoFailed_true_when_output_missing(tmp_path):
    assert main._videoFailed(None, str(tmp_path / "does-not-exist.mp4")) is True


def test_videoFailed_true_when_output_zero_bytes(tmp_path):
    outputPath = tmp_path / "empty.mp4"
    outputPath.write_bytes(b"")
    assert main._videoFailed(None, str(outputPath)) is True


def test_videoFailed_false_when_output_written(tmp_path):
    outputPath = tmp_path / "out.mp4"
    outputPath.write_bytes(b"x" * 1024)
    assert main._videoFailed(None, str(outputPath)) is False


def test_videoFailed_benchmark_ignores_missing_output(tmp_path):
    # Benchmark runs intentionally write no output; only a stored exception
    # counts as failure for them.
    assert main._videoFailed(None, str(tmp_path / "none.mp4"), benchmark=True) is False
    assert (
        main._videoFailed(RuntimeError("x"), str(tmp_path / "none.mp4"), benchmark=True)
        is True
    )


# --------------------------------------------------------------------------- #
# BUG B -- _notifyAdobe gating
# --------------------------------------------------------------------------- #


def _bareProcessor(processingError, output, benchmark=False):
    vp = object.__new__(main.VideoProcessor)
    vp.processingError = processingError
    vp.output = output
    vp.benchmark = benchmark
    return vp


def test_notifyAdobe_failed_run_does_not_complete(monkeypatch, tmp_path):
    monkeypatch.setattr(cs, "ADOBE", True, raising=False)
    state = _RecordingProgressState()
    vp = _bareProcessor(RuntimeError("boom"), str(tmp_path / "missing.mp4"))

    vp._notifyAdobe(state)

    assert state.completedCalls == []
    assert len(state.failedCalls) == 1
    assert "boom" in state.failedCalls[0]


def test_notifyAdobe_missing_output_reports_failed(monkeypatch, tmp_path):
    # No stored exception, but the encoder produced nothing -> still failed.
    monkeypatch.setattr(cs, "ADOBE", True, raising=False)
    state = _RecordingProgressState()
    vp = _bareProcessor(None, str(tmp_path / "never-written.mp4"))

    vp._notifyAdobe(state)

    assert state.completedCalls == []
    assert len(state.failedCalls) == 1


def test_notifyAdobe_success_reports_completed(monkeypatch, tmp_path):
    monkeypatch.setattr(cs, "ADOBE", True, raising=False)
    outputPath = tmp_path / "out.mp4"
    outputPath.write_bytes(b"x" * 1024)
    state = _RecordingProgressState()
    vp = _bareProcessor(None, str(outputPath))

    vp._notifyAdobe(state)

    assert state.failedCalls == []
    assert state.completedCalls == [str(outputPath)]


def test_notifyAdobe_noop_when_not_adobe(monkeypatch, tmp_path):
    monkeypatch.setattr(cs, "ADOBE", False, raising=False)
    state = _RecordingProgressState()
    vp = _bareProcessor(RuntimeError("boom"), str(tmp_path / "missing.mp4"))

    vp._notifyAdobe(state)

    assert state.completedCalls == []
    assert state.failedCalls == []


# --------------------------------------------------------------------------- #
# BUG A -- batch loop counts failures and exits non-zero
# --------------------------------------------------------------------------- #


class _StubProcessor:
    """Replaces VideoProcessor: fails when the input path contains 'fail'."""

    def __init__(self, args, results=None):
        self._failed = "fail" in results["videoPath"]

    def didFail(self):
        return self._failed


def _patchMainForBatch(monkeypatch, entries):
    """Wire main()'s parse/resolve/process seams to lightweight stubs."""
    monkeypatch.setattr("sys.argv", ["main.py"])
    # createParser is called inside main() via `from src.cli.parser import ...`,
    # which resolves the attribute on the module at call time -> patch there.
    monkeypatch.setattr(
        "src.cli.parser.createParser", lambda *a, **k: SimpleNamespace()
    )
    monkeypatch.setattr(
        "src.cli.validator.isAnyOtherProcessingMethodEnabled", lambda args: True
    )
    monkeypatch.setattr(
        "src.io.inputOutputHandler.processInputOutputPaths",
        lambda args, outputPath: entries,
    )
    monkeypatch.setattr(main, "VideoProcessor", _StubProcessor)
    # Don't truncate the repo's real TAS-Log.log during the test run.
    monkeypatch.setattr(main.logging, "basicConfig", lambda *a, **k: None)


def test_batch_with_failure_exits_nonzero(monkeypatch):
    entries = [
        {"videoPath": "clip_ok.mp4", "outputPath": "out_ok.mp4"},
        {"videoPath": "clip_fail.mp4", "outputPath": "out_fail.mp4"},
    ]
    _patchMainForBatch(monkeypatch, entries)

    with pytest.raises(SystemExit) as excinfo:
        main.main()
    assert excinfo.value.code == 1


def test_single_video_failure_exits_nonzero(monkeypatch):
    entries = [{"videoPath": "clip_fail.mp4", "outputPath": "out_fail.mp4"}]
    _patchMainForBatch(monkeypatch, entries)

    with pytest.raises(SystemExit) as excinfo:
        main.main()
    assert excinfo.value.code == 1


def test_batch_all_success_returns_cleanly(monkeypatch):
    entries = [
        {"videoPath": "clip_a.mp4", "outputPath": "out_a.mp4"},
        {"videoPath": "clip_b.mp4", "outputPath": "out_b.mp4"},
    ]
    _patchMainForBatch(monkeypatch, entries)

    # No failures -> main() falls off the end without raising SystemExit.
    assert main.main() is None
