import types

from src.depth.backends import mps


class _DummyProgress:
    def __init__(self, _total):
        self.calls = []

    def __enter__(self):
        return self.calls.append

    def __exit__(self, exc_type, exc, tb):
        return False


class _SentinelReadBuffer:
    def __init__(self, frames):
        self._items = list(frames) + [None]
        self._index = 0

    def read(self):
        if self._index >= len(self._items):
            raise AssertionError("process() attempted an extra read after EOF")
        item = self._items[self._index]
        self._index += 1
        return item


def _run_process(process_method):
    processed = []
    state = {"closed": False}
    fake_self = types.SimpleNamespace(
        totalFrames=0,
        depthBatch=4,
        readBuffer=_SentinelReadBuffer(["f1", "f2", "f3", "f4", "f5"]),
        processBatch=lambda frames: processed.append(list(frames)),
        writeBuffer=types.SimpleNamespace(
            close=lambda: state.__setitem__("closed", True),
        ),
        encodeBuffer=types.SimpleNamespace(
            put=lambda value: state.__setitem__("closed", value is None),
        ),
    )

    process_method(fake_self)

    assert processed == [["f1", "f2", "f3", "f4"], ["f5"]]
    assert state["closed"] is True


def testDepthMpsProcessStopsAfterShortFinalBatch(monkeypatch):
    monkeypatch.setattr(mps, "ProgressBarLogic", _DummyProgress)
    _run_process(mps.DepthMPS.process)


def testOgDepthV2MpsProcessStopsAfterShortFinalBatch(monkeypatch):
    monkeypatch.setattr(mps, "ProgressBarLogic", _DummyProgress)
    _run_process(mps.OGDepthV2MPS.process)
