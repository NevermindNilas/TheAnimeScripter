from src.depth.backends._batch import iterBatches


class _SentinelReadBuffer:
    def __init__(self, frames):
        self._items = list(frames) + [None]
        self._index = 0

    def read(self):
        if self._index >= len(self._items):
            raise AssertionError("iterBatches() attempted an extra read after EOF")
        item = self._items[self._index]
        self._index += 1
        return item


def testIterBatchesStopsAfterShortFinalBatch():
    reader = _SentinelReadBuffer(["f1", "f2", "f3", "f4", "f5"])

    batches = list(iterBatches(reader.read, 4))

    assert batches == [["f1", "f2", "f3", "f4"], ["f5"]]
