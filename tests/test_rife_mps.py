from queue import Queue

import pytest

pytest.importorskip("torch")

from src.interpolate.rife import RifeMPS


def testRifeMpsCachesRepeatedTimesteps():
    instance = RifeMPS.__new__(RifeMPS)
    instance.firstRun = False
    instance.staticStep = False
    instance._cachedTimestepValue = None

    fillCalls = []

    class DummyBuffer:
        def fill_(self, value):
            fillCalls.append(value)
            return self

    instance._timestep_buffer = DummyBuffer()

    processCalls = []

    def fakeProcessFrame(frame, toNorm):
        processCalls.append((frame, toNorm))
        if toNorm == "infer":
            return f"output-{len(processCalls)}"

    instance.processFrame = fakeProcessFrame

    interpQueue = Queue()

    RifeMPS.__call__(instance, "frame-1", interpQueue, framesToInsert=1)
    RifeMPS.__call__(instance, "frame-2", interpQueue, framesToInsert=1)
    RifeMPS.__call__(
        instance,
        "frame-3",
        interpQueue,
        framesToInsert=1,
        timesteps=[0.25],
    )

    assert fillCalls == [0.5, 0.25]
    assert interpQueue.qsize() == 3
