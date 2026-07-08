from src.interpolate._timesteps import fillTimestepBuffer, interpolateTimestep


def testRifeTimestepBufferCachesRepeatedValues():
    fillCalls = []

    class DummyBuffer:
        def fill_(self, value):
            fillCalls.append(value)
            return self

    buffer = DummyBuffer()
    cachedValue = None
    cachedValue = fillTimestepBuffer(buffer, cachedValue, 0.5)
    cachedValue = fillTimestepBuffer(buffer, cachedValue, 0.5)
    cachedValue = fillTimestepBuffer(buffer, cachedValue, 0.25)

    assert fillCalls == [0.5, 0.25]


def testRifeTimestepUsesProvidedValuesBeforeDefaulting():
    assert interpolateTimestep(0, framesToInsert=3, timesteps=[0.2]) == 0.2
    assert interpolateTimestep(1, framesToInsert=3, timesteps=[0.2]) == 0.5
