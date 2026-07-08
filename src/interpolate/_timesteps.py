def interpolateTimestep(index, framesToInsert, timesteps=None):
    if timesteps is not None and index < len(timesteps):
        return timesteps[index]
    return (index + 1) * 1 / (framesToInsert + 1)


def fillTimestepBuffer(buffer, cachedValue, timestep):
    if cachedValue != timestep:
        buffer.fill_(timestep)
        return timestep
    return cachedValue
