"""Wire enabled processing capabilities to their backends."""

from src.constants import ADOBE

if ADOBE:
    from src.server.aeComms import progressState


def initializeModels(self):
    outputWidth = self.width
    outputHeight = self.height
    upscaleProcess = None
    interpolateProcess = None
    restoreProcess = None
    dedupProcess = None

    if self.upscale:
        if ADOBE:
            progressState.update(
                {"status": f"Initializing upscale model: {self.upscaleMethod}..."}
            )
        outputWidth *= self.upscaleFactor
        outputHeight *= self.upscaleFactor

        from src.factories.upscale import buildUpscaleProcess

        upscaleProcess = buildUpscaleProcess(self)

    if self.interpolate:
        if ADOBE:
            progressState.update(
                {
                    "status": f"Initializing interpolation model: {self.interpolateMethod}..."
                }
            )

        from src.factories.interpolate import buildInterpolateProcess

        interpolateProcess = buildInterpolateProcess(self, outputWidth, outputHeight)

    if self.restore:
        if ADOBE:
            progressState.update(
                {"status": f"Initializing restore model: {self.restoreMethod}..."}
            )

        from src.factories.restore import buildRestoreProcess

        restoreProcess = buildRestoreProcess(self)

    if self.dedup:
        if ADOBE:
            progressState.update(
                {"status": f"Initializing deduplication: {self.dedupMethod}..."}
            )

        from src.factories.dedup import buildDedupProcess

        dedupProcess = buildDedupProcess(self)

    return (
        outputWidth,
        outputHeight,
        upscaleProcess,
        interpolateProcess,
        restoreProcess,
        dedupProcess,
    )
