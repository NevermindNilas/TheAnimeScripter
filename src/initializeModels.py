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
    sceneChangeProcess = None

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

        # The interpolation driver's fixed I/O buffers (and CUDA graph) are sized
        # at construction, so it must be built at the resolution it will actually
        # receive. In interpolate-first order it runs on the source stream; in
        # interpolate-last order it runs on the upscaled stream, one stage later.
        if self.upscale and not self.interpolateFirst:
            interpWidth, interpHeight = outputWidth, outputHeight
        else:
            interpWidth, interpHeight = self.width, self.height

        interpolateProcess = buildInterpolateProcess(self, interpWidth, interpHeight)

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

    if self.interpolate and getattr(self, "sceneChange", False):
        if ADOBE:
            progressState.update(
                {
                    "status": f"Initializing scene-cut detector: {self.sceneChangeMethod}..."
                }
            )

        from src.factories.sceneChange import buildSceneChangeProcess

        sceneChangeProcess = buildSceneChangeProcess(self)

    return (
        outputWidth,
        outputHeight,
        upscaleProcess,
        interpolateProcess,
        restoreProcess,
        dedupProcess,
        sceneChangeProcess,
    )
