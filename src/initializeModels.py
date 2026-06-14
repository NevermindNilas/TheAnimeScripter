"""
Thin coordinator: wires up per-capability factories and re-exports the
standalone driver functions so callers that import from this module keep
working without change.
"""

from src.constants import ADOBE

if ADOBE:
    from src.utils.aeComms import progressState

# Re-export standalone drivers so existing call sites (main.py) don't change.

# Re-export RestoreChain for any code that references it via this module.


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
