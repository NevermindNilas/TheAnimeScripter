"""Wire enabled processing capabilities to their backends."""

from src.constants import ADOBE

if ADOBE:
    from src.server.aeComms import progressState


def _announce(status):
    if ADOBE:
        progressState.update({"status": status})


_BUILD_ORDER = {
    "upscale": 0,
    "interpolate": 1,
    "restore": 2,
    "dedup": 3,
    "smooth-dedup": 3,
    "scenechange": 4,
}


def initializeModels(self, plan):
    """Build every enabled driver at the frame size its stage actually receives.

    ``plan`` is the chain the pipeline planner resolved from the typed flag order
    (``src/pipeline/planner.py``). Each driver is built with that stage's input
    size, because the fixed I/O buffers (and CUDA graphs) of the interpolation,
    restore and dedup backends are sized at construction: an interpolation placed
    after upscale is built at the upscaled size, one placed before it at the
    source size.

    Returns ``{unit name: driver}``. A ``hop`` unit has no driver.

    Drivers are built in a fixed order, not chain order: the order they are built
    in decides where their weights, engines and CUDA graph pools land and what
    memory cuDNN autotunes against, and the old engine's order is the measured
    baseline.
    """
    drivers = {}
    for unit in sorted(plan.units, key=lambda unit: _BUILD_ORDER.get(unit.name, 99)):
        name = unit.name
        if name == "upscale":
            _announce(f"Initializing upscale model: {self.upscaleMethod}...")

            from src.factories.upscale import buildUpscaleProcess

            drivers[name] = buildUpscaleProcess(
                self, width=unit.inWidth, height=unit.inHeight
            )

        elif name == "interpolate":
            _announce(f"Initializing interpolation model: {self.interpolateMethod}...")

            from src.factories.interpolate import buildInterpolateProcess

            drivers[name] = buildInterpolateProcess(self, unit.inWidth, unit.inHeight)

        elif name == "restore":
            _announce(f"Initializing restore model: {self.restoreMethod}...")

            from src.factories.restore import buildRestoreProcess

            drivers[name] = buildRestoreProcess(
                self, width=unit.inWidth, height=unit.inHeight
            )

        elif name in ("dedup", "smooth-dedup"):
            # --smooth_dedup reuses the dedup detectors, but off its own knobs: it
            # turns each duplicate into extra interpolation slots rather than
            # dropping a frame from the output, so the two are mutually exclusive
            # (validator.py).
            smooth = name == "smooth-dedup"
            method = self.smoothDedupMethod if smooth else self.dedupMethod
            sens = self.smoothDedupSens if smooth else self.dedupSens
            _announce(f"Initializing duplicate detection: {method}...")

            from src.factories.dedup import buildDedupProcess

            drivers[name] = buildDedupProcess(
                self, method, sens, width=unit.inWidth, height=unit.inHeight
            )

        elif name == "scenechange":
            _announce(f"Initializing scene-cut detector: {self.sceneChangeMethod}...")

            from src.factories.sceneChange import buildSceneChangeProcess

            drivers[name] = buildSceneChangeProcess(self)

    return drivers
