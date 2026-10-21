"""Wire the planned processing stages to their backends."""

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
    "resize": 5,
}

# The processor attributes the factories read for one stage's settings, by the
# stage kind that owns them. A view for one stage answers its own kind's names
# from the stage's settings and refuses every other kind's names, so a factory
# can never silently pick up stage 1's settings (or a removed global) for
# stage 2.
_STAGE_ATTRIBUTES = {
    "upscale": ("upscaleMethod", "upscaleFactor", "customModel"),
    "interpolate": (
        "interpolateMethod",
        "interpolateFactor",
        "ensemble",
        "dynamicScale",
        "staticStep",
    ),
    "restore": ("restoreMethod",),
    "dedup": ("dedupMethod", "dedupSens"),
    "smooth-dedup": (
        "smoothDedupMethod",
        "smoothDedupSens",
        "smoothDedupMaxSpan",
    ),
}
_ALL_STAGE_ATTRIBUTES = frozenset(
    name for names in _STAGE_ATTRIBUTES.values() for name in names
)


def stageSettings(unit) -> dict:
    """``unit``'s settings under the attribute names the factories read."""
    settings = unit.settings
    kind = unit.kind
    if kind == "upscale":
        return {
            "upscaleMethod": settings["method"],
            "upscaleFactor": int(settings["factor"]),
            "customModel": settings.get("customModel") or None,
        }
    if kind == "interpolate":
        return {
            "interpolateMethod": settings["method"],
            "interpolateFactor": settings["factor"],
            "ensemble": bool(settings.get("ensemble", False)),
            "dynamicScale": bool(settings.get("dynamicScale", False)),
            "staticStep": bool(settings.get("staticStep", False)),
        }
    if kind == "restore":
        methods = settings["methods"]
        return {
            "restoreMethod": [methods] if isinstance(methods, str) else list(methods)
        }
    if kind == "dedup":
        return {"dedupMethod": settings["method"], "dedupSens": settings["threshold"]}
    if kind == "smooth-dedup":
        return {
            "smoothDedupMethod": settings["detector"],
            "smoothDedupSens": settings["threshold"],
            "smoothDedupMaxSpan": int(settings["maxSpan"]),
        }
    return {}


class StageView:
    """What a factory sees as ``self`` while it builds one stage's driver.

    Answers the stage's own settings (``upscaleMethod``, ``interpolateFactor``,
    ...), the frame size and fps that stage receives, and falls back to the
    processor for everything run-wide (``half``, ``compileMode``,
    ``forceStatic``...). Any other stage's setting name raises, because reading
    it would be a run-wide value standing in for a per-stage one.
    """

    def __init__(self, processor, unit, fps=None):
        self._processor = processor
        self._unit = unit
        values = stageSettings(unit)
        values["width"] = unit.inWidth
        values["height"] = unit.inHeight
        if fps is not None:
            values["fps"] = fps
        self.__dict__.update(values)

    def __getattr__(self, name):
        # Only reached for names this stage does not define.
        if name in _ALL_STAGE_ATTRIBUTES:
            raise AttributeError(
                f"{name!r} is not a setting of stage {self._unit.name!r}; a "
                f"factory must not read another stage's settings"
            )
        return getattr(self._processor, name)


def initializeModels(self, plan):
    """Build one driver per planned unit, at the frame size that unit receives.

    ``plan`` is the plan the pipeline planner resolved from ``args.chain``
    (``src/pipeline/planner.py``). Each driver is built with its stage's own
    settings (through a ``StageView``) and input size, because the fixed I/O
    buffers (and CUDA graphs) of the interpolation, restore and dedup backends
    are sized at construction.

    Returns ``{unit id: driver}``. A ``hop`` unit has no driver.

    Drivers are built in a fixed order by kind, then in chain order within a
    kind: the order they are built in decides where their weights, engines and
    CUDA graph pools land and what memory cuDNN autotunes against, and the old
    engine's order is the measured baseline.
    """
    from src.pipeline.instances import checkStageInstances
    from src.pipeline.planner import factorFraction

    # Before any model loads: a backend that cannot run two live instances is
    # refused here, not by a crash halfway through building the second.
    checkStageInstances(plan)

    # The fps each unit receives: the source fps times every interpolate
    # factor before it. Only used for the factories' log lines.
    fpsAt = {}
    fps = getattr(self, "fps", None)
    for unit in plan.units:
        fpsAt[unit.name] = fps
        if unit.kind == "interpolate" and fps is not None:
            fps = fps * float(factorFraction(unit.spec["factor"]))

    order = {unit.name: index for index, unit in enumerate(plan.units)}
    drivers = {}
    for unit in sorted(
        plan.units,
        key=lambda unit: (_BUILD_ORDER.get(unit.kind, 99), order[unit.name]),
    ):
        name = unit.name
        kind = unit.kind
        if kind == "hop":
            continue
        view = StageView(self, unit, fpsAt[name])
        if kind == "upscale":
            _announce(f"Initializing upscale model: {view.upscaleMethod}...")

            from src.factories.upscale import buildUpscaleProcess

            drivers[name] = buildUpscaleProcess(
                view, width=unit.inWidth, height=unit.inHeight
            )

        elif kind == "interpolate":
            _announce(f"Initializing interpolation model: {view.interpolateMethod}...")

            from src.factories.interpolate import buildInterpolateProcess

            drivers[name] = buildInterpolateProcess(view, unit.inWidth, unit.inHeight)

        elif kind == "restore":
            _announce(f"Initializing restore model: {view.restoreMethod}...")

            from src.factories.restore import buildRestoreProcess

            drivers[name] = buildRestoreProcess(
                view, width=unit.inWidth, height=unit.inHeight
            )

        elif kind in ("dedup", "smooth-dedup"):
            # --smooth_dedup reuses the dedup detectors, but off its own knobs: it
            # turns each duplicate into extra interpolation slots rather than
            # dropping a frame from the output.
            smooth = kind == "smooth-dedup"
            method = view.smoothDedupMethod if smooth else view.dedupMethod
            sens = view.smoothDedupSens if smooth else view.dedupSens
            _announce(f"Initializing duplicate detection: {method}...")

            from src.factories.dedup import buildDedupProcess

            drivers[name] = buildDedupProcess(
                view, method, sens, width=unit.inWidth, height=unit.inHeight
            )

        elif kind == "scenechange":
            _announce(f"Initializing scene-cut detector: {self.sceneChangeMethod}...")

            from src.factories.sceneChange import buildSceneChangeProcess

            drivers[name] = buildSceneChangeProcess(view)

        elif kind == "resize":
            from src.pipeline.executor import Resize

            drivers[name] = Resize(unit.outWidth, unit.outHeight)

    return drivers
