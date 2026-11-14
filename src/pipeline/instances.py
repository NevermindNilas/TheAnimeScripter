"""Which backends may run as two live instances in one chain.

Every stage of a chain builds its own driver (no driver sharing), so a chain
such as ``upscale span → interpolate → upscale span`` holds two instances of
one class at once. Each backend family was audited for mutable module or class
state that two instances would share (CUDA-graph pools, timestep buffers,
TensorRT engine and context names, ORT sessions, SDK handles, the dynamic-scale
SSIM path); see ``tests/test_stageChainEngineInstances.py``. A backend that
cannot run two instances is listed here and refused twice: by the CLI
(``src/cli/validator.py:_finalizeChain``, a validation error naming the stages)
and again before any model loads (``initializeModels``), instead of crashing or
corrupting output mid-run.
"""

# method -> why a second live instance of it is refused. Empty: every audited
# backend runs two instances with outputs identical to a solo instance.
SINGLE_INSTANCE_METHODS: dict = {}

_MODEL_KINDS = ("upscale", "interpolate", "restore")


def _methodsOf(spec) -> tuple:
    settings = spec.settings
    if spec.kind == "restore":
        methods = settings.get("methods", ())
        return (methods,) if isinstance(methods, str) else tuple(methods)
    return (settings.get("method", ""),)


def _check(instances) -> None:
    """Raise if two of ``instances`` ((label, methods) per live driver) share a
    single-instance method. The message names both."""
    seen = {}
    for label, methods in instances:
        for method in methods:
            reason = SINGLE_INSTANCE_METHODS.get(method)
            if reason is None:
                continue
            if method in seen:
                raise ValueError(
                    f"{method} cannot run twice in one chain ({seen[method]} and "
                    f"{label}): {reason}"
                )
            seen[method] = label


def _passLabel(spec, passIndex: int) -> str:
    return f"{spec.id} (resample pass {passIndex})"


def checkChainInstances(chain) -> None:
    """Raise ``ValueError`` if ``chain`` (``StageSpec``s) needs two instances of
    a single-instance backend. The message names both stages.

    An interpolate stage with ``resample`` N runs N pass drivers of its method
    besides its own (none of its own at factor 1, see ``planner.elidesExpander``).
    """
    from src.pipeline.planner import elidesExpander, resampleCount

    def instances():
        for spec in chain:
            if spec is None or spec.kind not in _MODEL_KINDS:
                continue
            methods = _methodsOf(spec)
            for passIndex in range(1, resampleCount(spec) + 1):
                yield _passLabel(spec, passIndex), methods
            if not elidesExpander(spec):
                yield spec.id, methods

    _check(instances())


def checkStageInstances(plan) -> None:
    """Raise ``ValueError`` if ``plan`` needs two instances of a single-instance
    backend. Every model unit is one live driver, resample passes included."""
    _check(
        (
            _passLabel(unit.spec, unit.passIndex)
            if unit.kind == "resample"
            else unit.spec.id,
            _methodsOf(unit.spec),
        )
        for unit in plan.units
        if unit.spec is not None and unit.kind in _MODEL_KINDS + ("resample",)
    )
