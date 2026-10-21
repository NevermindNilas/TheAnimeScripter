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


def checkChainInstances(chain) -> None:
    """Raise ``ValueError`` if ``chain`` (``StageSpec``s) needs two instances of
    a single-instance backend. The message names both stages."""
    seen = {}
    for spec in chain:
        if spec is None or spec.kind not in _MODEL_KINDS:
            continue
        for method in _methodsOf(spec):
            reason = SINGLE_INSTANCE_METHODS.get(method)
            if reason is None:
                continue
            if method in seen:
                raise ValueError(
                    f"{method} cannot run twice in one chain ({seen[method]} and "
                    f"{spec.id}): {reason}"
                )
            seen[method] = spec.id


def checkStageInstances(plan) -> None:
    """Raise ``ValueError`` if ``plan`` needs two instances of a single-instance backend."""
    checkChainInstances(
        unit.spec
        for unit in plan.units
        if unit.spec is not None and unit.kind in _MODEL_KINDS
    )
