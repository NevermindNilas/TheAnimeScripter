"""Stage-adapter vocabulary shared by the planner and the executor.

A stage is one unit of the processing chain (dedup, restore, upscale,
interpolate, scene-cut, a device hop). The driver objects in ``src/upscale``,
``src/interpolate``, ``src/unifiedRestore.py`` and the rest are not changed to
fit this: the executor calls them the way the old engine did.
"""

DROP = object()
"""Returned by a filter stage to drop the slot it was given."""

_BASE_STATELESS_UPSCALE_METHODS = frozenset(
    {
        "shufflecugan",
        "cyte",
        "adore",
        "span",
        "open-proteus",
        "aniscale2",
        "rtmosr",
        "saryn",
        "fallin_soft",
        "fallin_strong",
        "gauss",
        "figsr",
        "smosr",
    }
)
_ORT_STATELESS_UPSCALE_METHODS = frozenset(
    method
    for base in _BASE_STATELESS_UPSCALE_METHODS - {"figsr"}
    for method in (f"{base}-directml", f"{base}-openvino")
)
STATELESS_UPSCALE_ALLOWLIST = {
    "UniversalPytorch": _BASE_STATELESS_UPSCALE_METHODS,
    "UniversalPytorchMPS": frozenset(
        f"{method}-mps" for method in _BASE_STATELESS_UPSCALE_METHODS
    ),
    "UniversalPytorchROCm": frozenset(
        f"{method}-rocm" for method in _BASE_STATELESS_UPSCALE_METHODS
    ),
    "UniversalDirectML": _ORT_STATELESS_UPSCALE_METHODS,
    "UniversalNCNN": frozenset({"shufflecugan-ncnn", "adore-ncnn", "span-ncnn"}),
    "UniversalTensorRT": frozenset(
        f"{method}-tensorrt" for method in _BASE_STATELESS_UPSCALE_METHODS - {"figsr"}
    ),
}


def isStatelessUpscale(upscaleProcess, upscaleMethod: str, customModel) -> bool:
    """Whether a built upscaler keeps no state between frames, so one output can
    serve every copy of the same source frame inside an interpolation gap.

    Resolved per built instance: the driver's class, the CLI method and the custom
    model all have to agree with the allowlist. A custom model is never stateless.
    """
    if customModel or getattr(upscaleProcess, "customModel", None):
        return False
    allowedMethods = STATELESS_UPSCALE_ALLOWLIST.get(type(upscaleProcess).__name__)
    return allowedMethods is not None and upscaleMethod in allowedMethods
