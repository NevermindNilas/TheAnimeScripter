"""Turn the typed flag order into the processing chain, once, before any model loads.

``planChain`` is pure: it takes the order from ``args.stageOrder`` and the
source size, and returns the chain with the resolution each stage receives.
Drivers are built from that chain (``initializeModels``), so every factory
sees its real input shape, and the executor (``src/pipeline/executor.py``)
runs the same chain.

Placement rules (the proposal's section 1):

- ``smooth-dedup`` is the detector for ``--smooth_dedup``. It always runs first,
  on decoded frames, and is not part of the typed order.
- ``scenechange`` is a modifier. It runs right before the first stage that
  changes the shape or the timeline (``upscale``, ``interpolate``), which is
  where the old engine ran it: after dedup and restore, before upscale.
- A hop to CUDA is inserted when a CPU-output backend (DirectML, OpenVINO, NCNN)
  feeds a CUDA-only consumer (FlowNetS dedup).
"""

from dataclasses import dataclass, field

ORDERABLE = ("dedup", "restore", "upscale", "interpolate")
"""The processing units that have a place in the order."""

_DEDUP_LIKE = ("dedup", "smooth-dedup")
_SCENE_BEFORE = ("upscale", "interpolate")
_CPU_OUTPUT_SUFFIXES = ("-directml", "-openvino", "-ncnn")
# LineThin runs on the CPU unless its method names a GPU backend.
_CPU_OUTPUT_METHODS = frozenset(
    {"linethinner-lite", "linethinner-medium", "linethinner-heavy"}
)
# Detectors that read their input on the GPU without moving it there first.
_CUDA_ONLY_CONSUMER_METHODS = frozenset(
    {"flownets", "flownets-rocm", "vmaf-cuda", "vmaf-rocm"}
)


@dataclass(frozen=True)
class Unit:
    """One stage of the chain, with the frame size it receives and emits."""

    name: str  # smooth-dedup | dedup | restore | upscale | interpolate | scenechange | hop
    kind: str  # filter | map | annotate | expand
    inWidth: int
    inHeight: int
    outWidth: int
    outHeight: int
    method: str = ""


@dataclass(frozen=True)
class Plan:
    units: tuple = ()
    outWidth: int = 0
    outHeight: int = 0
    line: str = ""
    hints: tuple = field(default_factory=tuple)

    def names(self) -> tuple:
        return tuple(unit.name for unit in self.units)

    def unit(self, name: str) -> Unit | None:
        for unit in self.units:
            if unit.name == name:
                return unit
        return None

    def upscaleBefore(self, name: str) -> bool:
        """Whether an upscale stage runs before ``name`` in this chain."""
        names = self.names()
        return (
            "upscale" in names
            and name in names
            and (names.index("upscale") < names.index(name))
        )


def _producesCpu(method: str) -> bool:
    return method.endswith(_CPU_OUTPUT_SUFFIXES) or method in _CPU_OUTPUT_METHODS


def _needsCuda(method: str) -> bool:
    return method in _CUDA_ONLY_CONSUMER_METHODS


def planChain(
    order,
    *,
    width: int,
    height: int,
    upscaleFactor: int = 1,
    interpolateFactor: float = 1,
    smoothDedup: bool = False,
    sceneChange: bool = False,
    methods=None,
) -> Plan:
    """Resolve the processing chain for ``order`` (the enabled ``ORDERABLE`` units).

    ``methods`` maps each unit name to its CLI method string, for shape-independent
    decisions (CPU hops, cost hints) and for the factories.
    """
    methods = dict(methods or {})
    orderList = list(order)
    unknown = [name for name in orderList if name not in ORDERABLE]
    if unknown:
        raise ValueError(f"not an orderable processing unit: {unknown}")
    if len(set(orderList)) != len(orderList):
        raise ValueError(f"a processing unit appears twice in the order: {orderList}")
    if smoothDedup and "dedup" in orderList:
        raise ValueError("--smooth_dedup and --dedup are mutually exclusive")
    if sceneChange and "interpolate" not in orderList:
        raise ValueError("--scenechange needs --interpolate")

    def methodOf(name: str) -> str:
        # Restore takes a list of methods and runs them in sequence, so the
        # chain's output lives where its last method leaves it.
        value = methods.get(name, "")
        if isinstance(value, (list, tuple)):
            return str(value[-1]) if value else ""
        return value if isinstance(value, str) else ""

    chain = []  # (name, method) in processing order, scenechange included
    if smoothDedup:
        chain.append(("smooth-dedup", methodOf("smooth-dedup")))
    for name in orderList:
        chain.append((name, methodOf(name)))
    if sceneChange:
        at = next(
            (index for index, (name, _m) in enumerate(chain) if name in _SCENE_BEFORE),
            len(chain),
        )
        chain.insert(at, ("scenechange", methodOf("scenechange")))

    units = []
    w, h = int(width), int(height)
    # Filters and the scene-cut detector pass frames through untouched, so the
    # stream stays wherever the last map/expand stage left it.
    onCpu = False
    for name, method in chain:
        if name in ("upscale",):
            outW, outH = w * int(upscaleFactor), h * int(upscaleFactor)
        else:
            outW, outH = w, h
        kind = {
            "smooth-dedup": "filter",
            "dedup": "filter",
            "restore": "map",
            "upscale": "map",
            "scenechange": "annotate",
            "interpolate": "expand",
        }[name]
        if onCpu and _needsCuda(method):
            units.append(Unit("hop", "map", w, h, w, h, "cuda"))
            onCpu = False
        units.append(Unit(name, kind, w, h, outW, outH, method))
        if kind in ("map", "expand"):
            onCpu = _producesCpu(method)
        w, h = outW, outH

    units = tuple(units)
    return Plan(
        units=units,
        outWidth=w,
        outHeight=h,
        line=orderLine(units),
        hints=tuple(costHints(units, interpolateFactor, upscaleFactor)),
    )


def orderLine(units) -> str:
    """The startup line, e.g. ``Order: dedup → restore → upscale → interpolate``."""
    shown = [unit.name for unit in units if unit.name not in ("scenechange", "hop")]
    if not shown:
        return ""
    text = "Order: " + " → ".join(shown)
    names = [unit.name for unit in units]
    if "scenechange" in names:
        index = names.index("scenechange")
        following = units[index + 1].name if index + 1 < len(units) else "the writer"
        text += f" (scenechange before {following})"
    return text


def costHints(units, interpolateFactor, upscaleFactor) -> list:
    """Plain-language costs of a stage placed after one that changes the cost."""
    hints = []
    for index, unit in enumerate(units):
        if unit.name not in ("restore",) + _DEDUP_LIKE:
            continue
        before = units[:index]
        if any(prior.name == "interpolate" for prior in before):
            hints.append(
                f"{unit.name} runs on {_factorText(interpolateFactor)} the source "
                f"frames (after interpolate)"
            )
        if any(prior.name == "upscale" for prior in before):
            hints.append(
                f"{unit.name} runs at {unit.inWidth}x{unit.inHeight} (after upscale)"
            )
            if unit.name in _DEDUP_LIKE and unit.method.startswith("flownets"):
                hints.append(
                    f"FlowNetS dedup after upscale is about {upscaleFactor}x stricter: "
                    f"its threshold is a mean flow in pixels, so --dedup_sens "
                    f"counts pixels at the upscaled size"
                )
    return hints


def _factorText(factor) -> str:
    value = float(factor)
    if value.is_integer():
        return f"{int(value)}x"
    return f"{value:g}x"
