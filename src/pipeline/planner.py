"""Turn the typed chain into the processing plan, once, before any model loads.

``planChain`` is pure: it takes ``args.chain`` (a tuple of
``src.pipeline.chain.StageSpec`` in processing order) and the source size, and
returns the units with the resolution each one receives. Drivers are built from
that plan (``initializeModels``), so every factory sees its real input shape,
and the executor (``src/pipeline/executor.py``) runs the same plan.

Placement rules:

- A leading ``resize`` (the chain's FIRST stage) is done by the decoder
  (``BuildBuffer(resize=...)``) and has no unit. Only the first stage
  qualifies; a second leading resize is an ordinary resize unit.
- ``smooth-dedup`` is the detector for the smooth interpolate stage. It always
  runs first, on decoded frames, and is not part of the typed order.
- ``scenechange`` is a modifier. It runs right before the first stage that
  changes the shape or the timeline (an ``upscale`` or ``interpolate``; a
  resize does not count), which is where the old engine ran it.
- A hop to CUDA is inserted when a CPU-output backend (DirectML, OpenVINO, NCNN)
  feeds a CUDA-only consumer (FlowNetS dedup). A resize keeps the device it is
  given, so it neither causes nor clears a hop.
- A resize whose target equals its incoming size has no unit (it is logged
  through ``Plan.skipped``).
"""

from dataclasses import dataclass, field
from fractions import Fraction

from src.pipeline.chain import KINDS, StageSpec

_ROLES = {
    "smooth-dedup": "filter",
    "dedup": "filter",
    "restore": "map",
    "upscale": "map",
    "resize": "map",
    "hop": "map",
    "scenechange": "annotate",
    "interpolate": "expand",
}
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
# The smallest frame side a resize may produce.
MIN_RESIZE_SIDE = 16
# The interpolation factor is resolved to a fraction with this bound on its
# denominator (the old engine's handling of a fractional --interpolate_factor).
FACTOR_MAX_DENOMINATOR = 100


@dataclass(frozen=True)
class Unit:
    """One stage of the plan, with the frame size it receives and emits.

    ``name`` is the unit id (``upscale``, ``upscale#2``, ``smooth-dedup``,
    ``scenechange``, ``hop``...). ``kind`` is the stage kind the executor and
    ``initializeModels`` dispatch on (one of ``KINDS`` or ``smooth-dedup``,
    ``scenechange``, ``hop``). ``role`` is the node type: filter, map, annotate
    or expand. ``spec`` is the typed stage (``None`` for ``scenechange`` and
    ``hop``; the smooth stage for ``smooth-dedup``).
    """

    name: str
    role: str
    inWidth: int
    inHeight: int
    outWidth: int
    outHeight: int
    method: str = ""
    kind: str = ""
    spec: StageSpec | None = None

    def __post_init__(self):
        if not self.kind:
            object.__setattr__(self, "kind", self.name)

    @property
    def settings(self):
        return self.spec.settings if self.spec is not None else {}

    @property
    def enlarges(self) -> bool:
        """Whether this unit hands on bigger frames than it receives."""
        return self.outWidth * self.outHeight > self.inWidth * self.inHeight


@dataclass(frozen=True)
class Plan:
    units: tuple = ()
    outWidth: int = 0
    outHeight: int = 0
    line: str = ""
    hints: tuple = field(default_factory=tuple)
    # The size the decoder hands the first unit. Differs from the source size
    # only when the chain opens with a resize, which the decoder does.
    decodeWidth: int = 0
    decodeHeight: int = 0
    decodeResize: bool = False
    # Product of every interpolate stage's factor, as resolved fractions.
    factor: Fraction = Fraction(1)
    # Log lines for resizes that were skipped because they change nothing.
    skipped: tuple = field(default_factory=tuple)

    def names(self) -> tuple:
        return tuple(unit.name for unit in self.units)

    def unit(self, name: str) -> Unit | None:
        for unit in self.units:
            if unit.name == name:
                return unit
        return None

    def ofKind(self, kind: str) -> tuple:
        return tuple(unit for unit in self.units if unit.kind == kind)

    def enlargedBefore(self, name: str) -> bool:
        """Whether an upscale, or a resize that enlarges, runs before ``name``."""
        for unit in self.units:
            if unit.name == name:
                return False
            if unit.kind == "upscale" or (unit.kind == "resize" and unit.enlarges):
                return True
        return False


def factorFraction(value) -> Fraction:
    """An interpolation factor as the fraction the gap planner runs on."""
    return Fraction(value).limit_denominator(FACTOR_MAX_DENOMINATOR)


def _evenRound(value: float) -> int:
    """The old --resize formula's rounding: to the nearest even integer."""
    return round(value / 2) * 2


def resizeDims(target, inWidth: int, inHeight: int) -> tuple[int, int]:
    """Concrete ``(width, height)`` for a resize ``target`` given its incoming size.

    ``("scale", F)`` reproduces the removed ``--resize --resize_factor F``
    exactly: the width is scaled and rounded to even, then the height is derived
    from the new width through the aspect ratio. ``("height", N)`` keeps the
    incoming aspect ratio. Derived sides are clamped to ``MIN_RESIZE_SIDE``.
    """
    mode = target[0]
    if mode == "size":
        width, height = int(target[1]), int(target[2])
    elif mode == "height":
        height = int(target[1])
        width = max(MIN_RESIZE_SIDE, _evenRound(inWidth * height / inHeight))
    elif mode == "scale":
        factor = float(target[1])
        if not factor > 0:
            raise ValueError(f"a resize scale must be greater than 0, got {factor:g}")
        aspectRatio = inWidth / inHeight
        width = _evenRound(inWidth * factor)
        height = _evenRound(width / aspectRatio)
        width = max(MIN_RESIZE_SIDE, width)
        height = max(MIN_RESIZE_SIDE, height)
    else:
        raise ValueError(f"unknown resize target: {target!r}")
    if width < MIN_RESIZE_SIDE or height < MIN_RESIZE_SIDE:
        raise ValueError(
            f"a resize target must be at least {MIN_RESIZE_SIDE}x{MIN_RESIZE_SIDE}, "
            f"got {width}x{height}"
        )
    return width, height


def resizeTargetText(target) -> str:
    mode = target[0]
    if mode == "size":
        return f"{int(target[1])}x{int(target[2])}"
    if mode == "height":
        return f"{int(target[1])}p"
    return f"{_factorText(target[1])}"


def _producesCpu(method: str) -> bool:
    return method.endswith(_CPU_OUTPUT_SUFFIXES) or method in _CPU_OUTPUT_METHODS


def _needsCuda(method: str) -> bool:
    return method in _CUDA_ONLY_CONSUMER_METHODS


def _methodOf(spec: StageSpec) -> str:
    """The method that decides where a stage leaves the stream (hops, hints)."""
    if spec.kind == "restore":
        # A restore stage runs its methods in sequence, so the stream lives where
        # its last method leaves it.
        methods = spec.settings.get("methods", ())
        if isinstance(methods, str):
            return methods
        return str(methods[-1]) if methods else ""
    if spec.kind == "resize":
        return ""
    return str(spec.settings.get("method", ""))


def _validate(chain) -> None:
    ids = set()
    for spec in chain:
        if not isinstance(spec, StageSpec):
            raise TypeError(f"a chain holds StageSpec entries, got {spec!r}")
        if spec.kind not in KINDS:
            raise ValueError(f"unknown stage kind: {spec.kind!r}")
        if spec.id in ids:
            raise ValueError(f"stage id {spec.id!r} appears twice in the chain")
        ids.add(spec.id)
    smooth = [spec for spec in chain if spec.smooth]
    if len(smooth) > 1:
        raise ValueError("at most one --smooth_dedup stage is allowed")
    if smooth:
        firstInterpolate = next(spec for spec in chain if spec.kind == "interpolate")
        if firstInterpolate is not smooth[0]:
            raise ValueError(
                "the --smooth_dedup stage must be the first interpolate stage: its "
                "gaps are measured in source frames"
            )
        if any(spec.kind == "dedup" for spec in chain):
            raise ValueError("--smooth_dedup and --dedup are mutually exclusive")


def planChain(
    chain,
    *,
    width: int,
    height: int,
    sceneChange: bool = False,
    sceneChangeMethod: str = "",
    decodeResize: bool = True,
) -> Plan:
    """Resolve the processing plan for ``chain`` on a ``width``x``height`` source.

    ``decodeResize=False`` keeps a leading resize as a unit instead of handing
    it to the decoder (for callers that feed frames themselves).
    """
    chain = tuple(chain)
    _validate(chain)
    if sceneChange and not any(spec.kind == "interpolate" for spec in chain):
        raise ValueError("--scenechange needs --interpolate")

    srcW, srcH = int(width), int(height)
    decodeW, decodeH = srcW, srcH
    decoderResizes = False
    skipped = []
    labels = []  # the order line, one entry per typed stage
    rest = chain
    if decodeResize and chain and chain[0].kind == "resize":
        lead = chain[0]
        rest = chain[1:]
        targetW, targetH = resizeDims(lead["target"], srcW, srcH)
        text = f"resize {resizeTargetText(lead['target'])}"
        if (targetW, targetH) == (srcW, srcH):
            skipped.append(
                f"{lead.id}: the source is already {srcW}x{srcH}, nothing to resize"
            )
            labels.append(f"{text} (no-op)")
        else:
            decodeW, decodeH = targetW, targetH
            decoderResizes = True
            labels.append(text)

    entries = []  # (name, kind, method, spec) in processing order
    smoothSpec = next((spec for spec in rest if spec.smooth), None)
    if smoothSpec is not None:
        detector = str(smoothSpec.settings.get("detector", ""))
        entries.append(("smooth-dedup", "smooth-dedup", detector, smoothSpec))
    for spec in rest:
        entries.append((spec.id, spec.kind, _methodOf(spec), spec))
    if sceneChange:
        at = next(
            (index for index, entry in enumerate(entries) if entry[1] in _SCENE_BEFORE),
            len(entries),
        )
        entries.insert(at, ("scenechange", "scenechange", sceneChangeMethod, None))

    units = []
    hops = 0
    w, h = decodeW, decodeH
    # Filters, the scene-cut detector and resizes keep the stream wherever the
    # last model stage left it.
    onCpu = False
    for name, kind, method, spec in entries:
        if kind == "upscale":
            factor = int(spec["factor"])
            outW, outH = w * factor, h * factor
        elif kind == "resize":
            outW, outH = resizeDims(spec["target"], w, h)
            text = f"resize {resizeTargetText(spec['target'])}"
            if (outW, outH) == (w, h):
                skipped.append(f"{name}: the frames are already {w}x{h}, skipped")
                labels.append(f"{text} (no-op)")
                continue
        else:
            outW, outH = w, h
        if onCpu and _needsCuda(method):
            hops += 1
            hopName = "hop" if hops == 1 else f"hop#{hops}"
            units.append(Unit(hopName, "map", w, h, w, h, "cuda", "hop"))
            onCpu = False
        unit = Unit(name, _ROLES[kind], w, h, outW, outH, method, kind, spec)
        units.append(unit)
        labels.append(_label(unit))
        if unit.role in ("map", "expand") and kind != "resize":
            onCpu = _producesCpu(method)
        w, h = outW, outH

    units = tuple(units)
    factor = Fraction(1)
    for unit in units:
        if unit.kind == "interpolate":
            factor *= factorFraction(unit.spec["factor"])
    return Plan(
        units=units,
        outWidth=w,
        outHeight=h,
        line=orderLine(units, labels),
        hints=tuple(costHints(units, decodeW)),
        decodeWidth=decodeW,
        decodeHeight=decodeH,
        decodeResize=decoderResizes,
        factor=factor,
        skipped=tuple(skipped),
    )


def _label(unit) -> str:
    """How one unit reads in the order line, e.g. ``interpolate rife4.26 2x``."""
    kind = unit.kind
    settings = unit.settings
    if kind == "scenechange":
        return ""
    if kind == "smooth-dedup":
        return f"smooth-dedup {unit.method}".rstrip()
    if kind == "dedup":
        return f"dedup {settings.get('method', '')}".rstrip()
    if kind == "restore":
        methods = settings.get("methods", ())
        if isinstance(methods, str):
            methods = (methods,)
        return " ".join(("restore", *map(str, methods)))
    if kind == "upscale":
        model = settings.get("customModel") or ""
        if model:
            import os

            text = f"upscale {os.path.basename(str(model))}"
        else:
            text = f"upscale {settings.get('method', '')}".rstrip()
        factor = int(settings.get("factor", 2))
        # 2x is the default and stays implicit, as it is on the command line.
        return text if factor == 2 else f"{text} {factor}x"
    if kind == "interpolate":
        text = (
            f"interpolate {settings.get('method', '')} "
            f"{_factorText(settings.get('factor', 1))}"
        )
        for key, word in (
            ("ensemble", "ensemble"),
            ("dynamicScale", "dynamic_scale"),
            ("staticStep", "static_step"),
        ):
            if settings.get(key):
                text += f" {word}"
        return text
    if kind == "resize":
        return f"resize {resizeTargetText(settings['target'])}"
    return unit.name


def orderLine(units, labels=None) -> str:
    """The startup line, e.g. ``Order: resize 1440p → upscale span → interpolate rife4.26 2x``.

    ``labels`` is the planner's list of stage labels (it also holds the decoder
    resize and skipped resizes, which have no unit); without it the labels are
    derived from ``units``.
    """
    if labels is None:
        labels = [
            _label(unit) for unit in units if unit.kind not in ("scenechange", "hop")
        ]
    labels = [label for label in labels if label]
    if not labels:
        return ""
    text = "Order: " + " → ".join(labels)
    names = [unit.name for unit in units]
    if "scenechange" in names:
        index = names.index("scenechange")
        following = units[index + 1].kind if index + 1 < len(units) else "the writer"
        text += f" (scenechange before {following})"
    return text


def costHints(units, decodeWidth: int | None = None) -> list:
    """Plain-language costs of a stage placed after one that changes the cost.

    A restore or dedup after interpolation runs on the product of the preceding
    interpolate factors times the source frames; one after an upscale or an
    enlarging resize runs at its (bigger) input size.
    """
    hints = []
    if decodeWidth is None:
        decodeWidth = units[0].inWidth if units else 0
    for index, unit in enumerate(units):
        if unit.kind not in ("restore",) + _DEDUP_LIKE:
            continue
        before = units[:index]
        interps = [prior for prior in before if prior.kind == "interpolate"]
        if interps:
            product = Fraction(1)
            for prior in interps:
                product *= factorFraction(prior.spec["factor"])
            names = ", ".join(prior.name for prior in interps)
            hints.append(
                f"{unit.name} runs on {_factorText(product)} the source "
                f"frames (after {names})"
            )
        enlargers = [
            prior
            for prior in before
            if prior.kind == "upscale" or (prior.kind == "resize" and prior.enlarges)
        ]
        if enlargers:
            names = ", ".join(prior.name for prior in enlargers)
            hints.append(
                f"{unit.name} runs at {unit.inWidth}x{unit.inHeight} (after {names})"
            )
            scale = unit.inWidth / decodeWidth if decodeWidth else 1
            if (
                unit.kind in _DEDUP_LIKE
                and unit.method.startswith("flownets")
                and scale > 1
            ):
                hints.append(
                    f"FlowNetS dedup after {names} is about {_factorText(scale)} "
                    f"stricter: its threshold is a mean flow in pixels, so "
                    f"--dedup_sens counts pixels at the upscaled size"
                )
    return hints


def _factorText(factor) -> str:
    value = float(factor)
    if value.is_integer():
        return f"{int(value)}x"
    return f"{value:g}x"
