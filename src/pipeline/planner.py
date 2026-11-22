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
- A resize whose target equals its incoming size has no unit (it is logged
  through ``Plan.skipped``).
- An interpolate stage with ``resample`` N > 0 gets N ``resample`` units right
  before it (``resample``, ``resample#2``, ...; ``passIndex`` 1..N). Each pass
  re-times every frame to a flow midpoint and keeps the frame count. At factor 1
  the stage's own unit is left out: it would only pass the frames through.
"""

from dataclasses import dataclass, field
from fractions import Fraction

from src.pipeline.chain import KINDS, StageSpec

_ROLES = {
    "smooth-dedup": "filter",
    "restore": "map",
    "upscale": "map",
    "resize": "map",
    "scenechange": "annotate",
    "resample": "resample",
    "interpolate": "expand",
}
_SCENE_BEFORE = ("upscale", "resample", "interpolate")
# The smallest frame side a resize may produce.
MIN_RESIZE_SIDE = 16
# The interpolation factor is resolved to a fraction with this bound on its
# denominator (the old engine's handling of a fractional --interpolate_factor).
FACTOR_MAX_DENOMINATOR = 100


@dataclass(frozen=True)
class Unit:
    """One stage of the plan, with the frame size it receives and emits.

    ``name`` is the unit id (``upscale``, ``upscale#2``, ``smooth-dedup``,
    ``scenechange``...). ``kind`` is the stage kind the executor and
    ``initializeModels`` dispatch on (one of ``KINDS`` or ``smooth-dedup``,
    ``scenechange``, ``resample``). ``role`` is the node type: filter,
    map, annotate, resample or expand. ``spec`` is the typed stage (``None`` for
    ``scenechange``; the smooth stage for ``smooth-dedup``; the
    interpolate stage for its ``resample`` passes). ``passIndex`` is a resample
    unit's 1-based pass number (0 for every other unit).
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
    passIndex: int = 0

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

    def outputFrames(self, sourceFrames: int) -> int:
        """Frames written for ``sourceFrames`` decoded frames.

        Each interpolate stage writes ``floor((n - 1) * factor) + 1`` for the
        ``n`` frames it receives: its output grid starts on its first frame and
        ends on its last, with nothing after the last (by design, no padding).
        Stages are chained in order, since a fractional factor's rounding
        compounds. ``sourceFrames * factor`` overcounted by ``factor - 1``
        (744 shown, 743 written at 2x).
        """
        frames = int(sourceFrames)
        for unit in self.units:
            if unit.kind == "interpolate" and frames > 0:
                factor = factorFraction(unit.spec["factor"])
                frames = (frames - 1) * factor.numerator // factor.denominator + 1
        return frames

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


def resampleCount(spec) -> int:
    """How many resample passes an interpolate ``spec`` asks for (0 = none).

    The one place the ``resample`` key is read: the CLI always sets it, but
    specs built by tests and tools may leave it out.
    """
    if spec is None or spec.kind != "interpolate":
        return 0
    return int(spec.settings.get("resample", 0))


def elidesExpander(spec) -> bool:
    """Whether a resampling stage at factor 1 runs without its own expander:
    at 1x it would only pass the resampled frames through."""
    return resampleCount(spec) > 0 and factorFraction(spec["factor"]) == 1


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


def _methodOf(spec: StageSpec) -> str:
    """The method a unit is labelled with (a restore stage's last one)."""
    if spec.kind == "restore":
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
    _validateResample(chain)


def _validateResample(chain) -> None:
    """The resample rules the CLI enforces, again for chains built elsewhere."""
    firstInterpolate = next(
        (spec for spec in chain if spec.kind == "interpolate"), None
    )
    for spec in chain:
        count = resampleCount(spec)
        if count < 0:
            raise ValueError(f"{spec.id}: resample must be 0 or more, got {count}")
        if count == 0:
            continue
        if spec.smooth:
            raise ValueError("resample cannot be combined with --smooth_dedup")
        if spec is not firstInterpolate:
            raise ValueError(
                f"{spec.id}: resample is only allowed on the first interpolate stage"
            )
        if str(spec["method"]).startswith("maxine"):
            raise ValueError(
                f"{spec.id}: resample is not supported by {spec['method']}: two "
                f"live Maxine frame-generation instances were never verified"
            )


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

    # (name, kind, method, spec, passIndex) in processing order.
    entries = []
    smoothSpec = next((spec for spec in rest if spec.smooth), None)
    if smoothSpec is not None:
        detector = str(smoothSpec.settings.get("detector", ""))
        entries.append(("smooth-dedup", "smooth-dedup", detector, smoothSpec, 0))
    passes = 0
    for spec in rest:
        method = _methodOf(spec)
        for _ in range(resampleCount(spec)):
            passes += 1
            name = "resample" if passes == 1 else f"resample#{passes}"
            entries.append((name, "resample", method, spec, passes))
        if not elidesExpander(spec):
            entries.append((spec.id, spec.kind, method, spec, 0))
    if sceneChange:
        at = next(
            (index for index, entry in enumerate(entries) if entry[1] in _SCENE_BEFORE),
            len(entries),
        )
        entries.insert(at, ("scenechange", "scenechange", sceneChangeMethod, None, 0))

    units = []
    w, h = decodeW, decodeH
    for name, kind, method, spec, passIndex in entries:
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
        unit = Unit(name, _ROLES[kind], w, h, outW, outH, method, kind, spec, passIndex)
        units.append(unit)
        labels.append(_label(unit))
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
        hints=tuple(costHints(units)),
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
    if kind == "resample":
        # A resampling stage's passes show on its interpolate label; at 1x, where
        # the stage has no unit of its own, its last pass carries that label.
        spec = unit.spec
        if unit.passIndex == resampleCount(spec) and elidesExpander(spec):
            return _interpolateLabel(spec)
        return ""
    if kind == "smooth-dedup":
        return f"smooth-dedup {unit.method}".rstrip()
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
        return _interpolateLabel(unit.spec)
    if kind == "resize":
        return f"resize {resizeTargetText(settings['target'])}"
    return unit.name


def _interpolateLabel(spec) -> str:
    """``interpolate rife4.26 2x``, plus its switches and ``resample N``."""
    settings = spec.settings
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
    passes = resampleCount(spec)
    if passes:
        text += f" resample {passes}"
    return text


def orderLine(units, labels=None) -> str:
    """The startup line, e.g. ``Order: resize 1440p → upscale span → interpolate rife4.26 2x``.

    ``labels`` is the planner's list of stage labels (it also holds the decoder
    resize and skipped resizes, which have no unit); without it the labels are
    derived from ``units``.
    """
    if labels is None:
        labels = [_label(unit) for unit in units if unit.kind != "scenechange"]
    labels = [label for label in labels if label]
    if not labels:
        return ""
    text = "Order: " + " → ".join(labels)
    names = [unit.name for unit in units]
    if "scenechange" in names:
        index = names.index("scenechange")
        following = units[index + 1].kind if index + 1 < len(units) else "the writer"
        if following == "resample":
            # The passes belong to their interpolate stage; the line never
            # names them on their own.
            following = "interpolate"
        text += f" (scenechange before {following})"
    return text


def costHints(units) -> list:
    """Plain-language costs of a stage placed after one that changes the cost.

    A restore after interpolation runs on the product of the preceding
    interpolate factors times the source frames; one after an upscale or an
    enlarging resize runs at its (bigger) input size. A resampling stage costs
    one model run per frame and one more model in memory per pass.
    """
    hints = []
    for index, unit in enumerate(units):
        if unit.kind == "resample" and unit.passIndex == 1:
            passes = resampleCount(unit.spec)
            hints.append(
                f"{unit.spec.id}: resample runs {passes} interpolation "
                f"pass{'es' if passes > 1 else ''} at {unit.inWidth}x{unit.inHeight}, "
                f"each one more model run per frame and one more model in memory"
            )
        if unit.kind != "restore":
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
    return hints


def _factorText(factor) -> str:
    value = float(factor)
    if value.is_integer():
        return f"{int(value)}x"
    return f"{value:g}x"
