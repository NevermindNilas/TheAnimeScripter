"""Processing order taken from the command line: flag order is processing order.

Each orderable processing unit (``dedup``, ``restore``, ``upscale``,
``interpolate``) sits where the first flag that turns it on or configures it
was typed. argparse does the recording: every enabling flag and every child
flag of a unit carries a :class:`StageOrderAction`, which notes the unit the
first time it fires. argparse has already resolved prefix abbreviations,
``--flag=value`` and nargs by then, so the position is the one the user meant.
``--json`` records the same units in the file's key order.

``--smooth_dedup`` is recorded under its own name and then overrides the
``interpolate`` position: wherever ``--interpolate`` was typed is ignored. The
smooth-dedup detector itself is not positional (the engine always runs it on
decoded frames), so it does not appear in the order.

:func:`resolveStageOrder` runs once, after every enable/disable step, and turns
the recorded positions plus the final enabled set into ``args.stageOrder``.
"""

import argparse
import functools

# The processing units that have a place in the order. Their values are the
# only ones ``args.stageOrder`` ever holds.
STAGE_UNITS = ("dedup", "restore", "upscale", "interpolate")

# Flag "parents" that name a unit of the order, or the smooth-dedup override.
# Every other parent in PARENT_FLAG_DEFAULTS (resize, segment, depth, ...) runs
# outside the frame loop and has no place in the order.
ORDER_PARENTS = frozenset(STAGE_UNITS) | {"smooth_dedup"}

# The one parent that is recorded but is not itself a unit: it positions
# "interpolate" (see resolveStageOrder).
_SMOOTH_DEDUP = "smooth_dedup"

_RAW_ATTRIBUTE = "_stageOrderRaw"


def recordStageOrder(namespace, parent: str, enabling: bool) -> None:
    """Note that ``parent`` was configured, in argv (or JSON key) order.

    ``enabling`` is True for the flag that turns the unit on (``--upscale``) and
    False for a child flag that does so implicitly (``--upscale_method``). Only
    the first firing of each kind is kept, so each parent contributes at most
    two entries and the list is a single sequence whose indices give the
    relative positions.
    """
    raw = getattr(namespace, _RAW_ATTRIBUTE, None)
    if raw is None:
        raw = []
        setattr(namespace, _RAW_ATTRIBUTE, raw)
    if (parent, enabling) not in raw:
        raw.append((parent, enabling))


class StageOrderAction(argparse.Action):
    """A parser action that behaves exactly as argparse's own, plus recording.

    The wrapped behaviour is chosen by the caller through the same keywords
    argparse uses: ``nargs=0, const=True`` gives ``store_true``; anything else
    stores the converted value, so ``type=``, ``choices=`` and ``nargs=`` keep
    their meaning. Only public argparse API is used.
    """

    def __init__(self, option_strings, dest, stageParent, stageEnabling, **kwargs):
        super().__init__(option_strings, dest, **kwargs)
        self.stageParent = stageParent
        self.stageEnabling = stageEnabling

    def __call__(self, parser, namespace, values, option_string=None):
        recordStageOrder(namespace, self.stageParent, self.stageEnabling)
        setattr(namespace, self.dest, self.const if self.nargs == 0 else values)


def orderedFlag(parent: str, *, enabling: bool, flag: bool = False) -> dict:
    """``add_argument`` keywords that route a flag through :class:`StageOrderAction`.

    Merge the result into the flag's own call. ``flag=True`` gives the
    ``store_true`` form; for a value flag, the caller's ``type``, ``nargs``,
    ``choices`` and ``default`` apply unchanged.
    """
    kwargs = {
        "action": functools.partial(
            StageOrderAction, stageParent=parent, stageEnabling=enabling
        )
    }
    if flag:
        kwargs.update(nargs=0, const=True, default=False)
    return kwargs


def resolveStageOrder(args) -> list[str]:
    """Set and return ``args.stageOrder``: the enabled units in processing order.

    Call it after ``_applyImpliedFlags`` and every other enable/disable step.
    A unit's place is its enabling flag's first position; the first child
    flag's position is used only when the enabling flag is absent. Units that
    are enabled without any recorded position (none today) follow the recorded
    ones in the canonical ``STAGE_UNITS`` order.
    """
    raw = list(getattr(args, _RAW_ATTRIBUTE, None) or [])
    firstIndex: dict[tuple[str, bool], int] = {}
    for index, key in enumerate(raw):
        firstIndex.setdefault(key, index)

    def positionOf(parent: str):
        if (parent, True) in firstIndex:
            return firstIndex[(parent, True)]
        return firstIndex.get((parent, False))

    def unitPosition(unit: str):
        if unit == "interpolate" and getattr(args, _SMOOTH_DEDUP, False):
            # --smooth_dedup sets the place of interpolation, wherever
            # --interpolate was typed. Fall back to --interpolate's own place
            # only if the smooth flag was never recorded.
            smoothPosition = positionOf(_SMOOTH_DEDUP)
            if smoothPosition is not None:
                return smoothPosition
        return positionOf(unit)

    enabled = [unit for unit in STAGE_UNITS if getattr(args, unit, False)]

    def sortKey(unit: str):
        position = unitPosition(unit)
        if position is None:
            return (1, len(raw), STAGE_UNITS.index(unit))
        return (0, position, 0)

    order = sorted(enabled, key=sortKey)
    args.stageOrder = order
    return order


def stageOrderWarning(order: list[str]) -> str | None:
    """The one order warning: ``--dedup`` placed after ``--interpolate``.

    Dedup after interpolation runs the interpolation model on every held
    drawing and then discards most of its in-between frames. The order is
    still honoured; this only says the inference is wasted.
    """
    if "dedup" not in order or "interpolate" not in order:
        return None
    if order.index("dedup") < order.index("interpolate"):
        return None
    return (
        "--dedup comes after --interpolate, so the in-between frames it "
        "interpolates across held drawings will mostly be discarded by dedup, "
        "wasting that inference. Put --dedup first, or use --smooth_dedup to "
        "interpolate across duplicates on purpose."
    )
