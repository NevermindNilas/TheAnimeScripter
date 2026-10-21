"""The processing chain from the command line and ``--json``.

Flag order is processing order, and every stage may be typed any number of
times: ``--resize 720p --upscale span --interpolate rife4.26 --upscale``.

argparse records what was typed. Every stage flag carries a
:class:`StageFlagAction`, which appends a :class:`StageEvent` with its values;
every per-stage long flag (``--upscale_method``, ``--interpolate_factor``,
``--custom_model``, ...) carries a :class:`ChildFlagAction`, which appends a
:class:`ChildEvent`. argparse has already resolved abbreviations, ``=value``
and nargs by then, so each event sits where the user meant it. ``--json``
produces the same events, from its ``"chain"`` list or from its key order.

:func:`buildStageDrafts` turns the events into one :class:`StageDraft` per
stage, in order:

1. each stage flag's values are classified (``src/cli/stageValues.py``);
2. a long flag with no stage of its kind anywhere creates one at its own
   position (only the flags that enabled their stage before, see
   ``CHILD_FLAGS``);
3. every other long flag binds to the most recent stage of its kind typed
   before it, or, typed before any, is a default for every stage of the kind;
4. ``--smooth_dedup`` merges with a single plain ``--interpolate`` when it has
   no inline interpolation values of its own (today's meaning of the pair);
   long flags bound to either side become the merged stage's;
5. settings resolve per stage: inline value > bound long flag > the previous
   same-kind stage ("sticky") > a pre-stage long-flag default > the built-in
   default.

The validator then resolves each draft (CUDA fallbacks, sensitivity mapping,
range checks) and :func:`specsFromDrafts` turns them into ``args.chain``.
"""

import argparse
import difflib
import functools
import logging
import os
import shlex
from collections import Counter
from dataclasses import dataclass, field

from src.cli.stageValues import (
    GRAMMAR_KINDS,
    SETTING_NAMES,
    StageChainError,
    ValueContext,
    classifyOccurrence,
    describeTarget,
    kindLabel,
)
from src.pipeline.chain import StageSpec

# Stage flag dest -> grammar kind. ``smooth_dedup`` builds an interpolate
# stage with ``smooth=True``.
STAGE_FLAGS = {kind: kind for kind in GRAMMAR_KINDS}

# The chain kinds that carry a per-kind boolean on ``args``. ``smooth_dedup``
# is True iff the smooth instance exists.
STAGE_BOOLEANS = (
    "dedup",
    "restore",
    "upscale",
    "interpolate",
    "smooth_dedup",
    "resize",
)

# Per-stage long flag dest -> (group, settings key, creates). ``group`` is the
# stage kind the flag binds to ("interpolate" includes the smooth stage,
# "smooth_dedup" is the smooth stage only). ``creates`` is True for the flags
# that enable their stage when none is typed -- exactly the flags that did so
# before (src/cli/config.py:PARENT_FLAG_DEFAULTS); --custom_model, --ensemble,
# --dynamic_scale and --static_step never enabled anything and still do not.
CHILD_FLAGS = {
    "dedup_method": ("dedup", "method", True),
    "dedup_sens": ("dedup", "sens", True),
    "restore_method": ("restore", "methods", True),
    "upscale_method": ("upscale", "method", True),
    "upscale_factor": ("upscale", "factor", True),
    "custom_model": ("upscale", "customModel", False),
    "interpolate_method": ("interpolate", "method", True),
    "interpolate_factor": ("interpolate", "factor", True),
    "ensemble": ("interpolate", "ensemble", False),
    "dynamic_scale": ("interpolate", "dynamicScale", False),
    "static_step": ("interpolate", "staticStep", False),
    "smooth_dedup_method": ("smooth_dedup", "detector", True),
    "smooth_dedup_sens": ("smooth_dedup", "sens", True),
    "smooth_dedup_max_span": ("smooth_dedup", "maxSpan", True),
}

INTERPOLATE_KEYS = ("method", "factor", "ensemble", "dynamicScale", "staticStep")
SMOOTH_KEYS = ("detector", "sens", "maxSpan")

# Settings keys each draft resolves, in a stable order.
KIND_KEYS = {
    "dedup": ("method", "sens"),
    "restore": ("methods",),
    "upscale": ("method", "factor", "customModel"),
    "interpolate": INTERPOLATE_KEYS,
    "smooth_dedup": INTERPOLATE_KEYS + SMOOTH_KEYS,
    "resize": ("target",),
}

# Used only when no parser is at hand (a namespace built by another
# entrypoint); the parser's own defaults win otherwise.
FALLBACK_DEFAULTS = {
    ("dedup", "method"): "ssim",
    ("dedup", "sens"): 35,
    ("restore", "methods"): ("anime1080fixer",),
    ("upscale", "method"): "shufflecugan",
    ("upscale", "factor"): 2,
    ("upscale", "customModel"): "",
    ("interpolate", "method"): "rife4.6",
    ("interpolate", "factor"): 2,
    ("interpolate", "ensemble"): False,
    ("interpolate", "dynamicScale"): False,
    ("interpolate", "staticStep"): False,
    ("smooth_dedup", "detector"): "ssim",
    ("smooth_dedup", "sens"): 35,
    ("smooth_dedup", "maxSpan"): 6,
}

# --json keys that configure a stage. Any of them next to "chain" is an error.
PER_STAGE_JSON_KEYS = frozenset(STAGE_FLAGS) | frozenset(CHILD_FLAGS)

REMOVED_FLAG_MESSAGES = {
    "output_scale": "--output_scale was removed: put `--resize {value}` last in the chain",
    "resize_factor": "--resize_factor was removed: use `--resize {value}`",
}
_REMOVED_FLAG_EXAMPLES = {"output_scale": "WxH", "resize_factor": "0.5"}

BARE_RESIZE_MESSAGE = (
    "--resize needs a target, e.g. --resize 1440p, --resize 1920x1080 or "
    "--resize 0.5. The on/off --resize and --resize_factor were removed: "
    "`--resize --resize_factor 0.5` is now `--resize 0.5`."
)

_EVENTS_ATTRIBUTE = "_stageEvents"
DRAFTS_ATTRIBUTE = "_stageDrafts"


# --- what argparse and --json record ------------------------------------------


@dataclass
class StageEvent:
    kind: str  # a grammar kind (``smooth_dedup`` included)
    values: tuple
    flag: str  # what to call it in a message: "--upscale", 'chain[2]'
    position: int
    where: str = ""  # message prefix for --json chain entries


@dataclass
class ChildEvent:
    group: str
    key: str
    value: object
    flag: str
    position: int
    creates: bool
    # A --json key at its parser default: lowest precedence, never binds, never
    # creates. The After Effects panel serializes its whole form, so these are
    # everywhere and must not ask for anything.
    default: bool = False


def recordedEvents(namespace):
    return list(getattr(namespace, _EVENTS_ATTRIBUTE, None) or [])


def _append(namespace, makeEvent):
    events = getattr(namespace, _EVENTS_ATTRIBUTE, None)
    if events is None:
        events = []
        setattr(namespace, _EVENTS_ATTRIBUTE, events)
    event = makeEvent(len(events))
    events.append(event)
    return event


def recordStage(namespace, kind, values, flag, where=""):
    return _append(
        namespace,
        lambda position: StageEvent(kind, tuple(values), flag, position, where),
    )


def recordChild(namespace, dest, value, flag, default=False):
    group, key, creates = CHILD_FLAGS[dest]
    if key == "methods" and value is not None:
        value = tuple(value) if isinstance(value, (list, tuple)) else (value,)
    return _append(
        namespace,
        lambda position: ChildEvent(
            group, key, value, flag, position, creates and not default, default
        ),
    )


class StageFlagAction(argparse.Action):
    """``--upscale [VALUE ...]``: one stage, its values classified later."""

    def __init__(self, option_strings, dest, stageKind, **kwargs):
        super().__init__(option_strings, dest, **kwargs)
        self.stageKind = stageKind

    def __call__(self, parser, namespace, values, option_string=None):
        if values is None:
            values = []
        elif isinstance(values, str):
            values = [values]
        recordStage(namespace, self.stageKind, values, self.option_strings[0])
        setattr(namespace, self.dest, True)


class ChildFlagAction(argparse.Action):
    """A per-stage long flag: stores like argparse's own action, and records.

    ``nargs=0, const=True`` gives ``store_true``; anything else stores the
    converted value, so ``type=``, ``choices=`` and ``nargs=`` keep their
    meaning.
    """

    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        value = self.const if self.nargs == 0 else values
        setattr(namespace, self.dest, value)
        recordChild(namespace, self.dest, value, self.option_strings[0])


class RemovedFlagAction(argparse.Action):
    """A removed flag: a migration error instead of "unrecognized arguments"."""

    def __call__(self, parser, namespace, values, option_string=None):
        parser.error(removedFlagMessage(self.dest, values))


def removedFlagMessage(dest, value=None):
    shown = value if value not in (None, "", True, False) else None
    return REMOVED_FLAG_MESSAGES[dest].format(
        value=shown if shown is not None else _REMOVED_FLAG_EXAMPLES[dest]
    )


def stageFlag(kind):
    """``add_argument`` keywords for a stage flag."""
    return {
        "action": functools.partial(StageFlagAction, stageKind=kind),
        "nargs": "*",
        "default": False,
        "metavar": "VALUE",
    }


def childFlag(*, flag=False):
    """``add_argument`` keywords for a per-stage long flag."""
    kwargs = {"action": ChildFlagAction}
    if flag:
        kwargs.update(nargs=0, const=True, default=False)
    return kwargs


def removedFlag():
    return {"action": RemovedFlagAction, "nargs": "?", "help": argparse.SUPPRESS}


# --- --json "chain" -----------------------------------------------------------


def splitChainEntry(entry):
    """``shlex`` (POSIX quoting) with backslash escapes off.

    Quotes work as in a shell, so ``upscale "D:/my models/x.pth"`` is one path;
    backslashes are kept literally, so an unquoted Windows path survives.
    """
    lexer = shlex.shlex(entry, posix=True)
    lexer.whitespace_split = True
    lexer.escape = ""
    lexer.commenters = ""
    return list(lexer)


def recordJsonChain(namespace, chain):
    """Record a ``"chain"`` list. Errors name the entry's index."""
    if not isinstance(chain, list):
        raise StageChainError(
            '"chain" must be a list of strings, e.g. '
            '["resize 1440p", "upscale span", "interpolate rife4.26 2x"].'
        )
    for index, entry in enumerate(chain):
        where = f'"chain"[{index}]'
        if not isinstance(entry, str):
            raise StageChainError(
                f'{where} must be a string such as "upscale span", got '
                f"{type(entry).__name__} {entry!r}."
            )
        if not entry.strip():
            raise StageChainError(f"{where} is empty; it must name a stage.")
        try:
            words = splitChainEntry(entry)
        except ValueError as error:
            raise StageChainError(
                f"{where}: cannot split {entry!r}: {error}."
            ) from None
        kind = words[0].lower()
        if kind not in STAGE_FLAGS:
            message = (
                f"{where}: unknown stage {words[0]!r}. A chain entry starts with "
                f"one of: {', '.join(GRAMMAR_KINDS)}."
            )
            close = difflib.get_close_matches(kind, GRAMMAR_KINDS, n=1, cutoff=0.6)
            if close:
                message += f" Did you mean {close[0]!r}?"
            raise StageChainError(message)
        recordStage(namespace, kind, words[1:], f"{where} {kind!r}", where=where)


# --- building the drafts --------------------------------------------------------


@dataclass(eq=False)
class StageDraft:
    """One stage while the chain is resolved. ``settings`` is final once
    :func:`buildStageDrafts` returns; the validator may still rewrite values
    (fallbacks, sensitivity mapping)."""

    kind: str  # a chain kind: dedup, restore, upscale, interpolate, resize
    smooth: bool
    position: int
    typed: bool  # a stage flag, not a long flag that created it
    flag: str
    occurrence: int  # typed order among the same grammar kind
    inline: dict = field(default_factory=dict)
    texts: dict = field(default_factory=dict)
    bound: dict = field(default_factory=dict)  # key -> (value, flag)
    settings: dict = field(default_factory=dict)
    index: int = 0
    where: str = ""
    # More than one stage of this kind: messages must say which one.
    repeated: bool = False

    def flagName(self, legacy):
        """``dedup_method`` for a kind typed once (the wording it always had),
        ``dedup #2: dedup_method`` when the kind repeats."""
        if not self.repeated:
            return legacy
        name = "smooth_dedup" if self.smooth else f"{self.kind} #{self.index}"
        return f"{name}: {legacy}"

    @property
    def grammarKind(self):
        return "smooth_dedup" if self.smooth else self.kind

    def provisionalLabel(self):
        label = kindLabel(self.grammarKind, self.occurrence)
        return f"{self.where} {label}".strip() if self.where else label

    @property
    def label(self):
        """``interpolate #2 (rife4.26)`` once indexed, for messages."""
        name = (
            "smooth_dedup"
            if self.smooth
            else f"{self.kind} #{self.index or self.occurrence}"
        )
        summary = summarize(self.kind, self.settings)
        return f"{name} ({summary})" if summary else name


def summarize(kind, settings):
    if kind == "restore":
        return " ".join(settings.get("methods", ()))
    if kind == "resize":
        target = settings.get("target")
        return describeTarget(target) if target else ""
    if kind == "upscale" and settings.get("customModel"):
        return (
            f"{settings.get('method', '')}, {os.path.basename(settings['customModel'])}"
        )
    return str(settings.get("method", ""))


def _accepts(draft, group):
    if group == "smooth_dedup":
        return draft.smooth
    if group == "interpolate":
        return draft.kind == "interpolate"
    return draft.kind == group


def _kindOfGroup(group):
    return "interpolate" if group == "smooth_dedup" else group


def _same(a, b):
    if isinstance(a, str) and isinstance(b, str) and (os.sep in a or "/" in a):
        return os.path.normcase(os.path.abspath(a)) == os.path.normcase(
            os.path.abspath(b)
        )
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return tuple(a) == tuple(b)
    return a == b


def _show(value):
    if isinstance(value, (list, tuple)):
        return " ".join(str(v) for v in value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


class ChainContext:
    """What the builder needs from the parser: method lists and defaults."""

    def __init__(self, values, defaults):
        self.values = values
        self.defaults = dict(defaults)

    @classmethod
    def fromParser(cls, parser):
        defaults = dict(FALLBACK_DEFAULTS)
        actions = {action.dest: action for action in parser._actions}
        for dest, (group, key, _creates) in CHILD_FLAGS.items():
            action = actions.get(dest)
            if action is None:
                continue
            value = action.default
            if key == "methods":
                value = tuple(value or ())
            defaults[(group, key)] = value
        return cls(ValueContext.fromParser(parser), defaults)


def _bind(draft, event):
    """Bind a long flag to ``draft``; conflicting values are an error."""
    key = event.key
    label = draft.provisionalLabel()
    if key in draft.inline and not _same(draft.inline[key], event.value):
        raise StageChainError(
            f"{label}: {event.flag} {_show(event.value)} conflicts with "
            f"'{draft.texts.get(key, _show(draft.inline[key]))}' given on "
            f"{draft.flag}. Give the {SETTING_NAMES[key]} once."
        )
    if key in draft.bound and not _same(draft.bound[key][0], event.value):
        other, otherFlag = draft.bound[key]
        raise StageChainError(
            f"{label}: {event.flag} {_show(event.value)} conflicts with "
            f"{otherFlag} {_show(other)}, both bound to this stage. Give the "
            f"{SETTING_NAMES[key]} once, or type the stage flag again for "
            "another stage."
        )
    draft.bound.setdefault(key, (event.value, event.flag))


def _mergeSmooth(drafts):
    """Rules 2 and 3 of --smooth_dedup (see module docstring)."""
    smooth = [draft for draft in drafts if draft.smooth]
    if not smooth:
        return drafts
    s = smooth[0]
    plain = [d for d in drafts if d.kind == "interpolate" and not d.smooth]
    if any(key in s.inline for key in INTERPOLATE_KEYS):
        return drafts

    if len(plain) == 1:
        p = plain[0]
        # Without inline values the smooth stage takes over the plain one,
        # however either side was tuned: a long flag typed after
        # --smooth_dedup binds to it, but still describes the ONE
        # interpolation the pair means (`--smooth_dedup --ensemble
        # --interpolate rife4.22`). Two stages need values on --smooth_dedup.
        for key in INTERPOLATE_KEYS:
            if key in p.inline:
                if key in s.bound and not _same(s.bound[key][0], p.inline[key]):
                    raise StageChainError(
                        f"smooth_dedup: {s.bound[key][1]} {_show(s.bound[key][0])} "
                        f"conflicts with '{p.texts[key]}' given on --interpolate, "
                        "which --smooth_dedup takes over. Give the "
                        f"{SETTING_NAMES[key]} once, or, for two interpolations, "
                        "put the smooth one's values on --smooth_dedup itself "
                        "(e.g. --smooth_dedup rife4.26 2x)."
                    )
                s.inline[key] = p.inline[key]
                s.texts[key] = p.texts[key]
            if key in p.bound:
                if key in s.bound and not _same(s.bound[key][0], p.bound[key][0]):
                    raise StageChainError(
                        f"smooth_dedup: {s.bound[key][1]} {_show(s.bound[key][0])} "
                        f"conflicts with {p.bound[key][1]} {_show(p.bound[key][0])}, "
                        "which belong to the same interpolation. Give the "
                        f"{SETTING_NAMES[key]} once."
                    )
                s.bound.setdefault(key, p.bound[key])
        # Position: --smooth_dedup itself, else a typed --interpolate (a
        # smooth option alone does not move it, e588f60a), else the first
        # smooth option, which beats an interpolate option.
        if not s.typed and p.typed:
            s.position = p.position
        return [d for d in drafts if d is not p]

    if len(plain) >= 2:
        raise StageChainError(
            f"--smooth_dedup has no interpolation settings of its own and there "
            f"are {len(plain)} plain --interpolate stages, so it is not clear "
            "which one it replaces: put the method/factor of the smooth "
            "interpolation on --smooth_dedup (e.g. --smooth_dedup rife4.26 2x)."
        )
    return drafts


def buildStageDrafts(events, context):
    """Every rule of the chain except the per-stage runtime validation.

    ``events`` are :class:`StageEvent`/:class:`ChildEvent` in typed order;
    ``context`` is a :class:`ChainContext`. Returns the drafts in processing
    order with ``settings`` fully resolved and ``index`` set. Raises
    :class:`StageChainError`.
    """
    stageEvents = [e for e in events if isinstance(e, StageEvent)]
    childEvents = sorted(
        (e for e in events if isinstance(e, ChildEvent)), key=lambda e: e.position
    )

    smoothFlags = [e for e in stageEvents if e.kind == "smooth_dedup"]
    if len(smoothFlags) > 1:
        raise StageChainError(
            f"--smooth_dedup was given {len(smoothFlags)} times; a chain has at "
            "most one smooth-dedup stage. Use --interpolate for the others."
        )

    # 1. the typed stages, values classified
    drafts = []
    occurrences = Counter()
    for event in stageEvents:
        occurrences[event.kind] += 1
        draft = StageDraft(
            kind=_kindOfGroup(event.kind),
            smooth=event.kind == "smooth_dedup",
            position=event.position,
            typed=True,
            flag=event.flag,
            occurrence=occurrences[event.kind],
            where=event.where,
        )
        if event.kind == "resize" and not event.values:
            raise StageChainError(
                f"{event.where}: {BARE_RESIZE_MESSAGE}"
                if event.where
                else BARE_RESIZE_MESSAGE
            )
        draft.inline, draft.texts = classifyOccurrence(
            event.kind, event.values, draft.provisionalLabel(), context.values
        )
        drafts.append(draft)

    # 2. a long flag with no stage of its kind anywhere creates one
    for group in ("smooth_dedup", "dedup", "restore", "upscale", "interpolate"):
        if any(_accepts(draft, group) for draft in drafts):
            continue
        creating = [e for e in childEvents if e.group == group and e.creates]
        if creating:
            first = creating[0]
            drafts.append(
                StageDraft(
                    kind=_kindOfGroup(group),
                    smooth=group == "smooth_dedup",
                    position=first.position,
                    typed=False,
                    flag=first.flag,
                    occurrence=1,
                )
            )

    # 3. bind every other long flag, or keep it as a default for its kind
    preDefaults = {}
    for event in childEvents:
        targets = [
            draft
            for draft in drafts
            if _accepts(draft, event.group) and draft.position <= event.position
        ]
        if event.default or not targets:
            preDefaults[(event.group, event.key)] = event.value
            continue
        _bind(max(targets, key=lambda draft: draft.position), event)

    # 4. --smooth_dedup and a plain --interpolate
    drafts = _mergeSmooth(drafts)
    drafts.sort(key=lambda draft: draft.position)

    interpolations = [draft for draft in drafts if draft.kind == "interpolate"]
    if any(draft.smooth for draft in interpolations) and not interpolations[0].smooth:
        first = interpolations[0]
        raise StageChainError(
            "--smooth_dedup must be the first interpolation in the chain, because "
            "its gaps are measured in source frames, but "
            f"{first.provisionalLabel()} ({first.flag}) comes before it. Move "
            "--smooth_dedup before it."
        )

    # 5. settings: inline > bound > sticky > pre-stage default > built-in
    previous = {}
    counts = Counter()
    for draft in drafts:
        group = draft.grammarKind
        if draft.kind == "resize":
            sticky = None
        elif draft.kind == "interpolate":
            # A smooth stage never inherits (it is the first interpolation);
            # a plain one inherits from the previous interpolation, smooth
            # included, for the interpolation keys only.
            sticky = None if draft.smooth else previous.get("interpolate")
        else:
            sticky = previous.get(draft.kind)
        settings = {}
        for key in KIND_KEYS[group]:
            # A plain dedup stage has a "sens" too: only the smooth stage's
            # detection keys take the smooth-dedup defaults.
            defaultGroup = (
                "smooth_dedup" if draft.smooth and key in SMOOTH_KEYS else draft.kind
            )
            if key in draft.inline:
                settings[key] = draft.inline[key]
            elif key in draft.bound:
                settings[key] = draft.bound[key][0]
            elif (
                sticky is not None
                and key in sticky.settings
                and not (
                    key == "customModel"
                    and ("method" in draft.inline or "method" in draft.bound)
                )
            ):
                # A stage that names its own upscale method does not inherit a
                # custom model: the model would silently replace the method.
                settings[key] = sticky.settings[key]
            elif (defaultGroup, key) in preDefaults:
                settings[key] = preDefaults[(defaultGroup, key)]
            elif key == "target":
                raise StageChainError(BARE_RESIZE_MESSAGE)
            else:
                settings[key] = context.defaults[(defaultGroup, key)]
        if "methods" in settings:
            settings["methods"] = tuple(settings["methods"])
        draft.settings = settings
        counts[draft.kind] += 1
        draft.index = counts[draft.kind]
        previous["interpolate" if draft.kind == "interpolate" else draft.kind] = draft
    for draft in drafts:
        draft.repeated = counts[draft.kind] > 1
    return drafts


# --- after the build -----------------------------------------------------------


def stageDrafts(args):
    return getattr(args, DRAFTS_ATTRIBUTE, None)


def firstDrafts(drafts):
    """The first draft of each chain kind, plus ``smooth`` for the smooth one."""
    first = {}
    for draft in drafts:
        first.setdefault(draft.kind, draft)
        if draft.smooth:
            first["smooth"] = draft
    return first


def syncStageGlobals(args, drafts=None):
    """Point the per-kind booleans and the global setting attributes at the
    chain: booleans are True iff a stage of the kind exists; the settings hold
    the FIRST instance's values, for code that is not per-stage.

    Also refreshes a provisional ``args.chain`` so output naming can read it
    before the final one is set.
    """
    if drafts is None:
        drafts = stageDrafts(args) or []
    perKind = Counter(draft.kind for draft in drafts)
    for draft in drafts:
        draft.repeated = perKind[draft.kind] > 1
    first = firstDrafts(drafts)
    for kind in ("dedup", "restore", "upscale", "interpolate", "resize"):
        setattr(args, kind, kind in first)
    args.smooth_dedup = "smooth" in first

    if "dedup" in first:
        s = first["dedup"].settings
        args.dedup_method = s["method"]
        if "threshold" in s:
            args.dedup_sens = s["threshold"]
            args.dedup_sens_raw = s["sens"]
        else:
            args.dedup_sens = s["sens"]
    if "restore" in first:
        args.restore_method = list(first["restore"].settings["methods"])
    if "upscale" in first:
        s = first["upscale"].settings
        args.upscale_method = s["method"]
        args.upscale_factor = s["factor"]
        args.custom_model = s["customModel"]
    if "interpolate" in first:
        s = first["interpolate"].settings
        args.interpolate_method = s["method"]
        args.interpolate_factor = s["factor"]
        args.ensemble = s["ensemble"]
        args.dynamic_scale = s["dynamicScale"]
        args.static_step = s["staticStep"]
    if "smooth" in first:
        s = first["smooth"].settings
        args.smooth_dedup_method = s["detector"]
        args.smooth_dedup_max_span = s["maxSpan"]
        if "threshold" in s:
            args.smooth_dedup_sens = s["threshold"]
            args.smooth_dedup_sens_raw = s["sens"]
        else:
            args.smooth_dedup_sens = s["sens"]
    args.chain = specsFromDrafts(drafts)


def specsFromDrafts(drafts):
    """The drafts as ``StageSpec``s, in order, with typed settings."""
    specs = []
    for draft in drafts:
        settings = dict(draft.settings)
        if draft.kind == "interpolate":
            settings["factor"] = float(settings["factor"])
            if draft.smooth:
                settings["sens"] = float(settings["sens"])
                settings["maxSpan"] = int(settings["maxSpan"])
        elif draft.kind == "upscale":
            settings["factor"] = int(settings["factor"])
            settings["customModel"] = settings["customModel"] or ""
        elif draft.kind == "dedup":
            settings["sens"] = float(settings["sens"])
        elif draft.kind == "restore":
            settings["methods"] = tuple(settings["methods"])
        specs.append(
            StageSpec(
                kind=draft.kind,
                index=draft.index or 1,
                settings=settings,
                smooth=draft.smooth,
            )
        )
    return tuple(specs)


def stageLabel(spec):
    """``interpolate #2 (rife4.26)`` / ``smooth_dedup (rife4.26)``."""
    name = "smooth_dedup" if spec.smooth else f"{spec.kind} #{spec.index}"
    summary = summarize(spec.kind, spec.settings)
    return f"{name} ({summary})" if summary else name


def stageOrderWarning(chain):
    """The one order warning: a dedup stage after an interpolate stage.

    Dedup after interpolation runs the interpolation model on every held
    drawing and then discards most of its in-between frames. The order is
    still honoured; this only says the inference is wasted.
    """
    kinds = [spec.kind if hasattr(spec, "kind") else spec for spec in chain]
    if "interpolate" not in kinds or "dedup" not in kinds:
        return None
    firstInterpolate = kinds.index("interpolate")
    if not any(kind == "dedup" for kind in kinds[firstInterpolate + 1 :]):
        return None
    return (
        "--dedup comes after --interpolate, so the in-between frames it "
        "interpolates across held drawings will mostly be discarded by dedup, "
        "wasting that inference. Put --dedup first, or use --smooth_dedup to "
        "interpolate across duplicates on purpose."
    )


def buildFromNamespace(args, parser):
    """Build the drafts from what argparse / --json recorded, store them on
    ``args`` and sync the globals. Raises :class:`StageChainError`."""
    context = ChainContext.fromParser(parser)
    drafts = buildStageDrafts(recordedEvents(args), context)
    setattr(args, DRAFTS_ATTRIBUTE, drafts)
    syncStageGlobals(args, drafts)
    logging.debug(
        "Stage drafts: %s",
        [(d.label, dict(d.settings)) for d in drafts],
    )
    return drafts


__all__ = [
    "BARE_RESIZE_MESSAGE",
    "CHILD_FLAGS",
    "ChainContext",
    "ChildEvent",
    "ChildFlagAction",
    "DRAFTS_ATTRIBUTE",
    "KIND_KEYS",
    "PER_STAGE_JSON_KEYS",
    "REMOVED_FLAG_MESSAGES",
    "RemovedFlagAction",
    "STAGE_BOOLEANS",
    "STAGE_FLAGS",
    "StageChainError",
    "StageDraft",
    "StageEvent",
    "StageFlagAction",
    "buildFromNamespace",
    "buildStageDrafts",
    "childFlag",
    "firstDrafts",
    "recordChild",
    "recordJsonChain",
    "recordStage",
    "recordedEvents",
    "removedFlag",
    "removedFlagMessage",
    "specsFromDrafts",
    "splitChainEntry",
    "stageDrafts",
    "stageFlag",
    "stageLabel",
    "stageOrderWarning",
    "summarize",
    "syncStageGlobals",
]
