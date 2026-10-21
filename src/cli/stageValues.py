"""The value grammar of the stage flags and of the JSON ``"chain"`` entries.

A stage flag (``--dedup``, ``--restore``, ``--upscale``, ``--interpolate``,
``--smooth_dedup``, ``--resize``) takes zero or more values. Each value is
classified on its own, so their order is free. The order of the checks is:

1. ``key=value`` with a key the kind knows (bool keys also stand alone);
2. a method name (exact match);
3. a bool word (interpolate and smooth_dedup): ``ensemble``, ``dynamic_scale``,
   ``static_step``;
4. a size (resize only): ``WxH``, ``Np``, ``4k``, ``8k``;
5. a factor ``Nx`` / ``xN``;
6. a percent ``N%``;
7. a bare number;
8. a path (upscale only): a custom model file, which must exist;
9. anything else is an error naming the stage, the value and close matches.

What a form means depends on the kind (see ``FORM_MEANING``). The JSON
``"chain"`` entries go through exactly the same functions, so the two inputs
cannot drift apart.
"""

import difflib
import os
import re

from src.cli.validation import CliValidationError

# The grammar kinds: the stage flags, by their flag name.
GRAMMAR_KINDS = ("dedup", "restore", "upscale", "interpolate", "smooth_dedup", "resize")

# Extensions --custom_model accepts (validation.validateCustomModelPath).
MODEL_EXTENSIONS = (".pt", ".pth", ".ckpt", ".safetensors", ".onnx")

BOOL_KEYS = ("ensemble", "dynamic_scale", "static_step")

_INTERPOLATE_KEYS = {
    "method": "method",
    "factor": "factor",
    "ensemble": "ensemble",
    "dynamic_scale": "dynamicScale",
    "static_step": "staticStep",
}

# key typed by the user -> settings key on the StageSpec.
KEYS = {
    "dedup": {"method": "method", "sens": "sens"},
    "restore": {"method": "methods"},
    "upscale": {"method": "method", "factor": "factor", "model": "customModel"},
    "interpolate": dict(_INTERPOLATE_KEYS),
    "smooth_dedup": {
        **_INTERPOLATE_KEYS,
        "detector": "detector",
        "sens": "sens",
        "max_span": "maxSpan",
    },
    "resize": {"size": "target", "scale": "target"},
}

# settings key -> the name a message uses for it.
SETTING_NAMES = {
    "method": "method",
    "methods": "method",
    "factor": "factor",
    "customModel": "model",
    "ensemble": "ensemble",
    "dynamicScale": "dynamic_scale",
    "staticStep": "static_step",
    "detector": "detector",
    "sens": "sens",
    "maxSpan": "max_span",
    "target": "target (size or scale)",
}

# What an unkeyed size / factor / percent / bare number means for each kind.
# A form missing from a kind's table is an error for that kind.
FORM_MEANING = {
    "dedup": {"percent": "sens", "number": "sens"},
    "restore": {},
    "upscale": {"factor": "factor", "number": "factor"},
    "interpolate": {"factor": "factor", "number": "factor"},
    # A bare number is deliberately absent: it would be a factor for
    # interpolation and a sensitivity for detection, so it is refused.
    "smooth_dedup": {"factor": "factor", "percent": "sens"},
    "resize": {
        "size": "target",
        "factor": "target",
        "percent": "target",
        "number": "target",
    },
}

# What each kind accepts, for error messages.
ACCEPTS = {
    "dedup": "a dedup method, a sensitivity 0-100 (40 or 40%), method=, sens=",
    "restore": "one or more restore methods (run in the order given), method=",
    "upscale": (
        "an upscale method, a factor (2x, 3x, 4x or 2, 3, 4), a custom model "
        "path, method=, factor=, model="
    ),
    "interpolate": (
        "an interpolation method, a factor (2x, 2.5x or 2), ensemble, "
        "dynamic_scale, static_step, method=, factor="
    ),
    "smooth_dedup": (
        "an interpolation method, a detection method, a factor (2x), a "
        "sensitivity (40%), ensemble, dynamic_scale, static_step, method=, "
        "factor=, detector=, sens=, max_span="
    ),
    "resize": (
        "a size (1920x1080, 1440p, 4k, 8k) or a scale (0.5, 0.5x, 50%), size=, scale="
    ),
}

_FORM_NAMES = {
    "size": "a size",
    "factor": "a factor",
    "percent": "a percentage",
    "number": "a number",
    "path": "a file path",
}

_NUMBER = r"(?:\d+(?:\.\d*)?|\.\d+)"
_NUMBER_RE = re.compile(rf"^{_NUMBER}$")
_FACTOR_RE = re.compile(rf"^(?:({_NUMBER})[xX]|[xX]({_NUMBER}))$")
_PERCENT_RE = re.compile(rf"^({_NUMBER})%$")
_SIZE_RE = re.compile(r"^(\d+)[xX](\d+)$")
_HEIGHT_RE = re.compile(r"^(\d+)[pP]$")
_K_RE = re.compile(r"^([48])[kK]$")
_SIGNED_RE = re.compile(rf"^[+-](?:{_NUMBER}[xX%]?|[xX]{_NUMBER}|\d+[xX]\d+|\d+[pP])$")

_K_HEIGHTS = {"4": 2160, "8": 4320}

# Resize targets below this are refused; derived ones are clamped to it.
MIN_DIMENSION = 16


class StageChainError(CliValidationError):
    """A stage value, binding or chain rule was broken. The message names the
    stage and the value, and is meant to be shown as it is."""


class UnknownStageValue(StageChainError):
    """Step 9: a value that is no form the kind knows."""

    def __init__(self, message, value):
        super().__init__(message)
        self.value = value


def _number(text):
    """A plain decimal as int (no point) or float, else None."""
    if not _NUMBER_RE.match(text):
        return None
    if "." in text:
        return float(text)
    return int(text)


def detectForm(value):
    """``(form, payload)`` for the unkeyed numeric forms, or ``(None, None)``.

    ``size`` payloads are ``("size", w, h)`` or ``("height", h)``; the others
    are the number as int or float.
    """
    match = _SIZE_RE.match(value)
    if match:
        return "size", ("size", int(match.group(1)), int(match.group(2)))
    match = _HEIGHT_RE.match(value)
    if match:
        return "size", ("height", int(match.group(1)))
    match = _K_RE.match(value)
    if match:
        return "size", ("height", _K_HEIGHTS[match.group(1)])
    match = _FACTOR_RE.match(value)
    if match:
        return "factor", _number(match.group(1) or match.group(2))
    match = _PERCENT_RE.match(value)
    if match:
        return "percent", _number(match.group(1))
    number = _number(value)
    if number is not None:
        return "number", number
    return None, None


def looksLikePath(value):
    return "/" in value or "\\" in value or value.lower().endswith(MODEL_EXTENSIONS)


def kindLabel(kind, occurrence=None):
    """``upscale #2`` / ``smooth_dedup`` (there is only ever one)."""
    if kind == "smooth_dedup" or occurrence is None:
        return kind
    return f"{kind} #{occurrence}"


class ValueContext:
    """The method lists the grammar matches against, keyed by grammar kind.

    ``methods["smooth_dedup"]`` is the detection list; interpolation methods
    for the smooth stage come from ``methods["interpolate"]``.
    """

    def __init__(self, methods):
        self.methods = {kind: tuple(values) for kind, values in methods.items()}

    @classmethod
    def fromParser(cls, parser):
        from src.cli.parser import capabilityMethods

        caps = capabilityMethods(parser)
        return cls(
            {
                kind: caps.get(kind, ())
                for kind in ("dedup", "restore", "upscale", "interpolate")
            }
            | {"smooth_dedup": caps.get("smooth_dedup", ())}
        )

    def methodKey(self, kind, value):
        """The settings key a bare method name sets on ``kind``, or None."""
        if kind == "smooth_dedup":
            if value in self.methods["interpolate"]:
                return "method"
            if value in self.methods["smooth_dedup"]:
                return "detector"
            return None
        if kind == "resize":
            return None
        if value in self.methods[kind]:
            return "methods" if kind == "restore" else "method"
        return None

    def vocabulary(self, kind):
        """Everything a typo of a ``kind`` value may have meant."""
        words = list(KEYS[kind])
        if kind in ("interpolate", "smooth_dedup"):
            words += list(self.methods["interpolate"])
        if kind == "smooth_dedup":
            words += list(self.methods["smooth_dedup"])
        elif kind in ("dedup", "restore", "upscale"):
            words += list(self.methods[kind])
        return words


# --- value conversion and range checks -------------------------------------


def _fmt(value):
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _checkSens(label, number, text):
    if not 0 <= number <= 100:
        raise StageChainError(
            f"{label}: sensitivity '{text}' is out of range; it must be 0-100."
        )
    return float(number)


def _checkUpscaleFactor(label, number, text):
    if float(number) != int(number) or int(number) not in (2, 3, 4):
        raise StageChainError(
            f"{label}: upscale factor '{text}' is not supported; use 2x, 3x or 4x."
        )
    return int(number)


def _checkInterpolateFactor(label, number, text):
    if number <= 0:
        raise StageChainError(
            f"{label}: interpolation factor '{text}' must be greater than 0."
        )
    return number


def _checkScale(label, number, text):
    if number <= 0:
        raise StageChainError(f"{label}: resize scale '{text}' must be greater than 0.")
    return ("scale", float(number))


def _checkDimension(label, value, text, what):
    if value < MIN_DIMENSION or value % 2:
        raise StageChainError(
            f"{label}: {what} in '{text}' must be an even number of at least "
            f"{MIN_DIMENSION} (got {value})."
        )


def _checkSize(label, payload, text):
    if payload[0] == "size":
        _checkDimension(label, payload[1], text, "the width")
        _checkDimension(label, payload[2], text, "the height")
    else:
        _checkDimension(label, payload[1], text, "the height")
    return payload


def _resizeTarget(label, form, number, text):
    if form == "size":
        return _checkSize(label, number, text)
    if form == "percent":
        return _checkScale(label, number / 100.0, text)
    return _checkScale(label, number, text)


def _modelPath(label, text):
    if not text:
        raise StageChainError(f"{label}: model= needs a file path.")
    path = os.path.abspath(os.path.expanduser(text))
    if not os.path.isfile(path):
        raise StageChainError(f"{label}: custom model file not found: {path}")
    return path


def _str2bool(label, key, text):
    from src.cli.parser import str2bool

    try:
        return str2bool(text)
    except Exception:
        raise StageChainError(
            f"{label}: '{key}={text}' is not a yes/no value; use {key}=true or "
            f"{key}=false (or {key} alone)."
        ) from None


def _numericKeyed(label, kind, key, text, forms):
    """``factor=`` / ``sens=`` / ``scale=``: one of the given unkeyed forms."""
    form, number = detectForm(text)
    if form not in forms:
        raise StageChainError(
            f"{label}: '{key}={text}' is not a valid {key}. "
            f"{kind} takes {ACCEPTS[kind]}."
        )
    return form, number


def _parseKeyed(kind, label, key, text, context):
    """``key=value`` for a key ``kind`` knows. Returns ``(settingsKey, value)``."""
    settingsKey = KEYS[kind][key]
    if key in BOOL_KEYS:
        return settingsKey, _str2bool(label, key, text)
    if key == "method":
        methods = context.methods["interpolate" if kind == "smooth_dedup" else kind]
        if text not in methods:
            raise UnknownStageValue(
                _unknownMessage(label, f"method={text}", text, methods, ()), text
            )
        return settingsKey, text
    if key == "detector":
        methods = context.methods["smooth_dedup"]
        if text not in methods:
            raise UnknownStageValue(
                _unknownMessage(label, f"detector={text}", text, methods, ()), text
            )
        return settingsKey, text
    if key == "model":
        return settingsKey, _modelPath(label, text)
    if key == "max_span":
        if not re.match(r"^[+-]?\d+$", text):
            raise StageChainError(
                f"{label}: 'max_span={text}' must be a whole number of frames."
            )
        return settingsKey, int(text)
    if key == "sens":
        _form, number = _numericKeyed(label, kind, key, text, ("number", "percent"))
        return settingsKey, _checkSens(label, number, text)
    if key == "factor":
        _form, number = _numericKeyed(label, kind, key, text, ("number", "factor"))
        if kind == "upscale":
            return settingsKey, _checkUpscaleFactor(label, number, text)
        return settingsKey, _checkInterpolateFactor(label, number, text)
    if key == "size":
        form, payload = _numericKeyed(label, kind, key, text, ("size",))
        return settingsKey, _checkSize(label, payload, text)
    if key == "scale":
        form, number = _numericKeyed(
            label, kind, key, text, ("number", "factor", "percent")
        )
        return settingsKey, _resizeTarget(label, form, number, text)
    raise AssertionError(f"unhandled key {key}")  # pragma: no cover


def _unknownMessage(label, shown, value, vocabulary, neighbours):
    """Step 9: name the stage and the value, and say what it may have meant."""
    message = f"{label}: unknown value '{shown}'."
    joined = []
    for neighbour, before in neighbours:
        for separator in ("", "-", ".", "_"):
            candidate = (
                f"{neighbour}{separator}{value}"
                if before
                else f"{value}{separator}{neighbour}"
            )
            if candidate in vocabulary and candidate not in joined:
                joined.append(candidate)
    if joined:
        message += (
            f" Did you mean {' or '.join(repr(j) for j in joined)}? A method "
            "name is one value; do not split it with a space."
        )
        return message
    lowered = {word.lower(): word for word in vocabulary}
    close = difflib.get_close_matches(value.lower(), list(lowered), n=3, cutoff=0.6)
    if close:
        message += f" Did you mean {', '.join(repr(lowered[c]) for c in close)}?"
    return message


def classifyValue(kind, value, label, context, neighbours=()):
    """Classify one value of a ``kind`` stage.

    Returns ``(settingsKey, value)``. ``neighbours`` are ``(text, before)``
    pairs of the adjacent values, used only to spot a method split by a space.
    """
    if not isinstance(value, str):
        raise StageChainError(f"{label}: values must be text, got {value!r}.")
    if value.strip() == "":
        raise StageChainError(f"{label}: empty value.")
    value = value.strip()

    # 1. key=value
    if "=" in value:
        key, _, rest = value.partition("=")
        key = key.strip().lower()
        if key in KEYS[kind]:
            return _parseKeyed(kind, label, key, rest.strip(), context)
        if kind == "upscale" and (os.path.isfile(value) or looksLikePath(value)):
            return "customModel", _modelPath(label, value)
        message = (
            f"{label}: unknown key '{key}' in '{value}'. "
            f"{kind} keys: {', '.join(KEYS[kind])}."
        )
        close = difflib.get_close_matches(key, list(KEYS[kind]), n=1, cutoff=0.6)
        if close:
            message += f" Did you mean '{close[0]}='?"
        raise StageChainError(message)

    # 2. a method name
    methodKey = context.methodKey(kind, value)
    if methodKey is not None:
        return methodKey, value

    # 3. bool words
    if kind in ("interpolate", "smooth_dedup") and value in BOOL_KEYS:
        return KEYS[kind][value], True

    # 4-7. size, factor, percent, bare number
    form, number = detectForm(value)
    if form is not None:
        meaning = FORM_MEANING[kind].get(form)
        if meaning is None:
            if kind == "smooth_dedup" and form == "number":
                raise StageChainError(
                    f"{label}: '{value}' is ambiguous: write {value}x for the "
                    f"interpolation factor, or {value}% (or sens={value}) for the "
                    "duplicate-detection sensitivity."
                )
            raise StageChainError(
                f"{label}: '{value}' is {_FORM_NAMES[form]}, which {kind} does "
                f"not take. {kind} takes {ACCEPTS[kind]}."
            )
        if kind == "resize":
            return "target", _resizeTarget(label, form, number, value)
        if meaning == "sens":
            return "sens", _checkSens(label, number, value)
        if kind == "upscale":
            return "factor", _checkUpscaleFactor(label, number, value)
        return "factor", _checkInterpolateFactor(label, number, value)

    # 8. a path
    if looksLikePath(value):
        if kind == "upscale":
            return "customModel", _modelPath(label, value)
        raise StageChainError(
            f"{label}: '{value}' looks like a file path; only --upscale takes a "
            "custom model file."
        )

    # 9. an error, with what it may have meant
    if _SIGNED_RE.match(value):
        raise StageChainError(
            f"{label}: '{value}' is not allowed; values must be positive and unsigned."
        )
    bare = re.sub(r"(?i)^x|[x%]$", "", value)
    try:
        float(bare)
    except ValueError:
        pass
    else:
        # 1e3, inf, nan, 1_000: float() takes them, the grammar does not.
        raise StageChainError(
            f"{label}: '{value}' is not a plain number; write numbers as plain "
            "decimals such as 2, 2.5 or 0.5."
        )
    raise UnknownStageValue(
        _unknownMessage(
            label, value, value, context.vocabulary(kind), list(neighbours)
        ),
        value,
    )


def classifyOccurrence(kind, values, label, context):
    """Classify every value of one stage occurrence.

    Returns ``(inline, texts)``: settings key -> value, and settings key -> the
    value text that set it (for messages). Restore methods accumulate into a
    tuple. Any other setting given twice with different values is an error;
    the same value twice is allowed.

    An unknown value is reported before any other error, so a space-split
    method next to a bare number (``rife 4.26``) is reported as the method,
    and the number is never used.
    """
    values = list(values)
    results = []
    unknown = None
    other = None
    for position, value in enumerate(values):
        neighbours = []
        if position > 0:
            neighbours.append((str(values[position - 1]).strip(), True))
        if position + 1 < len(values):
            neighbours.append((str(values[position + 1]).strip(), False))
        try:
            results.append(
                (value, *classifyValue(kind, value, label, context, neighbours))
            )
        except UnknownStageValue as error:
            unknown = unknown or error
        except StageChainError as error:
            other = other or error
    if unknown is not None:
        raise unknown
    if other is not None:
        raise other

    inline = {}
    texts = {}
    for text, key, value in results:
        if key == "methods":
            inline[key] = (*inline.get(key, ()), value)
            texts[key] = " ".join(inline[key])
            continue
        if key in inline and inline[key] != value:
            raise StageChainError(
                f"{label}: {SETTING_NAMES[key]} given twice ('{texts[key]}' and "
                f"'{text}'). Give each setting once per stage; repeat the flag "
                "for another stage."
            )
        inline[key] = value
        texts[key] = str(text).strip()
    return inline, texts


def describeTarget(target):
    """``1440p`` / ``1920x1080`` / ``0.5x`` for a resize target."""
    if target[0] == "size":
        return f"{target[1]}x{target[2]}"
    if target[0] == "height":
        return f"{target[1]}p"
    return f"{_fmt(target[1])}x"


__all__ = [
    "ACCEPTS",
    "BOOL_KEYS",
    "FORM_MEANING",
    "GRAMMAR_KINDS",
    "KEYS",
    "MIN_DIMENSION",
    "MODEL_EXTENSIONS",
    "SETTING_NAMES",
    "StageChainError",
    "UnknownStageValue",
    "ValueContext",
    "classifyOccurrence",
    "classifyValue",
    "describeTarget",
    "detectForm",
    "kindLabel",
    "looksLikePath",
]
