import logging

from src.infra.logAndPrint import logAndPrint

BACKEND_SUFFIXES = ("-directml", "-ncnn", "-tensorrt", "-mps", "-openvino")

# Preference order when downgrading a CUDA method, fastest first.
FALLBACK_SUFFIXES = ("-directml", "-ncnn")

# RIFE is the exception: its NCNN (Vulkan) path runs roughly an order of
# magnitude faster than its DirectML one (~265 fps against ~10-13 fps at 1080p),
# so interpolation must not be handed to DirectML just because a choice exists.
CAPABILITY_FALLBACK_SUFFIXES = {
    "interpolate": ("-ncnn", "-directml"),
}

MPS_SUFFIX = "-mps"

# Capabilities whose methods all have a working CPU path, so a method surviving
# the downgrade is not a problem worth warning about.
CPU_CAPABLE_CAPABILITIES = frozenset({"dedup"})

# Not auto-selected (OpenVINO is flagged experimental in the backends), but
# worth naming when telling the user what else they could pick.
SUGGESTABLE_SUFFIXES = ("-directml", "-ncnn", "-openvino", "-mps")

METHOD_ATTRIBUTES = {
    "interpolate_method": "interpolate",
    "upscale_method": "upscale",
    "segment_method": "segment",
    "depth_method": "depth",
    "restore_method": "restore",
    "dedup_method": "dedup",
    "obj_detect_method": "obj_detect",
}


def hasExplicitBackend(method):
    return any(backend in method.lower() for backend in BACKEND_SUFFIXES)


def _suffixOrder(preferMps, capability=None):
    order = CAPABILITY_FALLBACK_SUFFIXES.get(capability, FALLBACK_SUFFIXES)
    return (MPS_SUFFIX, *order) if preferMps else order


def fallbackMethod(
    method, availableModels, preferMps=False, choices=None, capability=None
):
    """Rewrite ``method`` to a non-CUDA sibling, or return it unchanged.

    ``choices`` is the CLI choice list for the flag being adjusted. Prefer it
    over ``availableModels``: the weight registry answers "are there weights
    under this name", not "can the factory build this method", and the two
    disagree. ``--segment`` is the worst case -- its registry names are
    ``segment*`` while its CLI names are ``anime*``, so no ``anime-directml``
    was ever found and the default ``--segment`` flag went on to construct
    ``AnimeSegment``, whose unguarded ``torch.cuda.Stream()`` then raised on
    every machine without CUDA. ``modelsList()`` likewise carries no
    ``rife*-directml`` entries at all.
    """
    base = method.lower()
    candidates = set(choices) if choices else set(availableModels)

    for suffix in _suffixOrder(preferMps, capability):
        candidate = f"{base}{suffix}"
        if candidate in candidates:
            return candidate

    return method


def _warnNoFallback(attr, method, choices, preferMps, capability):
    """Say so out loud when a CUDA-only method survives the downgrade.

    Silence here meant an opaque ``torch.cuda.Stream()`` traceback several
    seconds into the run, on a machine the user already knows has no usable
    CUDA.

    Stays quiet for capabilities whose every method has a CPU path (dedup's
    ssim/mse/vmaf all run on CPU by design), because there is nothing wrong to
    report. The suggestions include ``-openvino``, which is a real, wired-up
    choice even though it is never auto-selected, but exclude ``-mps`` off
    Darwin -- pointing a Windows user at an Apple Silicon backend is worse than
    saying nothing.
    """
    if capability in CPU_CAPABLE_CAPABILITIES:
        return

    tried = _suffixOrder(preferMps, capability)
    suggestable = tuple(
        suffix for suffix in SUGGESTABLE_SUFFIXES if suffix != MPS_SUFFIX or preferMps
    )
    alternatives = sorted(
        choice
        for choice in (choices or ())
        if any(choice.endswith(suffix) for suffix in suggestable)
    )
    message = (
        f"'{method}' has no {'/'.join(s.lstrip('-') for s in tried)} variant, so "
        f"--{attr} is unchanged and this run will fail if it needs CUDA."
    )
    if alternatives:
        shown = ", ".join(alternatives[:6])
        more = len(alternatives) - 6
        message += f" Alternatives: {shown}"
        message += f" (+{more} more, see --list_methods)." if more > 0 else "."
    logAndPrint(message, "yellow", level="WARNING")


def applyBackendFallbacks(args, availableModels, preferMps=False, methodChoices=None):
    methodChoices = methodChoices or {}

    for attr, flagName in METHOD_ATTRIBUTES.items():
        if not getattr(args, flagName, False):
            continue

        currentMethod = getattr(args, attr)
        choices = methodChoices.get(flagName)

        if attr == "restore_method" and isinstance(currentMethod, list):
            adjusted = []
            for method in currentMethod:
                if hasExplicitBackend(method):
                    logging.info(f"{attr} method {method} already uses a backend")
                    adjusted.append(method)
                    continue

                newMethod = fallbackMethod(
                    method,
                    availableModels,
                    preferMps=preferMps,
                    choices=choices,
                    capability=flagName,
                )
                if newMethod != method:
                    logging.info(f"Adjusted {attr} method from {method} to {newMethod}")
                else:
                    _warnNoFallback(attr, method, choices, preferMps, flagName)
                adjusted.append(newMethod)
            setattr(args, attr, adjusted)
            continue

        if hasExplicitBackend(currentMethod):
            logging.info(f"{attr} already uses a backend: {currentMethod}")
            continue

        newMethod = fallbackMethod(
            currentMethod,
            availableModels,
            preferMps=preferMps,
            choices=choices,
            capability=flagName,
        )
        if newMethod != currentMethod:
            logging.info(f"Adjusted {attr} from {currentMethod} to {newMethod}")
            setattr(args, attr, newMethod)
        else:
            logging.info(
                f"No adjustment for {attr} ({currentMethod} remains unchanged)"
            )
            _warnNoFallback(attr, currentMethod, choices, preferMps, flagName)

    if getattr(args, "moblur", False):
        moblurMethod = args.moblur_method
        if not any(
            backend in moblurMethod for backend in ("-directml", "-openvino", "-mps")
        ):
            base = moblurMethod.replace("-tensorrt", "")
            # Checked against the choice list rather than assigned blind: this
            # used to append "-directml" unconditionally, so any method without
            # a DirectML sibling was rewritten to a string nothing can build.
            # Every current --moblur_method choice happens to have one, so this
            # is a guard against the next one that does not.
            newMethod = fallbackMethod(
                base,
                availableModels,
                preferMps=preferMps,
                choices=methodChoices.get("moblur"),
                capability="moblur",
            )
            if newMethod != base:
                args.moblur_method = newMethod
                logging.info(
                    f"Adjusted moblur_method from {moblurMethod} to "
                    f"{args.moblur_method} because CUDA is unavailable"
                )
            else:
                # `base`, not `moblurMethod`: base is what was actually
                # probed, so naming the other one would report a string we
                # never looked up.
                _warnNoFallback(
                    "moblur_method",
                    base,
                    methodChoices.get("moblur"),
                    preferMps,
                    "moblur",
                )
