"""
ONNX Runtime execution-provider verification.

onnxruntime will happily construct an InferenceSession with a requested GPU
execution provider and then silently place every node on CPU when that provider
cannot actually serve the model (missing driver, unsupported op, ABI mismatch).
The only user-visible symptom is a ~30x slowdown with nothing on the console,
because the root logger writes to file only.

``get_providers()`` reports the providers the session actually got, so we compare
that against what was requested and emit a visible warning when the GPU provider
is absent. Logging/observability only -- this never raises and never changes
inference behaviour, device placement, or dtype.
"""

from src.infra.logAndPrint import logWarning


def warnIfProviderMissing(session, requested: str, context: str) -> bool:
    """Warn (visibly) if `requested` provider is not active on `session`.

    Returns True when the provider is present, False when it is missing and a
    warning was emitted.
    """
    try:
        active = session.get_providers()
    except Exception:
        # Never let an observability check break real work.
        return True

    if requested in active:
        return True

    logWarning(
        f"{context}: requested {requested} but ONNX Runtime fell back to {active}. "
        f"Expect significantly worse performance."
    )
    return False
