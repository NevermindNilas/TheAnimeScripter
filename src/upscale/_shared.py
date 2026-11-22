"""Torch-free helpers for upscale backends (padding, input multiples).

Deliberately free of torch/CUDA imports: backend modules import these names
function-level lazy, and any module-level torch import here would run on every
such import -- breaking torch-less loading (bare CI venv) for helpers that
are pure math. Backend constructors own `set_float32_matmul_precision`, not
this module.
"""


def calculatePadding(width, height, multiple=4):
    padW = (multiple - (width % multiple)) % multiple
    padH = (multiple - (height % multiple)) % multiple
    return (0, padW, 0, padH)


# Required input multiple per shipped arch, established empirically by running
# the smallestValidMultiple probe against the real weights (ONNX CPU sessions
# and spandrel-loaded .pth, 2026-07). The RealCUGAN-family cascade
# (adore/fallin_*/shufflecugan) needs mod-4 input; the others listed are fully
# convolutional and accept any size. NOTE: aniscale2 is a Compact arch
# (multiple 1) despite the 3b30f9e1 commit message grouping it with the
# RealCUGAN family, and adore needs 4 despite not being named there — trust
# the probe, not prose. Methods absent from this table (gauss, figsr, custom
# models, future archs) fall back to the runtime probe.
KNOWN_INPUT_MULTIPLES = {
    "adore": 4,
    "fallin_soft": 4,
    "fallin_strong": 4,
    "shufflecugan": 4,
    "aniscale2": 1,
    "open-proteus": 1,
    "span": 1,
    "cyte": 1,
    "rtmosr": 1,
    "saryn": 1,  # same RTMoSR arch as rtmosr
    "smosr": 1,
}

_BACKEND_SUFFIXES = (
    "-tensorrt",
    "-directml",
    "-openvino",
    "-ncnn",
    "-mps",
    "-rocm",
    "-kernel",
)


def lookupRequiredMultiple(upscaleMethod: str | None) -> int | None:
    """Table lookup of an arch's required input multiple, or None to probe.

    Strips the backend suffix so all variants of a method share one entry.
    Returns None for methods not in the table (including custom models), which
    callers must resolve with the smallestValidMultiple probe.
    """
    if not upscaleMethod:
        return None
    base = upscaleMethod
    for suffix in _BACKEND_SUFFIXES:
        base = base.removesuffix(suffix)
    return KNOWN_INPUT_MULTIPLES.get(base)


def resolveRequiredMultiple(upscaleMethod, customModel, makeRunOK):
    """The arch's required input multiple: the table for shipped archs, else the
    ``smallestValidMultiple`` probe. ``makeRunOK()`` builds the probe's runOK and
    is only called on a miss, so a table hit never builds a probe session (an
    ONNX one costs ~90-130ms of startup)."""
    if not customModel:
        known = lookupRequiredMultiple(upscaleMethod)
        if known is not None:
            return known
    return smallestValidMultiple(makeRunOK())


def customScaleRefusal(customModel, modelScale, upscaleFactor):
    """Why custom model ``customModel`` cannot run at ``upscaleFactor``x, or None.

    Every later stage and the writer are sized at the stage's factor, so a 4x
    model on a 2x stage (the default) squashed its output to 2x in the writer,
    crashed the next stage on a size mismatch, or, on TensorRT, had the engine
    write a 4x frame into the 2x output buffer. ``modelScale`` None (unknown)
    passes. The PyTorch backends let a 1x model through themselves (its frame
    was always resized by the writer); TensorRT and DirectML never ran one.
    """
    if modelScale is None or int(modelScale) == int(upscaleFactor):
        return None
    import os

    name = os.path.basename(str(customModel))
    message = (
        f"Custom model {name} upscales {modelScale}x, but its --upscale stage "
        f"is {upscaleFactor}x."
    )
    if int(modelScale) in (2, 3, 4):
        return message + (
            f" Give the stage {modelScale}x (--upscale <method> {modelScale}x "
            f"--custom_model {name})."
        )
    if int(modelScale) == 1:
        return message + (
            " A 1x model only runs on a PyTorch upscale method (no -tensorrt, "
            "-directml or -openvino suffix) from its .pth/.safetensors file."
        )
    return message + " Upscale stages take 2x, 3x or 4x models."


def onnxOutputScale(modelPath, half, multiple=1):
    """The upscale factor of the ONNX model at ``modelPath``, or None if unknown.

    A model with static input/output widths is read from its declared shapes;
    a dynamic one from one tiny CPU run. A run the model refuses (a static
    shape the declaration did not expose, an input floor) gives None, so the
    caller lets it through rather than refusing a model that ran before.
    """
    import numpy as np
    import onnxruntime as ort

    probe = ort.InferenceSession(modelPath, providers=["CPUExecutionProvider"])
    inputWidth = probe.get_inputs()[0].shape[-1]
    outputWidth = probe.get_outputs()[0].shape[-1]
    if isinstance(inputWidth, int) and isinstance(outputWidth, int):
        return outputWidth // inputWidth if inputWidth > 0 else None
    side = multiple * -(-48 // multiple)
    inputs = {
        probe.get_inputs()[0].name: np.zeros(
            (1, 3, side, side), np.float16 if half else np.float32
        )
    }
    try:
        return probe.run(None, inputs)[0].shape[-1] // side
    except Exception:
        return None


def onnxProbe(modelPath, half):
    """``smallestValidMultiple``'s runOK over a throwaway CPU ORT session."""
    import numpy as np
    import onnxruntime as ort

    probe = ort.InferenceSession(modelPath, providers=["CPUExecutionProvider"])
    inputName = probe.get_inputs()[0].name
    npDType = np.float16 if half else np.float32

    def runOK(h, w):
        try:
            probe.run(None, {inputName: np.zeros((1, 3, h, w), npDType)})
            return True
        except Exception:
            return False

    return runOK


def smallestValidMultiple(runOK, candidates=(1, 2, 4, 8, 16), floor=48):
    """
    Detect the spatial multiple an upscale arch requires its input to satisfy.

    Some archs (the RealCUGAN family: fallin_*, shufflecugan, aniscale2) run an
    internal 2-level down/up cascade whose skip-connection ``Add`` only lines up
    when both input dims are divisible by 4; feeding an odd dim raises a broadcast
    error (``274 by 275``). Fully-convolutional archs (compact/SPAN) accept any
    size and return 1 here, so callers add zero padding and keep bit-exact parity.

    The probe is resolution-independent: for each candidate ``m`` it runs a tiny
    square that is an *odd* multiple of ``m`` (so it is a multiple of ``m`` but not
    of ``2m``) and at least ``floor`` px (models reflect-pad internally and reject
    inputs smaller than that). ``runOK(h, w)`` must run one inference and return
    whether it succeeded.
    """

    def oddMultiple(m):
        k = 1
        while m * k < floor or k % 2 == 0:
            k += 1
        return m * k

    for m in candidates:
        side = oddMultiple(m)
        if runOK(side, side):
            return m
    return candidates[-1]
