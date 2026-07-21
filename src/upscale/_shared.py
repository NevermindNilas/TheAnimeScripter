import torch

from src.infra.isCudaInit import CudaChecker

checker = CudaChecker()

torch.set_float32_matmul_precision("medium")


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
# the probe, not prose. Methods absent from this table (gauss, figsr,
# shufflespan, custom models, future archs) fall back to the runtime probe.
KNOWN_INPUT_MULTIPLES = {
    "adore": 4,
    "fallin_soft": 4,
    "fallin_strong": 4,
    "shufflecugan": 4,
    "aniscale2": 1,
    "open-proteus": 1,
    "span": 1,
    "rtmosr": 1,
    "saryn": 1,  # same RTMoSR arch as rtmosr
    "smosr": 1,
}

_BACKEND_SUFFIXES = ("-tensorrt", "-directml", "-openvino", "-ncnn", "-mps")


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
