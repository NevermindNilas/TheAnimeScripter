import torch

from src.infra.isCudaInit import CudaChecker

checker = CudaChecker()

torch.set_float32_matmul_precision("medium")


def calculatePadding(width, height, multiple=4):
    padW = (multiple - (width % multiple)) % multiple
    padH = (multiple - (height % multiple)) % multiple
    return (0, padW, 0, padH)


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
