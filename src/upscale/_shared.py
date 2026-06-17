import torch

from src.constants import ADOBE
from src.utils.isCudaInit import CudaChecker

if ADOBE:
    pass

checker = CudaChecker()

torch.set_float32_matmul_precision("medium")


def calculatePadding(width, height, multiple=4):
    padW = (multiple - (width % multiple)) % multiple
    padH = (multiple - (height % multiple)) % multiple
    return (0, padW, 0, padH)
