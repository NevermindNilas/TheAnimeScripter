import os
import torch
import logging

from src.utils.modelOptimizer import ModelOptimizer
from src.utils.downloadModels import downloadModels, weightsDir, modelsMap, resolveWeightPath
from src.utils.isCudaInit import CudaChecker
from src.utils.logAndPrint import logAndPrint
from src.constants import ADOBE

if ADOBE:
    from src.utils.aeComms import progressState

checker = CudaChecker()

torch.set_float32_matmul_precision("medium")


def calculatePadding(width, height, multiple=4):
    padW = (multiple - (width % multiple)) % multiple
    padH = (multiple - (height % multiple)) % multiple
    return (0, padW, 0, padH)


