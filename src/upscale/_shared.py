import os
import torch
import logging

from src.model.modelOptimizer import ModelOptimizer
from src.model.download import downloadModels, resolveWeightPath
from src.model.registry import weightsDir, modelsMap
from src.infra.isCudaInit import CudaChecker
from src.infra.logAndPrint import logAndPrint
from src.constants import ADOBE

if ADOBE:
    from src.server.aeComms import progressState

checker = CudaChecker()

torch.set_float32_matmul_precision("medium")


def calculatePadding(width, height, multiple=4):
    padW = (multiple - (width % multiple)) % multiple
    padH = (multiple - (height % multiple)) % multiple
    return (0, padW, 0, padH)


