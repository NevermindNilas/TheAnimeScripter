import torch
import os

from upscale_ncnn_py import UPSCALE
from .downloadModels import downloadModels, weightsDir, modelsMap
from src.coloredPrints import green


class UniversalNCNN:
    def __init__(self, upscaleMethod, upscaleFactor, upscaleSkip):
        self.upscaleMethod = upscaleMethod
        self.upscaleFactor = upscaleFactor
        self.upscaleSkip = upscaleSkip

        self.filename = modelsMap(
            self.upscaleMethod,
            modelType="ncnn",
        )

        if self.filename.endswith("-ncnn.zip"):
            self.filename = self.filename[:-9]
        elif self.filename.endswith("-ncnn"):
            self.filename = self.filename[:-5]

        if not os.path.exists(
            os.path.join(weightsDir, self.upscaleMethod, self.filename)
        ):
            modelPath = downloadModels(
                model=self.upscaleMethod,
                modelType="ncnn",
            )
        else:
            modelPath = os.path.join(weightsDir, self.upscaleMethod, self.filename)

        if modelPath.endswith("-ncnn.zip"):
            modelPath = modelPath[:-9]
        elif modelPath.endswith("-ncnn"):
            modelPath = modelPath[:-5]

        lastSlash = modelPath.split("\\")[-1]
        modelPath = modelPath + "\\" + lastSlash

        self.model = UPSCALE(
            gpuid=0,
            tta_mode=False,
            tilesize=0,
            model_str=modelPath,
            num_threads=2,
        )

        if self.upscaleSkip is not None:
            self.skippedCounter = 0
            self.prevFrame = None

    def run(self, frame):
        if self.upscaleSkip is not None:
            if self.upscaleSkip.run(frame):
                self.skippedCounter += 1
                return self.prevFrame

        frame = self.model.process_torch(frame)

        if self.upscaleSkip is not None:
            self.prevFrame = frame

        return frame

    def getSkippedCounter(self):
        return self.skippedCounter
