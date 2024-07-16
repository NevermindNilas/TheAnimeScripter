import torch
from upscale_ncnn_py import UPSCALE

class UniversalNCNN:
    def __init__(
        self,
        upscaleMethod,
        upscaleFactor,
        upscaleSkip
    ):
        self.upscaleMethod = upscaleMethod
        self.upscaleFactor = upscaleFactor
        self.upscaleSkip = upscaleSkip

        match (self.upscaleMethod, self.upscaleFactor):
            case ("span-ncnn", 2):
                self.modelId = 4
            case ("span-ncnn", 4):
                self.modelId = 5
            case ("shufflecugan-ncnn", 2):
                self.modelId = 29
            case _:
                raise ValueError(
                    f"Invalid upscale method {self.upscaleMethod} with factor {self.upscaleFactor}"
                )

        self.model = UPSCALE(
            gpuid=0,
            tta_mode=False,
            tilesize=0,
            model=self.modelId,
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
            
        frame = self.model.process_cv2(frame.cpu().numpy())
        frame = torch.from_numpy(frame)

        if self.upscaleSkip is not None:
            self.prevFrame = frame

        return frame
    
    def getSkippedCounter(self):
        return self.skippedCounter

    

