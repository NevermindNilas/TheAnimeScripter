import torch
from upscale_ncnn_py import UPSCALE

class UniversalNCNN:
    def __init__(
        self,
        upscaleMethod,
        upscaleFactor,
        nt,
    ):
        self.upscaleMethod = upscaleMethod
        self.upscaleFactor = upscaleFactor
        self.nt = nt

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

        if self.nt == 1:
            self.nt == 2

        self.model = UPSCALE(
            gpuid=0,
            tta_mode=False,
            tilesize=0,
            model=self.modelId,
            num_threads=self.nt,
        )

    def run(self, frame):
        frame = self.model.process_cv2(frame.cpu().numpy())
        return torch.from_numpy(frame).permute(2, 0, 1)
