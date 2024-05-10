import os

from rife_ncnn_vulkan_python import Rife
from src.downloadModels import downloadModels, modelsMap, weightsDir


class rifeNCNN:
    def __init__(
        self, interpolateMethod, ensemble=False, nt=1, width=1920, height=1080
    ):
        self.interpolateMethod = interpolateMethod
        self.nt = nt
        self.height = height
        self.width = width
        self.ensemble = ensemble

        # Since the built in models folder use the default naming without
        # Ncnn or other suffixes, we need to change the name to match the
        # folder name

        UHD = True if width >= 3840 or height >= 2160 else False
        scale = 2 if UHD else 1

        match interpolateMethod:
            case "rife4.15-ncnn" | "rife-ncnn":
                self.interpolateMethod = "rife-v4.15-ncnn"
            case "rife4.6-ncnn":
                self.interpolateMethod = "rife-v4.6-ncnn"
            case "rife4.15-lite-ncnn":
                self.interpolateMethod = "rife-v4.15-lite-ncnn"
            case "rife4.16-lite-ncnn":
                self.interpolateMethod = "rife-v4.16-lite-ncnn"

        self.filename = modelsMap(
            self.interpolateMethod,
            ensemble=self.ensemble,
        )

        if not os.path.exists(
            os.path.join(weightsDir, self.interpolateMethod)
        ):
            modelPath = downloadModels(
                model=self.interpolateMethod,
                ensemble=self.ensemble,
            )
        else:
            modelPath = os.path.join(weightsDir, self.interpolateMethod, self.filename)

        if modelPath.endswith("-ncnn.zip"):
            modelPath = modelPath[:-9]
        elif modelPath.endswith("-ncnn"):
            modelPath = modelPath[:-5]        

        self.rife = Rife(
            gpuid=0,
            model=modelPath,
            scale=scale,
            tta_mode=False,
            tta_temporal_mode=False,
            uhd_mode=UHD,
            num_threads=self.nt,
        )

        self.frame1 = None
        self.shape = (self.height, self.width)

    def make_inference(self, timestep):
        output = self.rife.process_fast(self.frame1, self.frame2, timestep=timestep)

        return output

    def cacheFrame(self):
        self.frame1 = self.frame2.copy()

    def run(self, frame):
        if self.frame1 is None:
            self.frame1 = frame
            return False

        self.frame2 = frame
        return True
