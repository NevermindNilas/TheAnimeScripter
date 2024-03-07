import realesrgan_ncnn_py as renp

from realcugan_ncnn_py import Realcugan
from span_ncnn_py import Span

class CuganNCNN:
    def __init__(self, num_threads, upscale_factor):
        """
        Barebones for now
        """
        self.num_threads = num_threads
        self.upscale_factor = upscale_factor

        self.realcugan = Realcugan(
            num_threads=self.num_threads,
            gpuid=0,
            tta_mode=False,
            scale=self.upscale_factor,
        )

    def run(self, frame):
        frame = self.realcugan.process_cv2(frame)
        return frame

class SpanNCNN:
    def __init__(
        self,
        upscale_factor,
        half,
    ):
        self.upscale_factor = upscale_factor
        self.half = half

        self.model = Span(gpuid=0, tta_mode=False, model=0)

    def run(self, frame):
        frame = self.model.process_cv2(frame)
        return frame

class RealEsrganNCNN:
    def __init__(
        self,
        upscaleFactor: int = 2,
    ):
        self.upscaleFactor = upscaleFactor
        """
        # 0: {"param": "realesr-animevideov3-x2.param", "bin": "realesr-animevideov3-x2.bin", "scale": 2},
        # 1: {"param": "realesr-animevideov3-x3.param", "bin": "realesr-animevideov3-x3.bin", "scale": 3},
        # 2: {"param": "realesr-animevideov3-x4.param", "bin": "realesr-animevideov3-x4.bin", "scale": 4},
        # 3: {"param": "realesrgan-x4plus-anime.param", "bin": "realesrgan-x4plus-anime.bin", "scale": 4},
        # 4: {"param": "realesrgan-x4plus.param", "bin": "realesrgan-x4plus.bin", "scale": 4}
        """
        modelChooserMap = {
            2: 0,
            3: 1,
            4: 2,
        }

        modelChooser = modelChooserMap.get(self.upscaleFactor)
        self.model = renp.Realesrgan(
            gpuid=0,
            tta_mode=False,
            tilesize=0,
            model=modelChooser,
        )

    def run(self, frame):
        return self.model.process_cv2(frame)
