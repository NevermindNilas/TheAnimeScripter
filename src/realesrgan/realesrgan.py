import realesrgan_ncnn_py as rrs

class RealEsrganNCNN():
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
        self.model = rrs.Realesrgan(
            gpuid = 0,
            tta_mode = False,
            tilesize= 0,
            model = modelChooser,
        )
        
    def run(self, frame):
        return self.model.process_cv2(frame)