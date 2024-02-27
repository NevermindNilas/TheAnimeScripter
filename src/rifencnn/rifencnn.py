from rife_ncnn_vulkan_python import Rife

from PIL import Image

"""
Models:

rife-v4.6
rife-v4.13-lite
rife-v4.14-lite
rife-v4.14

Barebones until I figure out how this exactly works
"""


class rifeNCNN:
    def __init__(self, UHD, interpolate_method, ensemble=False):
        self.UHD = UHD
        self.interpolate_method = interpolate_method

        match self.interpolate_method:
            case "rife-ncnn" | "rife4.14-ncnn":
                self.interpolate_method = "rife-v4.14"
            case "rife4.14-lite-ncnn":
                self.interpolate_method = "rife-v4.14-lite"
            case "rife4.13-lite-ncnn":
                self.interpolate_method = "rife-v4.13-lite"
            case "rife4.6-ncnn":
                self.interpolate_method = "rife-v4.6"

        if ensemble:
            self.interpolate_method += "-ensemble"

        self.rife = Rife(
            gpuid=0,
            model=self.interpolate_method,
            scale=2,
            tta_mode=False,
            tta_temporal_mode=False,
            uhd_mode=self.UHD,
            num_threads=1,
        )
        
        self.frame1 = None
        
    def make_inference(self, timestep):
        output = self.rife.process(self.frame1, self.frame2, timestep=timestep)

        return output

    def cacheFrame(self):
        self.frame1 = self.frame2.copy()
        
    def run(self, frame):
        if self.frame1 is None:
            self.frame1 = Image.fromarray(frame)
            return False
        
        self.frame2 = Image.fromarray(frame)
        
        return True
