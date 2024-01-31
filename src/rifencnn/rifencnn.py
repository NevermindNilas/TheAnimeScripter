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

class rifeNCNN():
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

        self.rife = Rife(gpuid=0, model=self.interpolate_method, scale=2,
                         tta_mode=False, tta_temporal_mode=ensemble, uhd_mode=self.UHD, num_threads=1)

    def make_inference(self, timestep):
        output = self.rife.process(self.frame1, self.frame2, timestep=timestep)

        return output

    def run(self, prev_frame, frame):
        self.frame1 = Image.fromarray(prev_frame)
        self.frame2 = Image.fromarray(frame)
