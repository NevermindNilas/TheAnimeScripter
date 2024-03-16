from rife_ncnn_vulkan_python import Rife

class rifeNCNN:
    def __init__(self, UHD, interpolate_method, ensemble=False):
        self.UHD = UHD
        self.interpolate_method = interpolate_method

        # Since the built in models folder use the default naming without
        # Ncnn or other suffixes, we need to change the name to match the
        # folder name 
        match self.interpolate_method:
            case "rife-ncnn" | "rife4.15-ncnn":
                self.interpolate_method = "rife-v4.15"            
            case "rife4.14-ncnn":
                self.interpolate_method = "rife-v4.14"
            case "rife4.13-lite-ncnn":
                self.interpolate_method = "rife-v4.13-lite"
            case "rife4.6-ncnn":
                self.interpolate_method = "rife-v4.6"

        # Add ensemble suffix if needed, lowers performance but can improve
        # quality
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
            self.frame1 = frame
            return False
        
        self.frame2 = frame
        return True
