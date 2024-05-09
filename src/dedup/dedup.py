import cv2

class DedupSSIMCuda:
    def __init__(
        self,
        ssimThreshold=0.9,
        sampleSize=224,
        half=True,
    ):  
        """
        A Cuda accelerated version of the SSIM deduplication method

        Args:
            ssimThreshold: float, SSIM threshold to consider two frames as duplicates
            sampleSize: int, size of the frame to be used for comparison
            half: bool, use half precision for the comparison
        """
        self.ssimThreshold = ssimThreshold
        self.sampleSize = sampleSize
        self.prevFrame = None
        self.half = half

        self.sampleSize = 224 #forcing 224 for now since performance seems to be degraded the lower I go with this. Somewhere in there is a sweet spot I'm sure.

        import torch
        import torch.nn.functional as F
        from torchmetrics.image import StructuralSimilarityIndexMeasure

        self.torch = torch
        self.F = F
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.ssim = StructuralSimilarityIndexMeasure(data_range=255).to(self.DEVICE)

    def run(self, frame):
        """
        Returns True if the frames are duplicates
        """
        if self.prevFrame is None:
            self.prevFrame = self.processFrame(frame)
            return False

        frame = self.processFrame(frame)

        self.ssim.update(self.prevFrame, frame)
        self.prevFrame = frame.clone()

        return self.ssim.compute() > self.ssimThreshold

    def processFrame(self, frame):
        frame = self.torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).to(self.DEVICE)
        frame = frame.half() if self.half else frame.float()
        frame = self.F.interpolate(frame, (self.sampleSize, self.sampleSize), mode="nearest")
        return frame

#class DedupMSECuda:
#    def __init__(
#        self,
#        mseThreshold=1000,
#        sampleSize=224,
#        half=True,
#    ):
#        """
#        A Cuda accelerated version of the MSE deduplication method
#
#        Args:
#            mseThreshold: float, MSE threshold to consider two frames as duplicates
#            sampleSize: int, size of the frame to be used for comparison
#            half: bool, use half precision for the comparison
#        """
#        self.mseThreshold = mseThreshold
#        self.sampleSize = sampleSize
#        self.prevFrame = None
#        self.half = half
#
#        import torch
#        import torch.nn.functional as F
#        from torchmetrics.image import RootMeanSquaredErrorUsingSlidingWindow
#
#        self.torch = torch
#        self.F = F
#        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#        self.mse = RootMeanSquaredErrorUsingSlidingWindow(window_size=8).to(self.DEVICE)
#
#    def run(self, frame):
#        """
#        Returns True if the frames are duplicates
#        """
#        if self.prevFrame is None:
#            self.prevFrame = self.processFrame(frame)
#            return False
#
#        frame = self.processFrame(frame)
#        self.prevFrame = frame.clone()
#
#        return self.F.mse_loss(self.prevFrame, frame) < self.mseThreshold
#    
#    def processFrame(self, frame):
#        frame = self.torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).to(self.DEVICE)
#        frame = frame.half() if self.half else frame.float()
#        frame = self.F.interpolate(frame, (self.sampleSize, self.sampleSize), mode="nearest")
#        return frame
#                                   
            

class DedupSSIM:
    def __init__(
        self,
        ssimThreshold=0.9,
        sampleSize=32,
    ):
        self.ssimThreshold = ssimThreshold
        self.sampleSize = sampleSize
        self.prevFrame = None

        from skimage.metrics import structural_similarity as ssim

        self.ssim = ssim
    def run(self, frame):
        """
        Returns True if the frames are duplicates
        """
        if self.prevFrame is None:
            self.prevFrame = self.processFrame(frame)
            return False

        frame = self.processFrame(frame)

        score = self.ssim(self.prevFrame, frame)
        self.prevFrame = frame.copy()

        return score > self.ssimThreshold

    def processFrame(self, frame):
        frame = cv2.resize(frame, (self.sampleSize, self.sampleSize))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return frame


class DedupMSE:
    def __init__(
        self,
        mseThreshold=1000,
        sampleSize=32,
    ):
        self.mseThreshold = mseThreshold
        self.sampleSize = sampleSize
        self.prevFrame = None

        from skimage.metrics import mean_squared_error as mse

        self.mse = mse
    def run(self, frame):
        """
        Returns True if the frames are duplicates
        """
        if self.prevFrame is None:
            self.prevFrame = self.processFrame(frame)
            return False

        frame = self.processFrame(frame)

        score = self.mse(self.prevFrame, frame)
        self.prevFrame = frame.copy()

        return score < self.mseThreshold

    def processFrame(self, frame):
        frame = cv2.resize(frame, (self.sampleSize, self.sampleSize))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return frame
