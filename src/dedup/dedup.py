import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr


class DedupSSIM:
    def __init__(
        self,
        ssimThreshold=0.9,
        sampleSize=32,
    ):
        self.ssimThreshold = ssimThreshold
        self.sampleSize = sampleSize
        self.prevFrame = None
        
    def run(self, frame):
        """
        Returns True if the frames are duplicates
        """
        if self.prevFrame is None:
            self.prevFrame = cv2.resize(frame, (self.sampleSize, self.sampleSize))
            self.prevFrame = cv2.cvtColor(self.prevFrame, cv2.COLOR_BGR2GRAY)
            return False
        
        frame = cv2.resize(frame, (self.sampleSize, self.sampleSize))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        score = ssim(self.prevFrame, frame)

        if self.prevFrame is not None:
            self.prevFrame = frame
            
        return score > self.ssimThreshold


class DedupPSNR:
    def __init__(
        self,
        psnrThreshold=30,
        sampleSize=32,
    ):
        self.psnrThreshold = psnrThreshold
        self.sampleSize = sampleSize

    def run(self, prevFrame, frame):
        """
        Returns True if the frames are duplicates
        """
        prevFrame = cv2.resize(prevFrame, (self.sampleSize, self.sampleSize))
        frame = cv2.resize(frame, (self.sampleSize, self.sampleSize))

        prevFrame = cv2.cvtColor(prevFrame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        score = psnr(prevFrame, frame)

        return score > self.psnrThreshold


class DedupMSE:
    def __init__(
        self,
        mseThreshold=0.9,
        sampleSize=32,
    ):
        self.mseThreshold = mseThreshold
        self.sampleSize = sampleSize

    def run(self, prevFrame, frame):
        """
        Returns True if the frames are duplicates
        """
        prevFrame = cv2.resize(prevFrame, (self.sampleSize, self.sampleSize))
        frame = cv2.resize(frame, (self.sampleSize, self.sampleSize))

        prevFrame = cv2.cvtColor(prevFrame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        score = mse(prevFrame, frame)

        return score < self.mseThreshold