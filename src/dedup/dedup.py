import cv2
from skimage.metrics import structural_similarity as ssim, mean_squared_error as mse


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
            self.prevFrame = self.processFrame(frame)
            return False

        frame = self.processFrame(frame)

        score = ssim(self.prevFrame, frame)
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

    def run(self, frame):
        """
        Returns True if the frames are duplicates
        """
        if self.prevFrame is None:
            self.prevFrame = self.processFrame(frame)
            return False

        frame = self.processFrame(frame)

        score = mse(self.prevFrame, frame)
        self.prevFrame = frame.copy()

        return score < self.mseThreshold

    def processFrame(self, frame):
        frame = cv2.resize(frame, (self.sampleSize, self.sampleSize))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return frame
