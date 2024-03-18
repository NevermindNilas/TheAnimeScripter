import cv2
from skimage.metrics import structural_similarity as ssim

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
        frame = cv2.resize(frame, (self.sampleSize, self.sampleSize))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prevFrame is None:
            self.prevFrame = frame.copy()
            return False

        score = ssim(self.prevFrame, frame)
        self.prevFrame = frame.copy()

        return score > self.ssimThreshold
