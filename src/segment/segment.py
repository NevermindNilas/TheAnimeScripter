import numpy as np
import logging
import os
import cv2
import torch

from src.downloadModels import downloadModels, weightsDir
from concurrent.futures import ThreadPoolExecutor
from .train import AnimeSegmentation
from src.ffmpegSettings import BuildBuffer, WriteBuffer


class Segment:
    def __init__(
        self,
        input,
        output,
        ffmpeg_path,
        width,
        height,
        fps,
        inpoint=0,
        outpoint=0,
        encode_method="x264",
        custom_encoder="",
        nt=1,
        buffer_limit = 50,
    ):
        self.input = input
        self.output = output
        self.ffmpeg_path = ffmpeg_path
        self.width = width
        self.height = height
        self.fps = fps
        self.inpoint = inpoint
        self.outpoint = outpoint
        self.encode_method = encode_method
        self.custom_encoder = custom_encoder
        self.nt = nt
        self.buffer_limit = buffer_limit

        self.handleModel()

        self.readBuffer = BuildBuffer(
            input=self.input,
            ffmpegPath=self.ffmpeg_path,
            inpoint=self.inpoint,
            outpoint=self.outpoint,
            dedup=False,
            dedupSens=None,
            width=self.width,
            height=self.height,
            resize=False,
            resizeMethod=None,
            queueSize=self.buffer_limit,
        )

        self.writeBuffer = WriteBuffer(
            self.output,
            self.ffmpeg_path,
            self.encode_method,
            self.custom_encoder,
            self.width,
            self.height,
            self.fps,
            self.buffer_limit,
            sharpen=False,
            sharpen_sens=None,
            grayscale=False,
            transparent=True,
        )

        with ThreadPoolExecutor(max_workers=3) as executor:
            executor.submit(self.readBuffer.start, verbose=True)
            executor.submit(self.process)
            executor.submit(self.writeBuffer.start, verbose=True)

    def handleModel(self):
        filename = "isnetis.ckpt"
        if not os.path.exists(os.path.join(weightsDir, "segment", filename)):
            model_path = downloadModels(model="segment")
        else:
            model_path = os.path.join(weightsDir, "segment", filename)

        if torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        self.model = AnimeSegmentation.try_load(
            "isnet_is", model_path, self.device, img_size=1024
        )
        self.model.eval()
        self.model.to(self.device)

    """
    def get_character_bounding_box(self, image) -> tuple:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        boxes = [cv2.boundingRect(c) for c in contours]
        x = min([b[0] for b in boxes])
        y = min([b[1] for b in boxes])
        w = max([b[0] + b[2] for b in boxes]) - x
        h = max([b[1] + b[3] for b in boxes]) - y

        return x, y, w, h
    """

    def get_mask(self, input_img):
        s = 1024
        input_img = (input_img / 255).astype(np.float32)
        h, w = h0, w0 = input_img.shape[:-1]
        h, w = (s, int(s * w / h)) if h > w else (int(s * h / w), s)
        ph, pw = s - h, s - w
        img_input = np.zeros([s, s, 3], dtype=np.float32)
        img_input[ph // 2 : ph // 2 + h, pw // 2 : pw // 2 + w] = cv2.resize(
            input_img, (w, h)
        )
        img_input = np.transpose(img_input, (2, 0, 1))
        img_input = img_input[np.newaxis, :]
        tmpImg = torch.from_numpy(img_input).type(torch.FloatTensor).to(self.device)
        with torch.no_grad():
            pred = self.model(tmpImg)
            pred = pred.cpu().numpy()[0]
            pred = np.transpose(pred, (1, 2, 0))
            pred = pred[ph // 2 : ph // 2 + h, pw // 2 : pw // 2 + w]
            pred = cv2.resize(pred, (w0, h0))[:, :, np.newaxis]
            return pred

    def processFrame(self, frame):
        try:
            mask = self.get_mask(frame)
            mask = (mask * 255).astype(np.uint8)
            mask = np.squeeze(mask, axis=2)
            frame_with_mask = np.concatenate((frame, mask[..., np.newaxis]), axis=2)

            self.writeBuffer.write(frame_with_mask)
        except Exception as e:
            logging.exception(f"An error occurred while processing the frame, {e}")

    def process(self):
        frameCount = 0
        with ThreadPoolExecutor(max_workers=self.nt) as executor:
            while True:
                if self.readBuffer.getSizeOfQueue() < self.buffer_limit:
                    frame = self.readBuffer.read()
                    if frame is None:
                        if (
                            self.readBuffer.isReadingDone()
                            and self.readBuffer.getSizeOfQueue() == 0
                        ):
                            break

                    executor.submit(self.processFrame, frame)
                    frameCount += 1

        logging.info(f"Processed {frameCount} frames")

        self.writeBuffer.close()
